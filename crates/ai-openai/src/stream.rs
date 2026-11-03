//! Reading a server-sent-events stream across arbitrary network chunks.
//!
//! Everything here is pure (no I/O): jobs.rs owns the connection and feeds bytes in. The reply
//! is bounded as it grows. No line or event longer than [`MAX_SSE_LINE`] is accepted, and the total
//! kept - every text delta plus every tool call's name and arguments - may not pass
//! [`MAX_STREAM_TOTAL`], the same size as the whole-reply cap in `http::MAX_BODY`. A server
//! that streams past either is refused rather than allowed to grow the reply until the
//! request timeout.

use craft_ai_core::{AiError, ErrorKind, Usage};
use serde_json::Value;
use std::time::Duration;

use crate::chat::{Completion, MAX_TOOL_CALLS, error_text, usage_from};

/// Largest single SSE line or event accepted, in bytes.
pub(crate) const MAX_SSE_LINE: usize = 1024 * 1024;
/// Largest total a stream may keep: every text delta plus every tool call's name and
/// arguments. The same size as the whole-reply cap (`http::MAX_BODY`), so a stream can never
/// hold more than a whole reply would; a server that streams past it is refused rather than
/// allowed to grow the reply until the request timeout.
pub(crate) const MAX_STREAM_TOTAL: usize = crate::http::MAX_BODY;

/// Reads a server-sent-events stream across arbitrary chunks.
#[derive(Debug)]
pub(crate) struct SseState {
    /// All text so far (needed for the structured check at the end).
    pub text: String,
    tools: Vec<(String, String)>,
    pub usage: Option<Usage>,
    /// `[DONE]` was seen.
    pub done: bool,
    /// A `finish_reason` was seen.
    pub finished: bool,
    /// Largest total the stream may keep before it is refused.
    cap: usize,
    line_buffer: Vec<u8>,
    data: String,
    event_type: String,
    pub last_event_type: String,
    pub last_event_id: String,
    pub retry: Option<Duration>,
    after_cr: bool,
}

impl Default for SseState {
    /// The full-size cap; only a test shrinks it.
    fn default() -> Self {
        Self {
            text: String::new(),
            tools: Vec::new(),
            usage: None,
            done: false,
            finished: false,
            cap: MAX_STREAM_TOTAL,
            line_buffer: Vec::new(),
            data: String::new(),
            event_type: String::new(),
            last_event_type: String::new(),
            last_event_id: String::new(),
            retry: None,
            after_cr: false,
        }
    }
}

impl SseState {
    /// Shrinks the total cap so a test can drive a stream past it cheaply.
    #[cfg(test)]
    pub(crate) fn with_cap(mut self, cap: usize) -> Self {
        self.cap = cap;
        self
    }

    /// Bytes kept from the stream so far: the text plus every tool call's name and arguments.
    fn kept(&self) -> usize {
        self.text.len()
            + self
                .tools
                .iter()
                .map(|(name, args)| name.len() + args.len())
                .sum::<usize>()
    }

    /// Feeds bytes and returns text deltas from complete, blank-line-delimited events.
    pub fn feed(&mut self, bytes: &[u8]) -> Result<Vec<String>, AiError> {
        let mut deltas = Vec::new();
        for &byte in bytes {
            if self.after_cr {
                self.after_cr = false;
                if byte == b'\n' {
                    continue;
                }
            }
            match byte {
                b'\r' | b'\n' => {
                    if let Some(delta) = self.process_line()? {
                        deltas.push(delta);
                    }
                    self.after_cr = byte == b'\r';
                }
                _ => {
                    if self.line_buffer.len() >= MAX_SSE_LINE {
                        return Err(AiError::new(
                            ErrorKind::Provider,
                            "the stream held a line that is too long",
                        ));
                    }
                    self.line_buffer.push(byte);
                }
            }
            if self.done {
                break;
            }
        }
        Ok(deltas)
    }

    /// EOF discards an event that has not reached a blank line.
    pub fn finish(&mut self) -> Result<Vec<String>, AiError> {
        self.line_buffer.clear();
        self.data.clear();
        self.event_type.clear();
        Ok(Vec::new())
    }

    fn process_line(&mut self) -> Result<Option<String>, AiError> {
        let line = String::from_utf8_lossy(&self.line_buffer).into_owned();
        self.line_buffer.clear();
        if line.is_empty() {
            return self.dispatch();
        }
        if line.starts_with(':') {
            return Ok(None);
        }
        let (field, value) = line.split_once(':').unwrap_or((&line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "data" => {
                if self
                    .data
                    .len()
                    .saturating_add(value.len())
                    .saturating_add(1)
                    > MAX_SSE_LINE
                {
                    return Err(AiError::new(
                        ErrorKind::Provider,
                        "the stream held an event that is too long",
                    ));
                }
                self.data.push_str(value);
                self.data.push('\n');
            }
            "event" => self.event_type = value.to_owned(),
            "id" if !value.contains('\0') => self.last_event_id = value.to_owned(),
            "retry" if !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()) => {
                if let Ok(milliseconds) = value.parse::<u64>() {
                    self.retry = Some(Duration::from_millis(milliseconds));
                }
            }
            _ => {}
        }
        Ok(None)
    }

    fn dispatch(&mut self) -> Result<Option<String>, AiError> {
        if self.data.is_empty() {
            self.event_type.clear();
            return Ok(None);
        }
        self.last_event_type = std::mem::take(&mut self.event_type);
        self.data.pop();
        let data = std::mem::take(&mut self.data);
        if data == "[DONE]" {
            self.done = true;
            return Ok(None);
        }
        let v: Value = serde_json::from_str(&data).map_err(|_| {
            AiError::new(ErrorKind::Provider, "the stream held an unreadable event")
        })?;
        if let Some(e) = v.get("error").filter(|e| !e.is_null()) {
            return Err(AiError::new(ErrorKind::Provider, error_text(e)));
        }
        if let Some(u) = usage_from(&v) {
            self.usage = Some(u);
        }
        let Some(choice) = v.pointer("/choices/0") else {
            return Ok(None);
        };
        if choice.get("finish_reason").is_some_and(|f| !f.is_null()) {
            self.finished = true;
        }
        if let Some(calls) = choice
            .pointer("/delta/tool_calls")
            .and_then(Value::as_array)
        {
            for call in calls {
                self.tool_delta(call);
            }
        }
        let delta = choice
            .pointer("/delta/content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty());
        let out = delta.map(|d| {
            self.text.push_str(d);
            d.to_owned()
        });
        if self.kept() > self.cap {
            return Err(AiError::new(
                ErrorKind::Provider,
                "the streamed reply was too large",
            ));
        }
        Ok(out)
    }

    /// Appends one tool-call fragment to its slot, ignoring a hostile index.
    fn tool_delta(&mut self, call: &Value) {
        let index = call
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|i| usize::try_from(i).ok())
            .unwrap_or(0);
        if index >= MAX_TOOL_CALLS {
            return;
        }
        if self.tools.len() <= index {
            self.tools.resize(index + 1, (String::new(), String::new()));
        }
        if let Some(slot) = self.tools.get_mut(index) {
            if let Some(n) = call.pointer("/function/name").and_then(Value::as_str) {
                slot.0.push_str(n);
            }
            if let Some(a) = call.pointer("/function/arguments").and_then(Value::as_str) {
                slot.1.push_str(a);
            }
        }
    }

    /// The finished reply, once the stream has ended.
    pub fn into_completion(self) -> Completion {
        let tool_calls = self
            .tools
            .into_iter()
            .filter(|(name, _)| !name.is_empty())
            .map(|(n, a)| (n, if a.is_empty() { "{}".to_owned() } else { a }))
            .collect();
        Completion {
            text: self.text,
            tool_calls,
            usage: self.usage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Text, tool calls, usage and the end markers all accumulate from whole lines.
    #[test]
    fn sse_accumulates_text_tools_and_usage() {
        let mut s = SseState::default();
        let lines = [
            ": keep-alive\n\n",
            "event: message\n",
            "id: 7\n",
            "retry: 1000\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
            "data:{\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\":1}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":4}}\n\n",
            "data: [DONE]\n\n",
        ];
        let mut deltas = Vec::new();
        for l in lines {
            deltas.extend(s.feed(l.as_bytes()).unwrap());
        }
        assert_eq!(deltas, ["Hel", "lo"]);
        assert!(s.done && s.finished);
        assert_eq!(s.last_event_type, "");
        assert_eq!(s.last_event_id, "7");
        assert_eq!(s.retry, Some(std::time::Duration::from_millis(1000)));
        let c = s.into_completion();
        assert_eq!(c.text, "Hello");
        assert_eq!(c.tool_calls, [("f".into(), "{\"a\":1}".into())]);
        assert_eq!(c.usage.unwrap().input_units, Some(2));
    }

    /// Broken events, streamed errors and hostile indices are refused or ignored, not panics.
    #[test]
    fn sse_errors_and_garbage_are_errors_not_panics() {
        let mut s = SseState::default();
        assert!(s.feed(b"data: {not json\n").unwrap().is_empty());
        assert!(s.feed(b"\n").is_err());
        let mut s = SseState::default();
        assert!(
            s.feed(b"data: {\"error\":{\"message\":\"overloaded\"}}\n\n")
                .is_err()
        );
        // Hostile indices are ignored rather than allocated.
        s.feed(b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":4000000000,\"function\":{\"name\":\"x\"}}]}}]}\n\n")
            .unwrap();
        assert!(s.into_completion().tool_calls.is_empty());
    }

    /// A stream past its total is refused, not grown; text and tool arguments share one cap.
    #[test]
    fn a_stream_past_its_total_is_refused_not_grown() {
        // The stream may never keep more than a whole reply could hold: the same cap as
        // http::MAX_BODY. Without it a hostile server streams deltas for as long as the
        // request timeout, and the reply grows without end.
        assert_eq!(SseState::default().cap, crate::http::MAX_BODY);
        let mut s = SseState::default().with_cap(10);
        s.feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"0123456789\"}}]}\n\n")
            .unwrap();
        assert_eq!(
            s.feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n")
                .unwrap_err()
                .kind,
            ErrorKind::Provider
        );
        // Tool-call arguments count against the same total, not a separate one.
        let mut s = SseState::default().with_cap(10);
        s.feed(b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"0123456789\"}}]}}]}\n\n")
            .unwrap();
        assert!(
            s.feed(b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"x\"}}]}}]}\n\n")
                .is_err()
        );
    }

    #[test]
    fn multiline_data_waits_for_boundary_across_chunks_and_crlf() {
        let mut state = SseState::default();
        assert!(state.feed(b": heartbeat\r\nevent: message\r\nid: first\r\nretry: 12\r\ndata: {\"choices\":[{\"delta\":\r\n").unwrap().is_empty());
        assert!(
            state
                .feed(b"data: {\"content\":\"ok\"}}]}\r")
                .unwrap()
                .is_empty()
        );
        assert_eq!(state.feed(b"\n\r").unwrap(), ["ok"]);
        assert!(state.feed(b"\n").unwrap().is_empty());
        assert_eq!(state.last_event_id, "first");
        assert_eq!(state.last_event_type, "message");
        assert_eq!(state.retry, Some(std::time::Duration::from_millis(12)));
        assert!(state.feed(b"data: [DO").unwrap().is_empty());
        assert!(state.feed(b"NE]\r\n").unwrap().is_empty());
        assert!(!state.done);
        state.feed(b"\r\n").unwrap();
        assert!(state.done);
    }

    #[test]
    fn eof_does_not_dispatch_unterminated_events_and_cr_is_a_boundary() {
        let mut state = SseState::default();
        assert!(state.feed(b"data: [DONE]\n").unwrap().is_empty());
        assert!(!state.done);
        assert!(state.feed(b"\n: keepalive\r\r").unwrap().is_empty());
        assert!(state.done);
        let mut state = SseState::default();
        assert!(state.feed(b"data: [DONE]\n").unwrap().is_empty());
        assert!(state.finish().unwrap().is_empty());
        assert!(!state.done);
    }

    #[test]
    fn an_event_with_many_data_fields_is_bounded() {
        let mut state = SseState::default();
        let first = format!("data: {}\n", "x".repeat(MAX_SSE_LINE - 6));
        assert!(state.feed(first.as_bytes()).unwrap().is_empty());
        let error = state.feed(b"data: more!\n").unwrap_err();
        assert_eq!(error.kind, ErrorKind::Provider);
        assert!(error.message.contains("event that is too long"));
    }
}
