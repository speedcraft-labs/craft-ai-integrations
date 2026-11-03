//! One TLS setting and one proxy policy for every call.
//!
//! Every adapter and downloader builds its own `ureq` agent, so the rules that must hold
//! everywhere live here once instead of in each builder.
//!
//! The TLS rule: trust what the operating system trusts. The alternative, a bundled list of
//! public roots (`ureq`'s default), fails on a machine where antivirus or a company proxy
//! re-signs HTTPS with its own root, which the operating system has installed but a bundled
//! list has not. The launcher hit exactly that. The price is that those installed roots are
//! trusted too, so software the person has chosen to install can read the traffic, API keys
//! included. For a desktop app acting for its user that is the usual rule; a service that
//! cannot accept it should not use this crate.
//!
//! The proxy rule: `ureq`'s default agent picks up `HTTP(S)_PROXY` and `ALL_PROXY` from the
//! environment and sends every request through the host they name — requests to the person's
//! own machine included, unless their own `NO_PROXY` happens to exempt it. Loopback traffic
//! (ComfyUI prompts and images, local model servers) is cleartext the consent architecture
//! promises never leaves the computer, so a loopback agent takes no proxy at all, whatever the
//! environment says. Hosted providers keep the environment's proxy: a person or company that
//! sets a proxy variable has arranged its network that way, and refusing would break those
//! installs.
//!
//! `tests/every_agent_uses_it.rs` fails if any crate builds an agent without taking both
//! settings from here, so a new adapter cannot quietly go back to the bundled list, or to the
//! environment's proxy for its loopback traffic.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use craft_ai_core::endpoint::Endpoint;
use std::time::{Duration, SystemTime};
use ureq::Proxy;
use ureq::tls::{RootCerts, TlsConfig};

/// The TLS setting for an agent: certificates are checked by the platform's own verifier.
///
/// Use it as `ureq::Agent::config_builder().tls_config(craft_ai_http::tls_config())`.
pub fn tls_config() -> TlsConfig {
    TlsConfig::builder()
        .root_certs(RootCerts::PlatformVerifier)
        .build()
}

/// No proxy, whatever the environment says.
///
/// The setting for an agent whose traffic stays on the person's own machine. `ureq`'s default
/// would send even loopback requests to the host a proxy variable names, and that would move
/// cleartext the consent architecture promises never leaves the computer onto a third host.
pub fn loopback_proxy() -> Option<Proxy> {
    None
}

/// The proxy the environment names, if it names one.
///
/// The setting for an agent that talks to hosted providers. A person or company that sets a
/// proxy variable has arranged its network that way, and hosted calls should follow it.
pub fn hosted_proxy() -> Option<Proxy> {
    Proxy::try_from_env()
}

/// The proxy setting for an agent that talks to one base address, decided from the address.
///
/// A loopback base — loopback by [`Endpoint::is_loopback`], the same rule that decides
/// whether a connection needs consent at all — takes [`loopback_proxy`]. Any other base, one
/// that will not parse as an endpoint included, takes [`hosted_proxy`]: only an address proven
/// to be this machine is exempt from the environment's proxy.
///
/// An agent that talks to many addresses (`ai-media`'s downloader, redirect hops included)
/// cannot decide from one base. It must set [`loopback_proxy`] on the agent and override the
/// proxy per request.
pub fn proxy_for(base: &str) -> Option<Proxy> {
    if Endpoint::parse(base).is_ok_and(|endpoint| endpoint.is_loopback()) {
        loopback_proxy()
    } else {
        hosted_proxy()
    }
}

/// Converts a Retry-After delay or IMF-fixdate to a bounded wait at the supplied time.
pub fn parse_retry_after(header: &str, now: SystemTime) -> Option<Duration> {
    const MAX_WAIT: Duration = Duration::from_secs(24 * 60 * 60);
    let value = header.trim();
    if !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Some(Duration::from_secs(value.parse::<u64>().unwrap_or(u64::MAX)).min(MAX_WAIT));
    }
    let date = chrono::NaiveDateTime::parse_from_str(value, "%a, %d %b %Y %H:%M:%S GMT").ok()?;
    if date.format("%a, %d %b %Y %H:%M:%S GMT").to_string() != value {
        return None;
    }
    Some(
        date.and_utc()
            .signed_duration_since(chrono::DateTime::<chrono::Utc>::from(now))
            .to_std()
            .unwrap_or(Duration::ZERO)
            .min(MAX_WAIT),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, UNIX_EPOCH};

    #[test]
    fn retry_after_accepts_seconds_and_imf_fixdates() {
        let now = UNIX_EPOCH + Duration::from_secs(784_111_770);
        assert_eq!(
            parse_retry_after(" 12 ", now),
            Some(Duration::from_secs(12))
        );
        assert_eq!(
            parse_retry_after("Sun, 06 Nov 1994 08:49:50 GMT", now),
            Some(Duration::from_secs(20))
        );
        assert_eq!(
            parse_retry_after("Sun, 06 Nov 1994 08:49:30 GMT", now),
            Some(Duration::ZERO)
        );
        assert_eq!(
            parse_retry_after("86401", now),
            Some(Duration::from_secs(86400))
        );
        assert_eq!(
            parse_retry_after("999999999999999999999999999", now),
            Some(Duration::from_secs(86400))
        );
    }

    #[test]
    fn retry_after_refuses_garbage_and_noncanonical_dates() {
        let now = UNIX_EPOCH;
        for header in [
            "",
            "-1",
            "+1",
            "1.5",
            "tomorrow",
            "Sun, 06 Nov 1994 08:49:50 UTC",
            "Sun, 32 Nov 1994 08:49:50 GMT",
            "Mon, 06 Nov 1994 08:49:50 GMT",
        ] {
            assert_eq!(parse_retry_after(header, now), None, "{header}");
        }
    }

    #[test]
    fn certificates_are_checked_by_the_platform() {
        assert!(matches!(
            tls_config().root_certs(),
            RootCerts::PlatformVerifier
        ));
    }

    #[test]
    fn an_agent_built_with_it_keeps_the_setting() {
        let config = ureq::Agent::config_builder()
            .tls_config(tls_config())
            .build();
        assert!(matches!(
            config.tls_config().root_certs(),
            RootCerts::PlatformVerifier
        ));
    }
}
