//! Context Studio addition (S603): compositor presentation-time exposure.
//!
//! Split into this separate, additive file so the upstream-owned `window.rs`
//! carries only the irreducible hook — one struct field, its ctor init, this
//! `mod` line, and a single delegating call in `on_request_frame`. Everything
//! CS-specific (the sticky record logic + the public accessor) lives here,
//! giving upstream pulls a near-zero conflict surface. Behavior is byte-for-byte
//! identical to the original inline version; this is a structural refactor only.
//!
//! Consumer: cs-ui `conversation/view.rs` scroll animation reads
//! `Window::last_presentation_time_nanos()` to drive dt from real present cadence
//! instead of callback-fire wall-clock (fixes the S603 "steppy scroll").

use std::cell::Cell;

use crate::{RequestFrameOptions, Window};

/// Sticky-record the compositor's VSync presentation time from a frame request.
///
/// STICKY / last-known: the Wayland `Presented` feedback is async and
/// `RequestFrameOptions.presentation_time_nanos` is frequently `None` between
/// events (the backend `.take()`s it on each `frame()`), so overwriting
/// unconditionally would discard a good timestamp on the very next request and
/// defeat the consumer's `present[n]-present[n-1]` delta. Only advance on a real
/// `Some`; keep the last known otherwise. Called once per `on_request_frame`,
/// even on throttled frames.
pub(crate) fn cs_record_presentation_time(
    cell: &Cell<Option<u64>>,
    options: &RequestFrameOptions,
) {
    if let Some(nanos) = options.presentation_time_nanos {
        cell.set(Some(nanos));
    }
}

impl Window {
    /// The compositor's VSync presentation time of the most recently presented
    /// frame the platform reported, in monotonic nanoseconds; `None` only until
    /// the first present, or on platforms that don't report it (non-Wayland, or
    /// `wp_presentation_time` unsupported). Sticky / last-known — see
    /// [`cs_record_presentation_time`]. Lets animation code compute frame dt from
    /// real present cadence instead of the wall-clock at callback-fire time.
    pub fn last_presentation_time_nanos(&self) -> Option<u64> {
        self.last_presentation_time_nanos.get()
    }
}
