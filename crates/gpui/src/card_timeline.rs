//! Per-frame diagnostic for DiagnosticCard texture capture.
//!
//! Writes structured events to `/tmp/cs-card-timeline.log`.
//! Shared across crates — gpui, gpui_wgpu, gpui-component, cs-ui all call into here.
//! Frame counter bumped by list.rs paint loop; read by all instrumentation points.
//!
//! TEMP: Remove after P5.1 Phase B debugging complete.

use std::collections::HashSet;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

static FRAME: AtomicU64 = AtomicU64::new(0);
static WATCHED_SLOTS: LazyLock<Mutex<HashSet<usize>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));
static LAST_PRIMITIVE_COUNT: AtomicU64 = AtomicU64::new(u64::MAX);
static CAPTURE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
/// Counter for primitives dropped by content_mask clipping in insert_primitive.
/// Reset before element.paint(), read after to get per-element clip-drop count.
static CLIP_DROP_COUNT: AtomicU64 = AtomicU64::new(0);
/// Whether clip-drop counting is active (only during watched item MISS paint).
static CLIP_DROP_ACTIVE: AtomicU64 = AtomicU64::new(0);

/// Increment the global frame counter. Called once per list paint pass.
/// Returns the new frame number.
///
/// NOTE: Does NOT clear the watched set — that must happen before layout,
/// not at paint start. See `clear_watched_slots()`.
pub fn bump_frame() -> u64 {
    FRAME.fetch_add(1, Ordering::Relaxed) + 1
}

/// Current frame number.
pub fn frame() -> u64 {
    FRAME.load(Ordering::Relaxed)
}

/// Add a list slot index to the watched set (called by view.rs for each DiagnosticCard).
/// `slot_index` = real_ix + LIST_SPACER_OFFSET (what list.rs/texture_cache.rs see).
/// Set is cleared each frame by `bump_frame()`.
pub fn add_watched_slot(slot_index: usize) {
    if let Ok(mut set) = WATCHED_SLOTS.lock() {
        set.insert(slot_index);
    }
}

/// Clear the watched slot set. Must be called before layout phase (before render_item
/// callbacks populate it), NOT during paint. Layout populates → paint reads.
pub fn clear_watched_slots() {
    if let Ok(mut set) = WATCHED_SLOTS.lock() {
        set.clear();
    }
}

/// Check if this slot is in the watched set.
pub fn is_watched(slot_index: usize) -> bool {
    WATCHED_SLOTS
        .lock()
        .map(|set| set.contains(&slot_index))
        .unwrap_or(false)
}

/// Returns true if a PNG dump should be triggered for this capture.
/// Triggers on: first capture (prev == MAX) or primitive count change.
/// NOTE: Has side effects — updates LAST_PRIMITIVE_COUNT on every call.
pub fn should_dump_capture(primitive_count: u32) -> bool {
    let prev = LAST_PRIMITIVE_COUNT.swap(primitive_count as u64, Ordering::Relaxed);
    prev == u64::MAX || prev as u32 != primitive_count
}

/// Monotonic sequence number for PNG filenames.
pub fn next_capture_sequence() -> u64 {
    CAPTURE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
}

/// Append a structured event line to /tmp/cs-card-timeline.log.
pub fn log_event(event: &str) {
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/cs-card-timeline.log")
    {
        let _ = writeln!(f, "[frame={}] {}", FRAME.load(Ordering::Relaxed), event);
    }
}

/// Start counting clip-dropped primitives. Reset counter and activate.
pub fn begin_clip_drop_count() {
    CLIP_DROP_COUNT.store(0, Ordering::Relaxed);
    CLIP_DROP_ACTIVE.store(1, Ordering::Relaxed);
}

/// Stop counting and return the number of primitives dropped by clipping.
pub fn end_clip_drop_count() -> u64 {
    CLIP_DROP_ACTIVE.store(0, Ordering::Relaxed);
    CLIP_DROP_COUNT.load(Ordering::Relaxed)
}

/// Called by scene.rs insert_primitive when a primitive is dropped by clip check.
/// Only increments when clip-drop counting is active.
#[inline]
pub fn record_clip_drop() {
    if CLIP_DROP_ACTIVE.load(Ordering::Relaxed) != 0 {
        CLIP_DROP_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

/// Truncate the log file and write a startup marker.
/// Called once at application init to separate runs.
pub fn init_log() {
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("/tmp/cs-card-timeline.log")
    {
        let _ = writeln!(f, "[startup] card_timeline initialized (multi-slot tracking)");
    }
}
