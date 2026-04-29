//! S521 cache-telemetry diagnostic surface.
//!
//! Cross-crate instrumentation to answer "do table-containing AgentMessage cards
//! actually reach the HIT path, and if not, why?" Emits four event categories to
//! the cs-debug file tracer (via a callback registered by cs-app at startup):
//!
//! - **Per-card-frame** (Category A): one event per visible card per painted
//!   frame. Captures cache state, layout height, paint cost, admission outcome,
//!   markdown-render fingerprint, table cell count, etc.
//! - **Per-frame summary** (Category B): one event per painted frame. Aggregate
//!   counts and texture-pool occupancy.
//! - **Session-start** (Category C): one event at app start. Captures fork pin,
//!   vendor rev, app commit, and runtime cache constants for ground-truth baseline.
//! - **Per-cell paint** (Category D): one event per painted table cell. Gated by
//!   `is_per_cell_trace_enabled()` (toggled from cs-ui Render Cache sidepane), to
//!   avoid trace-flood when not investigating cell-level cost.
//!
//! ## Architecture
//!
//! - Vendor (gpui-component) marks per-card facts (markdown_render_ran,
//!   state_update_writes, cell_count, has_table) into a thread-local recorder
//!   while painting.
//! - Fork (gpui list.rs paint loop) calls `begin_card` before each
//!   `item.element.paint()` to reset the recorder and `end_card` after to
//!   harvest the snapshot, then emits a `PerCardFrameEvent` via the registered
//!   callback. After the loop, emits one `PerFrameSummaryEvent`.
//! - cs-app registers the callback at startup; the callback forwards each event
//!   to `cs_core::cs_log!(Subsystem::RenderCache, Level::Debug, "event=...")`.
//!
//! ## Zero-overhead in production
//!
//! All emission and recording is gated by `#[cfg(feature = "texture-cache-debug")]`.
//! Public entry points are always present (so callers compile unconditionally) but
//! their bodies compile to empty no-ops without the feature flag, which the
//! optimizer dead-code-eliminates.

#[cfg(feature = "texture-cache-debug")]
use std::cell::{Cell, RefCell};
#[cfg(feature = "texture-cache-debug")]
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
#[cfg(feature = "texture-cache-debug")]
use std::sync::atomic::Ordering;
#[cfg(feature = "texture-cache-debug")]
use std::sync::{LazyLock, Mutex};

// ---------------------------------------------------------------------------
// Event types (always present so consumer signatures compile cross-feature).
// ---------------------------------------------------------------------------

/// Outcome of the most recent admission attempt for this region_id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionOutcome {
    /// Successfully admitted to the texture cache (or already cached).
    Admitted,
    /// Texture proposed dimensions exceeded `max_texture_dimension_2d`. Card stays
    /// on MISS path forever even when stable. The latent failure mode S521 set
    /// out to detect.
    RejectedTooLarge,
    /// Card is in the streaming set; cache bypass per ST1 invariant.
    RejectedStreaming,
    /// Card visible for fewer than `TRANSIENT_SKIP_FRAMES`; capture skipped.
    RejectedTransient,
    /// HIT path — cached entry already present.
    RejectedAlreadyCached,
    /// Splice cleared `visible_frames`; capture skipped this frame.
    RejectedSpliceClear,
    /// `caching_enabled == false` — DIRECT/PLAIN paint with no admission attempt.
    RejectedNotEligible,
}

impl AdmissionOutcome {
    /// Compact string used in NDJSON payloads.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Admitted => "Admitted",
            Self::RejectedTooLarge => "RejectedTooLarge",
            Self::RejectedStreaming => "RejectedStreaming",
            Self::RejectedTransient => "RejectedTransient",
            Self::RejectedAlreadyCached => "RejectedAlreadyCached",
            Self::RejectedSpliceClear => "RejectedSpliceClear",
            Self::RejectedNotEligible => "RejectedNotEligible",
        }
    }
}

/// Cache routing classification for a card on a single painted frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheState {
    Hit,
    Miss,
    Transient,
    Streaming,
    /// `caching_enabled == false` — cards rendered DIRECT, no cache participation.
    Plain,
}

impl CacheState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hit => "HIT",
            Self::Miss => "MISS",
            Self::Transient => "TRANSIENT",
            Self::Streaming => "STREAMING",
            Self::Plain => "PLAIN",
        }
    }
}

/// One event per visible card per painted frame (Category A).
#[derive(Clone, Debug)]
pub struct PerCardFrameEvent {
    pub frame_id: u64,
    pub card_index: usize,
    /// "msg-text" / "tool" / "thinking" / "user" / "agent_status" / "session_header" /
    /// "top_spacer" / "bottom_spacer" / "permission" / "plan" — provided by cs-ui via
    /// the existing `card_type_name()` reflection.
    pub card_type: &'static str,
    /// True if the parsed markdown body contains at least one `BlockNode::Table`.
    /// Always false for non-AgentMessage cards.
    pub has_table: bool,
    pub cache_state: CacheState,
    /// Most-recent attempt outcome from gpui_wgpu (1-frame latency: reflects last
    /// capture attempt). `Admitted` for HIT and successful MISS; `RejectedTooLarge`
    /// for cards that exceeded `max_texture_dimension_2d`; etc.
    pub admission_outcome: AdmissionOutcome,
    pub visible_frames_count: u32,
    /// `-1.0` if no cached_height yet (pre-stabilization).
    pub cached_height: f32,
    /// 0 if no cached texture for this region_id.
    pub texture_size_bytes: u64,
    /// `(width, height)` of the captured (or last-proposed) texture. `(0, 0)` if
    /// no proposal made (e.g. STREAMING). For `RejectedTooLarge`, captures the
    /// proposed dimensions that were rejected so we can identify oversized cards.
    pub texture_dimensions: (u32, u32),
    /// Laid-out height of the card this frame, regardless of cache state.
    pub card_layout_height: f32,
    /// True if `TextViewState::render` (vendor state.rs:359) executed for this
    /// card's TextViewState entity during this card's paint. The smoking-gun
    /// diagnostic: if this is true on every frame for a stable card, the
    /// unconditional `state.update(cx, ...)` writes in `TextView::request_layout`
    /// are triggering per-frame notifies.
    pub markdown_render_ran_this_frame: bool,
    /// Count of `state.update(cx, ...)` calls fired in `TextView::request_layout`
    /// during this card's paint. Always 5 in steady-state code; included for
    /// regression detection.
    pub state_update_writes_this_frame: u32,
    /// Of the `state_update_writes_this_frame` writes, how many actually changed
    /// the stored value (vs no-op rewrite). Combined with
    /// `markdown_render_ran_this_frame`, answers "do unconditional writes
    /// trigger notify?"
    pub state_update_changed_values: u32,
    /// Total cells (rows × cols) for this card if it contains a table; 0 otherwise.
    pub cell_count: u32,
    /// Wall-clock time spent in `item.element.paint()` for this card this frame.
    pub card_paint_ns: u64,
}

/// One event per painted frame (Category B).
#[derive(Clone, Debug)]
pub struct PerFrameSummaryEvent {
    pub frame_id: u64,
    pub total_visible_cards: u32,
    pub cards_in_hit: u32,
    pub cards_in_miss: u32,
    pub cards_in_transient: u32,
    pub cards_in_streaming: u32,
    pub cards_in_plain: u32,
    /// Count of currently-visible cards whose most recent admission was
    /// `RejectedTooLarge` (latent oversized-card failure mode).
    pub cards_rejected_too_large: u32,
    pub total_textures_cached: u32,
    pub total_texture_bytes: u64,
    pub texture_pool_capacity_bytes: u64,
    pub texture_pool_utilization_pct: f32,
    /// Wall-clock total spent in the per-card paint loop this frame (sum of
    /// `card_paint_ns` plus per-card classification/branching overhead).
    pub frame_paint_total_ns: u64,
}

/// One event at app start, before the first frame (Category C).
#[derive(Clone, Debug)]
pub struct SessionStartEvent {
    /// `DEFAULT_BUDGET_BYTES` from the texture pool.
    pub texture_cache_capacity_bytes: u64,
    /// `TRANSIENT_SKIP_FRAMES` constant from list.rs.
    pub transient_skip_frames: u32,
    /// gpui fork rev string at compile time (from `CS_GPUI_PIN_REV` env var stamped
    /// by `cs-app/build.rs`).
    pub gpui_pin_rev: &'static str,
    /// vendor/gpui-component rev string at compile time (from
    /// `CS_GPUI_COMPONENT_VENDOR_REV`).
    pub gpui_component_vendor_rev: &'static str,
    /// CS app commit at compile time (from `CS_APP_COMMIT`).
    pub cs_app_commit: &'static str,
}

/// One event per painted table cell on a MISS or TRANSIENT frame (Category D).
/// Gated by `is_per_cell_trace_enabled()` on top of `texture-cache-debug` feature.
#[derive(Clone, Debug)]
pub struct PerCellPaintEvent {
    pub frame_id: u64,
    pub card_index: usize,
    pub row: u32,
    pub col: u32,
    pub cell_paint_ns: u64,
    pub cell_text_len: u32,
}

// ---------------------------------------------------------------------------
// Callback wiring (mirrors ScrollTelemetry pattern in elements/list.rs).
// ---------------------------------------------------------------------------

/// Aggregated cache-telemetry callback shape. cs-app registers one closure that
/// matches on the inner enum and forwards each variant to its `cs_log!` line.
pub enum CacheTelemetryEvent<'a> {
    PerCardFrame(&'a PerCardFrameEvent),
    PerFrameSummary(&'a PerFrameSummaryEvent),
    SessionStart(&'a SessionStartEvent),
    PerCellPaint(&'a PerCellPaintEvent),
}

#[cfg(feature = "texture-cache-debug")]
type CacheTelemetryCallback = Box<dyn Fn(CacheTelemetryEvent<'_>) + 'static>;

#[cfg(feature = "texture-cache-debug")]
thread_local! {
    /// Registered callback. Set once at app start via `set_cache_telemetry_callback`.
    static CACHE_TELEMETRY_CB: Cell<Option<*const CacheTelemetryCallback>> =
        const { Cell::new(None) };
    /// Per-card recorder reset at `begin_card` and harvested at `end_card`. Allows
    /// vendor (gpui-component) instrumentation to leave fingerprints that fork
    /// (gpui list.rs) reads when emitting the `PerCardFrameEvent` for that card.
    static PER_CARD_RECORDER: RefCell<PerCardRecorder> =
        RefCell::new(PerCardRecorder::default());
}

/// Sticky last-known card-type by index, populated whenever `record_card_type`
/// fires (i.e. when cs-ui's `render_item` runs). Read by `end_card` for HIT
/// frames where the closure is skipped, so HIT-frame events still carry an
/// accurate `card_type` instead of "Unknown".
#[cfg(feature = "texture-cache-debug")]
static LAST_CARD_TYPE: LazyLock<Mutex<HashMap<usize, &'static str>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Atomic gate for per-cell paint events. Toggled at runtime from cs-ui
/// (Render Cache sidepane sub-toggle). OFF by default; ON when Klaus clicks the
/// "Render Cache (per-cell)" entry to investigate cell-level cost.
#[cfg(feature = "texture-cache-debug")]
static PER_CELL_TRACE_ENABLED: AtomicBool = AtomicBool::new(false);

/// Stub when the feature is off — keeps the symbol present so callers compile.
#[cfg(not(feature = "texture-cache-debug"))]
static PER_CELL_TRACE_ENABLED: AtomicBool = AtomicBool::new(false);

/// Internal per-card scratch state, written by vendor instrumentation between
/// `begin_card` and `end_card`. NOTE: `card_type` is NOT in this struct
/// because `record_card_type` is called from cs-ui during PREPAINT (before
/// `begin_card` runs during PAINT), so a per-card field would be cleared by
/// the `begin_card` reset before `end_card` could harvest it. card_type is
/// stored in `LAST_CARD_TYPE` keyed by card_index instead.
#[derive(Clone, Default, Debug)]
pub struct PerCardRecorder {
    /// Card index set by the most recent `begin_card`. `None` when no card paint
    /// is active.
    pub current_card_index: Option<usize>,
    /// True if `TextViewState::render` ran during this card's paint.
    pub markdown_render_ran: bool,
    /// Count of `state.update(cx, ...)` writes attempted in `TextView::request_layout`.
    pub state_update_writes: u32,
    /// Of `state_update_writes`, how many actually changed the stored value.
    pub state_update_changed: u32,
    /// True if `render_table` ran during this card's paint.
    pub has_table: bool,
    /// Sum of cells across all tables painted during this card's paint
    /// (rows × cols per table).
    pub cell_count: u32,
}

/// Snapshot returned by `end_card` for inclusion in `PerCardFrameEvent`.
#[derive(Clone, Debug, Default)]
pub struct PerCardSnapshot {
    /// CS-side card-type label, looked up by card_index from `LAST_CARD_TYPE`
    /// (populated by cs-ui's `record_card_type` during prepaint). Returns
    /// `"Unknown"` if no record exists for this index — only happens for the
    /// first frame after the cache is cleared, since render_item runs every
    /// frame for non-HIT cards and populates the cache.
    pub card_type: &'static str,
    /// True if `TextViewState::render` ran during this card's paint.
    pub markdown_render_ran: bool,
    /// Count of `state.update(cx, ...)` writes attempted.
    pub state_update_writes: u32,
    /// Of those writes, how many actually changed the stored value.
    pub state_update_changed: u32,
    /// True if the body contained at least one markdown table.
    pub has_table: bool,
    /// Sum of cells across all tables painted (rows × cols per table).
    pub cell_count: u32,
}

/// Register the cache-telemetry callback. Call once at app startup. The callback
/// must outlive all paint passes; in practice, register a `Box::leak`'d closure
/// that forwards to the cs-debug file tracer.
pub fn set_cache_telemetry_callback<F>(callback: F)
where
    F: Fn(CacheTelemetryEvent<'_>) + 'static,
{
    #[cfg(feature = "texture-cache-debug")]
    {
        let boxed: CacheTelemetryCallback = Box::new(callback);
        let leaked = Box::leak(Box::new(boxed));
        CACHE_TELEMETRY_CB.with(|cell| {
            cell.set(Some(leaked as *const CacheTelemetryCallback));
        });
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = callback;
    }
}

/// True when per-cell paint events should fire. Both gates must be on:
/// `texture-cache-debug` feature compiled in AND runtime toggle enabled.
#[inline]
pub fn is_per_cell_trace_enabled() -> bool {
    #[cfg(feature = "texture-cache-debug")]
    {
        PER_CELL_TRACE_ENABLED.load(Ordering::Relaxed)
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        false
    }
}

/// Toggle per-cell paint events at runtime. Called from the cs-ui Render Cache
/// sidepane sub-toggle handler.
pub fn set_per_cell_trace_enabled(enabled: bool) {
    #[cfg(feature = "texture-cache-debug")]
    {
        PER_CELL_TRACE_ENABLED.store(enabled, Ordering::Relaxed);
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = enabled;
    }
}

// ---------------------------------------------------------------------------
// Per-card recorder API (called by fork list.rs around each card paint).
// ---------------------------------------------------------------------------

/// Reset the per-card recorder for a new card. Must be called immediately before
/// `item.element.paint()` in the list paint loop.
#[inline]
pub fn begin_card(card_index: usize) {
    #[cfg(feature = "texture-cache-debug")]
    {
        PER_CARD_RECORDER.with(|cell| {
            let mut rec = cell.borrow_mut();
            *rec = PerCardRecorder::default();
            rec.current_card_index = Some(card_index);
        });
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = card_index;
    }
}

/// Harvest and clear the per-card recorder. Call after `item.element.paint()` in
/// the list paint loop. Returns the snapshot for inclusion in the
/// `PerCardFrameEvent`. card_type is looked up from `LAST_CARD_TYPE` (populated
/// by cs-ui's `record_card_type` during prepaint, keyed by card_index).
#[inline]
pub fn end_card() -> PerCardSnapshot {
    #[cfg(feature = "texture-cache-debug")]
    {
        PER_CARD_RECORDER.with(|cell| {
            let mut rec = cell.borrow_mut();
            let card_index = rec.current_card_index;
            let card_type = card_index
                .and_then(|idx| {
                    LAST_CARD_TYPE.lock().ok().and_then(|m| m.get(&idx).copied())
                })
                .unwrap_or("Unknown");
            let snap = PerCardSnapshot {
                card_type,
                markdown_render_ran: rec.markdown_render_ran,
                state_update_writes: rec.state_update_writes,
                state_update_changed: rec.state_update_changed,
                has_table: rec.has_table,
                cell_count: rec.cell_count,
            };
            *rec = PerCardRecorder::default();
            snap
        })
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        PerCardSnapshot::default()
    }
}

/// Record the CS-side card-type label for a card. Called from cs-ui's
/// `render_item` closure (which runs during PREPAINT, BEFORE the fork's
/// `begin_card` resets the per-card recorder during PAINT). Therefore writes
/// directly to the `LAST_CARD_TYPE` cache keyed by card_index — the
/// per-card recorder's `card_type` field can't survive the recorder reset
/// between prepaint and paint. `card_index` must match the value the fork's
/// list.rs paint loop uses for `begin_card` (i.e. CS-UI passes
/// `real_ix + LIST_SPACER_OFFSET`).
#[inline]
pub fn record_card_type(card_index: usize, card_type: &'static str) {
    #[cfg(feature = "texture-cache-debug")]
    {
        if let Ok(mut m) = LAST_CARD_TYPE.lock() {
            m.insert(card_index, card_type);
        }
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = (card_index, card_type);
    }
}

/// Per-card cached-height fed by cs-ui when known (the CS-side `CardState`
/// owns this; it isn't visible from the fork). cs-ui calls
/// `record_cached_height(f32)` inside `render_item` for stable cards;
/// `end_card`'s snapshot reads the value via `last_cached_height(card_index)`
/// (sticky cache, like card_type, so HIT frames still carry it).
#[cfg(feature = "texture-cache-debug")]
static LAST_CACHED_HEIGHT: LazyLock<Mutex<HashMap<usize, f32>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Record the CS-side cached_height for the current card (the value
/// `view.rs::CardState::cached_height` was set to by `stabilize_card`). Pass
/// `-1.0` for "not stabilized yet" so the cache stays sticky across HIT frames.
#[inline]
pub fn record_cached_height(card_index: usize, height: f32) {
    #[cfg(feature = "texture-cache-debug")]
    {
        if let Ok(mut m) = LAST_CACHED_HEIGHT.lock() {
            m.insert(card_index, height);
        }
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = (card_index, height);
    }
}

/// Look up the most recent cached_height for a card index. Returns `-1.0` when
/// no value has ever been recorded.
#[inline]
pub fn last_cached_height(card_index: usize) -> f32 {
    #[cfg(feature = "texture-cache-debug")]
    {
        LAST_CACHED_HEIGHT
            .lock()
            .ok()
            .and_then(|m| m.get(&card_index).copied())
            .unwrap_or(-1.0)
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = card_index;
        -1.0
    }
}

// ---------------------------------------------------------------------------
// Vendor-facing record API (called by gpui-component).
// ---------------------------------------------------------------------------

/// Mark that `TextViewState::render` ran this frame for the current card. Called
/// at the entry of vendor `state.rs` `TextViewState::render`.
#[inline]
pub fn record_markdown_render_ran() {
    #[cfg(feature = "texture-cache-debug")]
    {
        PER_CARD_RECORDER.with(|cell| {
            cell.borrow_mut().markdown_render_ran = true;
        });
    }
}

/// Record one of the unconditional `state.update(cx, ...)` writes in vendor
/// `text_view.rs::TextView::request_layout`. `changed = true` iff the prior value
/// differed from the incoming value (compute that via `state.read(cx)` BEFORE
/// calling `state.update`).
#[inline]
pub fn record_state_update(changed: bool) {
    #[cfg(feature = "texture-cache-debug")]
    {
        PER_CARD_RECORDER.with(|cell| {
            let mut rec = cell.borrow_mut();
            rec.state_update_writes = rec.state_update_writes.saturating_add(1);
            if changed {
                rec.state_update_changed = rec.state_update_changed.saturating_add(1);
            }
        });
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = changed;
    }
}

/// Record that the current card's body contains a markdown table. Called from
/// vendor `node.rs::render_table` at function entry. `cell_count = rows × cols`.
#[inline]
pub fn record_table(cell_count: u32) {
    #[cfg(feature = "texture-cache-debug")]
    {
        PER_CARD_RECORDER.with(|cell| {
            let mut rec = cell.borrow_mut();
            rec.has_table = true;
            // Multiple tables per card: accumulate cells across all of them.
            rec.cell_count = rec.cell_count.saturating_add(cell_count);
        });
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = cell_count;
    }
}

/// Get the current card index (set by the most recent `begin_card`). None when
/// no card paint is active. Used by vendor instrumentation that needs to emit
/// per-cell events without explicit threading.
#[inline]
pub fn current_card_index() -> Option<usize> {
    #[cfg(feature = "texture-cache-debug")]
    {
        PER_CARD_RECORDER.with(|cell| cell.borrow().current_card_index)
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        None
    }
}

// ---------------------------------------------------------------------------
// Emission API (called by fork list.rs and cs-app session-start).
// ---------------------------------------------------------------------------

/// Emit a per-card-frame event to the registered callback.
#[inline]
pub fn emit_per_card_frame(event: &PerCardFrameEvent) {
    #[cfg(feature = "texture-cache-debug")]
    {
        with_callback(|cb| cb(CacheTelemetryEvent::PerCardFrame(event)));
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = event;
    }
}

/// Emit a per-frame summary event to the registered callback.
#[inline]
pub fn emit_per_frame_summary(event: &PerFrameSummaryEvent) {
    #[cfg(feature = "texture-cache-debug")]
    {
        with_callback(|cb| cb(CacheTelemetryEvent::PerFrameSummary(event)));
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = event;
    }
}

/// Emit a session-start event to the registered callback.
#[inline]
pub fn emit_session_start(event: &SessionStartEvent) {
    #[cfg(feature = "texture-cache-debug")]
    {
        with_callback(|cb| cb(CacheTelemetryEvent::SessionStart(event)));
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = event;
    }
}

/// Emit a per-cell paint event to the registered callback. Always check
/// `is_per_cell_trace_enabled()` BEFORE constructing the event payload to avoid
/// allocation when the toggle is off.
#[inline]
pub fn emit_per_cell_paint(event: &PerCellPaintEvent) {
    #[cfg(feature = "texture-cache-debug")]
    {
        with_callback(|cb| cb(CacheTelemetryEvent::PerCellPaint(event)));
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = event;
    }
}

// ---------------------------------------------------------------------------
// Cross-crate admission + pool registry (gpui_wgpu writes, list.rs reads).
//
// gpui_wgpu's `TexturePool` is `pub(crate)` to its crate, so direct accessors
// are not available. Instead, gpui_wgpu publishes admission-outcome and texture
// size facts into this thread-safe registry (mirrors `card_timeline.rs`'s
// LazyLock<Mutex<...>> style). list.rs reads at per-card-frame emit time and
// at end-of-frame summary emit time.
// ---------------------------------------------------------------------------

/// Admission state recorded by gpui_wgpu for a region_id.
#[derive(Clone, Copy, Debug)]
pub struct AdmissionRecord {
    /// Outcome of the most recent admission attempt.
    pub outcome: AdmissionOutcome,
    /// Proposed (or captured) texture dimensions. `(0, 0)` if no proposal made.
    pub proposed_dimensions: (u32, u32),
}

/// Per-region cached-texture facts read by list.rs at per-card-frame emit time.
#[derive(Clone, Copy, Debug, Default)]
pub struct TextureSizeRecord {
    /// Memory cost in bytes for this region_id's cached texture (0 if uncached).
    pub bytes: u64,
    /// Captured dimensions in pixels (0×0 if uncached).
    pub width: u32,
    pub height: u32,
}

/// Pool-wide aggregates read by list.rs at end-of-frame summary emit time.
#[derive(Clone, Copy, Debug, Default)]
pub struct TexturePoolSummary {
    pub total_textures_cached: u32,
    pub total_texture_bytes: u64,
    pub capacity_bytes: u64,
}

#[cfg(feature = "texture-cache-debug")]
static ADMISSION_REGISTRY: LazyLock<Mutex<HashMap<u64, AdmissionRecord>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[cfg(feature = "texture-cache-debug")]
static TEXTURE_SIZE_REGISTRY: LazyLock<Mutex<HashMap<u64, TextureSizeRecord>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[cfg(feature = "texture-cache-debug")]
static POOL_SUMMARY: LazyLock<Mutex<TexturePoolSummary>> =
    LazyLock::new(|| Mutex::new(TexturePoolSummary::default()));

/// gpui_wgpu publishes the admission outcome for a region after each capture
/// attempt (or each rejection). list.rs reads via `get_admission_record` when
/// emitting `PerCardFrameEvent`.
#[inline]
pub fn record_admission_outcome(
    region_id: u64,
    outcome: AdmissionOutcome,
    proposed_dimensions: (u32, u32),
) {
    #[cfg(feature = "texture-cache-debug")]
    {
        if let Ok(mut map) = ADMISSION_REGISTRY.lock() {
            map.insert(
                region_id,
                AdmissionRecord {
                    outcome,
                    proposed_dimensions,
                },
            );
        }
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = (region_id, outcome, proposed_dimensions);
    }
}

/// Look up the most recent admission outcome for a region. None when no
/// admission attempt has been recorded yet.
#[inline]
pub fn get_admission_record(region_id: u64) -> Option<AdmissionRecord> {
    #[cfg(feature = "texture-cache-debug")]
    {
        ADMISSION_REGISTRY
            .lock()
            .ok()
            .and_then(|m| m.get(&region_id).copied())
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = region_id;
        None
    }
}

/// gpui_wgpu publishes per-region texture size after each successful capture
/// (and clears entries when a region is invalidated).
#[inline]
pub fn record_texture_size(region_id: u64, record: TextureSizeRecord) {
    #[cfg(feature = "texture-cache-debug")]
    {
        if let Ok(mut map) = TEXTURE_SIZE_REGISTRY.lock() {
            if record.bytes == 0 && record.width == 0 && record.height == 0 {
                map.remove(&region_id);
            } else {
                map.insert(region_id, record);
            }
        }
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = (region_id, record);
    }
}

/// Look up the cached texture facts for a region. Returns the default
/// (zero-bytes, zero-dimensions) record when no texture is currently cached.
#[inline]
pub fn get_texture_size(region_id: u64) -> TextureSizeRecord {
    #[cfg(feature = "texture-cache-debug")]
    {
        TEXTURE_SIZE_REGISTRY
            .lock()
            .ok()
            .and_then(|m| m.get(&region_id).copied())
            .unwrap_or_default()
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = region_id;
        TextureSizeRecord::default()
    }
}

/// gpui_wgpu publishes the pool-wide aggregates once per frame after admission
/// processing. list.rs reads at end-of-frame summary emit time.
#[inline]
pub fn set_texture_pool_summary(summary: TexturePoolSummary) {
    #[cfg(feature = "texture-cache-debug")]
    {
        if let Ok(mut s) = POOL_SUMMARY.lock() {
            *s = summary;
        }
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = summary;
    }
}

/// Read the most recent pool-wide aggregates. Returns the default summary when
/// gpui_wgpu has not published yet (e.g. before the first paint).
#[inline]
pub fn texture_pool_summary() -> TexturePoolSummary {
    #[cfg(feature = "texture-cache-debug")]
    {
        POOL_SUMMARY
            .lock()
            .map(|s| *s)
            .unwrap_or_default()
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        TexturePoolSummary::default()
    }
}

#[cfg(feature = "texture-cache-debug")]
fn with_callback<F: FnOnce(&CacheTelemetryCallback)>(f: F) {
    CACHE_TELEMETRY_CB.with(|cell| {
        if let Some(ptr) = cell.get() {
            // SAFETY: pointer is to a `Box::leak`'d callback registered at app
            // startup. Lifetime exceeds all paint passes; never freed.
            let cb = unsafe { &*ptr };
            f(cb);
        }
    });
}
