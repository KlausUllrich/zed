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
//! ## Architecture — Prepaint records, Paint reads
//!
//! GPUI's element protocol runs in two phases: **PREPAINT** (where
//! `Render::render` is invoked, the element tree is constructed, layout runs)
//! and **PAINT** (where `Element::paint` walks the already-constructed tree
//! and emits primitives). Vendor (gpui-component) instrumentation in
//! `state.rs::TextViewState::render`, `text_view.rs::TextView::request_layout`,
//! and `node.rs::render_table` ALL run during PREPAINT — by the time list.rs's
//! paint loop runs, the element tree is built and vendor's record_X calls
//! have already fired. The fork's `begin_card`/`end_card` brackets only
//! surround PAINT, so they cannot capture vendor's prepaint-side fingerprints
//! via a per-card recorder.
//!
//! Resolution: vendor instrumentation writes to **`PREPAINT_REGISTRY`**, a
//! frame-keyed by-index HashMap, keyed by **`PREPAINT_CARD_IDX`** (a
//! thread-local set by cs-ui's `render_item` closure at the start of each
//! item via `enter_prepaint_card(idx)`). cs-app's paint-side `end_card` reads
//! the registry by `current_card_index` (set by paint's `begin_card`) AND
//! checks `frame_id == card_timeline::frame()` so only this frame's
//! fingerprints survive (stale entries from prior frames are returned as
//! defaults). For cards on HIT path where `render_item` doesn't run at all,
//! the registry has no entry for the current frame → defaults returned →
//! markdown_render_ran=false / writes=0 / cells=0, which is correct.
//!
//! - **Vendor (PREPAINT)** marks per-card facts via record_X functions that
//!   look up `PREPAINT_CARD_IDX` and write to `PREPAINT_REGISTRY`.
//! - **cs-ui (PREPAINT)** sets `PREPAINT_CARD_IDX` at the start of `render_item`
//!   and clears at the end (or leaves set; next iteration overwrites).
//!   Also writes `LAST_CARD_TYPE` and `LAST_CACHED_HEIGHT` directly by index.
//! - **Fork (PAINT)** list.rs paint loop calls `begin_card` (sets
//!   `current_card_index` for the cell-paint emit fallback path) → element
//!   paint → `end_card` (reads PREPAINT_REGISTRY by index + frame_id) →
//!   emits `PerCardFrameEvent`. After the loop, emits one
//!   `PerFrameSummaryEvent`.
//! - **cs-app** registers the callback at startup; the callback forwards each
//!   event to `cs_core::cs_log!(Subsystem::RenderCache, Level::Debug, "event=...")`.
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
    /// `caching_enabled == false` — DIRECT/PLAIN paint with no admission attempt.
    /// Pair with `NotEligibleReason` on `PerCardFrameEvent` for the specific cause.
    RejectedNotEligible,
}

// NOTE: `RejectedSpliceClear` was removed in S521 Phase 2.7. It was a dead
// variant — never emitted by any call site. The splice-driven disable is now
// surfaced via `NotEligibleReason::Splice` on `PerCardFrameEvent`, which
// records the *immediate* fork-side cause of `caching_enabled == false`.
// Investigation per Rule 30 confirmed no consumer matched on it.

impl AdmissionOutcome {
    /// Compact string used in NDJSON payloads.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Admitted => "Admitted",
            Self::RejectedTooLarge => "RejectedTooLarge",
            Self::RejectedStreaming => "RejectedStreaming",
            Self::RejectedTransient => "RejectedTransient",
            Self::RejectedAlreadyCached => "RejectedAlreadyCached",
            Self::RejectedNotEligible => "RejectedNotEligible",
        }
    }
}

/// Specific cause for `AdmissionOutcome::RejectedNotEligible`. Recorded as
/// `last_disable_reason` on `ListStateInner` whenever `caching_enabled` flips
/// to `false`, snapshotted into the per-card-frame event during paint. `None`
/// for frames where admission_outcome is anything other than `RejectedNotEligible`.
///
/// Purpose: subdivide the lumped "PLAIN/RejectedNotEligible" bucket so the
/// trace tells us WHICH gate caused the card to be cache-ineligible. The
/// 977/6604 events at `RejectedNotEligible` in Klaus's S521 smoke trace
/// previously carried no causal information.
///
/// **Scope: only causes of `caching_enabled == false`.** This enum does NOT
/// cover `visible_frames.clear()` events from `scroll_to()` / `scroll_to_max()`.
/// Those clears reset the per-item visible-frames counter; the next frame's
/// affected cards emit `cache_state=TRANSIENT` + `admission=RejectedTransient`,
/// not `RejectedNotEligible`. See the comment at the `visible_frames.clear()`
/// site in `scroll_to_max()` for the disable-vs-clear distinction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotEligibleReason {
    /// `set_item_caching_enabled(true)` has never been called this session —
    /// either the global `ITEM_CACHING_ENABLED` atomic is `false` (S506
    /// hotfix default) so view.rs never propagates an enable to the fork,
    /// or the session simply hasn't reached a scroll/drag start yet.
    NeverEnabled,
    /// User explicitly disabled via F9 / Diagnostics panel toggle. Plumbed
    /// from `cache_circuit_breaker::set_user_enabled(false)` through the
    /// `disable_reason` parameter on `set_item_caching_enabled`.
    UserDisabled,
    /// Circuit breaker auto-tripped on violation density or surface health.
    /// Plumbed from `cache_circuit_breaker::trip()` through the
    /// `disable_reason` parameter on `set_item_caching_enabled`.
    BreakerTripped,
    /// Scroll animation ended normally (`velocity_converged`, `boundary_stop`,
    /// `drag_end`, etc.) — `invalidate_all_item_caches()` flipped
    /// `caching_enabled` to false. The expected steady-state at-rest cause.
    ScrollStopped,
    /// `splice_focusable()` or `splice_with_heights()` shifted item indices,
    /// invalidating the entire cache. Frames immediately following a splice
    /// are PLAIN until caching is re-enabled.
    Splice,
    /// `invalidate_all_caches()` — DPI / font / theme change forced a
    /// global cache flush.
    GlobalInvalidation,
}

impl NotEligibleReason {
    /// Compact string used in NDJSON payloads.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NeverEnabled => "NeverEnabled",
            Self::UserDisabled => "UserDisabled",
            Self::BreakerTripped => "BreakerTripped",
            Self::ScrollStopped => "ScrollStopped",
            Self::Splice => "Splice",
            Self::GlobalInvalidation => "GlobalInvalidation",
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
    /// Specific cause when `admission_outcome == RejectedNotEligible`. `None`
    /// otherwise. Subdivides the lumped PLAIN bucket — see `NotEligibleReason`.
    pub not_eligible_reason: Option<NotEligibleReason>,
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

/// Card-index thread-local set by cs-ui's `render_item` closure during PREPAINT.
/// Vendor (gpui-component) record_X functions read this to know which card is
/// currently rendering. None when no `render_item` body is active. See module
/// docs for full lifecycle.
#[cfg(feature = "texture-cache-debug")]
thread_local! {
    static PREPAINT_CARD_IDX: Cell<Option<usize>> = const { Cell::new(None) };
}

/// S521 Phase 2.7: thread-local "pending" disable reason set by cs-ui code
/// that disables caching INDIRECTLY (i.e. via the cs-ui `ITEM_CACHING_ENABLED`
/// atomic, not via the fork's `set_item_caching_enabled` API). The next
/// fork-side disable event consumes this — picking up `BreakerTripped` or
/// `UserDisabled` instead of the natural site-specific reason
/// (e.g. `ScrollStopped` from `invalidate_all_item_caches`). The "next fork
/// disable" is typically the scroll-stop call following a breaker trip;
/// without this thread-local, the trace would attribute that disable to
/// `ScrollStopped` and lose the actual root cause.
///
/// Lifecycle:
/// 1. cs-ui calls `set_pending_external_disable_reason(BreakerTripped)`
///    BEFORE flipping the cs-ui atomic.
/// 2. The atomic flips; the fork's `caching_enabled` is still whatever it was.
/// 3. Next scroll-stop / splice / global-invalidation on the fork side calls
///    `take_pending_external_disable_reason()` and uses the returned `Some(X)`
///    in preference to the site's natural reason.
/// 4. If consumed: thread-local cleared. If no fork-side disable happens
///    before the next external set, the new reason overwrites the stale one.
#[cfg(feature = "texture-cache-debug")]
thread_local! {
    static PENDING_EXTERNAL_DISABLE_REASON: Cell<Option<NotEligibleReason>> =
        const { Cell::new(None) };
}

/// Set the pending external disable reason. Called by cs-ui breaker trips
/// and user-toggle paths immediately BEFORE flipping the cs-ui caching
/// atomic. See `PENDING_EXTERNAL_DISABLE_REASON` lifecycle docs.
#[inline]
pub fn set_pending_external_disable_reason(reason: NotEligibleReason) {
    #[cfg(feature = "texture-cache-debug")]
    {
        PENDING_EXTERNAL_DISABLE_REASON.with(|c| c.set(Some(reason)));
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = reason;
    }
}

/// Consume the pending external disable reason if present, returning it and
/// clearing the thread-local. Called by every fork-side disable site
/// (`splice_focusable`, `splice_with_heights`, `invalidate_all_item_caches`,
/// `invalidate_all_caches`) BEFORE writing its own natural reason. If `Some`,
/// the external cause supersedes the site's natural cause.
#[inline]
pub fn take_pending_external_disable_reason() -> Option<NotEligibleReason> {
    #[cfg(feature = "texture-cache-debug")]
    {
        PENDING_EXTERNAL_DISABLE_REASON.with(|c| c.take())
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        None
    }
}

/// Per-card prepaint fingerprint recorded by vendor instrumentation. Frame-aware:
/// `frame_id` is checked at read time so stale entries (from cards that weren't
/// re-rendered this frame) return defaults. Reset on the first record_X call of
/// a new frame for that card.
#[derive(Clone, Copy, Debug, Default)]
struct PrepaintFingerprint {
    /// `card_timeline::frame()` value at the moment of the last record_X call
    /// for this card. Used to gate stale reads in `end_card`.
    frame_id: u64,
    markdown_render_ran: bool,
    state_update_writes: u32,
    state_update_changed: u32,
    has_table: bool,
    cell_count: u32,
}

/// Frame-keyed by-index registry written by vendor record_X functions during
/// PREPAINT, read by `end_card` during PAINT.
#[cfg(feature = "texture-cache-debug")]
static PREPAINT_REGISTRY: LazyLock<Mutex<HashMap<usize, PrepaintFingerprint>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Internal helper: reset-or-fetch a registry entry for the current frame.
/// Returns a mutable handle (via callback) so the caller can update fields.
/// Caller MUST hold `PREPAINT_CARD_IDX` set to the right index. Returns false
/// when PREPAINT_CARD_IDX is None — callers skip the record on false.
#[cfg(feature = "texture-cache-debug")]
fn with_prepaint_entry<F: FnOnce(&mut PrepaintFingerprint)>(f: F) -> bool {
    let idx = match PREPAINT_CARD_IDX.with(|c| c.get()) {
        Some(idx) => idx,
        None => return false,
    };
    // S521 Bug 2b: anticipate the `bump_frame()` that runs at the START of
    // `list.rs::fn paint()` (line ~2497) — the SOLE call site in the codebase.
    // Vendor instrumentation runs during PREPAINT (BEFORE that bump), so writes
    // here would tag fingerprints with FRAME=N. `end_card()` runs during PAINT
    // (AFTER the bump) and reads FRAME=N+1; without this `+ 1`, the frame_id
    // filter at line ~477 rejects every fingerprint as stale, and the
    // md_render/writes/changed/cells fields emit as zero on every card.
    // Encodes the single-bump-per-frame invariant — if a second `bump_frame()`
    // is ever added in the GPUI fork, this asymmetric `+ 1` will misalign and
    // the resulting all-zero regression points back to this comment.
    let frame = crate::card_timeline::frame() + 1;
    if let Ok(mut reg) = PREPAINT_REGISTRY.lock() {
        let entry = reg.entry(idx).or_default();
        // Frame transition: stale fingerprint from a prior frame; reset before
        // applying the new record.
        if entry.frame_id != frame {
            *entry = PrepaintFingerprint {
                frame_id: frame,
                ..PrepaintFingerprint::default()
            };
        }
        f(entry);
        true
    } else {
        false
    }
}

/// Atomic gate for per-cell paint events. Toggled at runtime from cs-ui
/// (Render Cache sidepane sub-toggle). OFF by default; ON when Klaus clicks the
/// "Render Cache (per-cell)" entry to investigate cell-level cost.
#[cfg(feature = "texture-cache-debug")]
static PER_CELL_TRACE_ENABLED: AtomicBool = AtomicBool::new(false);

/// Stub when the feature is off — keeps the symbol present so callers compile.
#[cfg(not(feature = "texture-cache-debug"))]
static PER_CELL_TRACE_ENABLED: AtomicBool = AtomicBool::new(false);

/// Internal per-card scratch state during PAINT. Holds only the
/// `current_card_index` (set by `begin_card`, read by the cell-paint emit's
/// fallback path). All vendor-side per-card fingerprints
/// (markdown_render_ran, state_update_writes, has_table, cell_count) live in
/// `PREPAINT_REGISTRY` instead, keyed by card_index, because they're written
/// during PREPAINT (before `begin_card` runs).
#[derive(Clone, Default, Debug)]
pub struct PerCardRecorder {
    /// Card index set by the most recent `begin_card`. `None` when no card paint
    /// is active.
    pub current_card_index: Option<usize>,
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

/// Harvest the per-card snapshot. Call after `item.element.paint()` in the
/// list paint loop. Returns the snapshot for inclusion in the
/// `PerCardFrameEvent`. Reads:
/// - `card_type` from `LAST_CARD_TYPE` (populated by cs-ui's
///   `record_card_type` during prepaint).
/// - Vendor fingerprints (markdown_render_ran, state_update_writes,
///   state_update_changed, has_table, cell_count) from `PREPAINT_REGISTRY`,
///   gated by `frame_id == card_timeline::frame()` so stale entries from
///   prior frames return as defaults. For HIT cards (no `render_item` ran
///   this frame), defaults are correct because no markdown render took place.
#[inline]
pub fn end_card() -> PerCardSnapshot {
    #[cfg(feature = "texture-cache-debug")]
    {
        let card_index = PER_CARD_RECORDER.with(|cell| {
            let mut rec = cell.borrow_mut();
            let idx = rec.current_card_index;
            *rec = PerCardRecorder::default();
            idx
        });
        let card_type = card_index
            .and_then(|idx| {
                LAST_CARD_TYPE.lock().ok().and_then(|m| m.get(&idx).copied())
            })
            .unwrap_or("Unknown");
        let current_frame = crate::card_timeline::frame();
        let fingerprint = card_index
            .and_then(|idx| {
                PREPAINT_REGISTRY
                    .lock()
                    .ok()
                    .and_then(|m| m.get(&idx).copied())
            })
            .filter(|fp| fp.frame_id == current_frame)
            .unwrap_or_default();
        PerCardSnapshot {
            card_type,
            markdown_render_ran: fingerprint.markdown_render_ran,
            state_update_writes: fingerprint.state_update_writes,
            state_update_changed: fingerprint.state_update_changed,
            has_table: fingerprint.has_table,
            cell_count: fingerprint.cell_count,
        }
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
// Prepaint scope API (called by cs-ui render_item closure).
// ---------------------------------------------------------------------------

/// Set the active card index for PREPAINT-side vendor instrumentation. Called
/// at the start of cs-ui's `render_item` closure for the items-lookup branch.
/// Pair with `exit_prepaint_card` at closure exit (or rely on the next
/// invocation overwriting; subsequent record_X calls outside any scope are
/// safely guarded via `current_prepaint_card_index().is_some()`).
#[inline]
pub fn enter_prepaint_card(card_index: usize) {
    #[cfg(feature = "texture-cache-debug")]
    {
        PREPAINT_CARD_IDX.with(|c| c.set(Some(card_index)));
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = card_index;
    }
}

/// Clear the active card index after the render_item closure body completes.
/// Optional — leaving it set is harmless because the next `enter_prepaint_card`
/// overwrites it. Recommended to call when the closure exits an items-lookup
/// branch and might be followed by emit sites that should NOT be attributed
/// to this card.
#[inline]
pub fn exit_prepaint_card() {
    #[cfg(feature = "texture-cache-debug")]
    {
        PREPAINT_CARD_IDX.with(|c| c.set(None));
    }
}

/// Get the active prepaint card index. Used by vendor's per-cell paint emit
/// (since cell_paint events fire DURING prepaint, before fork's begin_card has
/// set the paint-side `current_card_index`). None when no `render_item` body
/// is active.
#[inline]
pub fn current_prepaint_card_index() -> Option<usize> {
    #[cfg(feature = "texture-cache-debug")]
    {
        PREPAINT_CARD_IDX.with(|c| c.get())
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        None
    }
}

// ---------------------------------------------------------------------------
// Vendor-facing record API (called by gpui-component during PREPAINT).
// All record_X functions write to PREPAINT_REGISTRY keyed by
// PREPAINT_CARD_IDX, frame-keyed via card_timeline::frame().
// ---------------------------------------------------------------------------

/// Mark that `TextViewState::render` ran this frame for the active prepaint
/// card. Called at the entry of vendor `state.rs::TextViewState::render`.
/// No-op when called outside a `render_item` scope (PREPAINT_CARD_IDX is None).
#[inline]
pub fn record_markdown_render_ran() {
    #[cfg(feature = "texture-cache-debug")]
    {
        let _ = with_prepaint_entry(|fp| {
            fp.markdown_render_ran = true;
        });
    }
}

/// Record one of the unconditional `state.update(cx, ...)` writes in vendor
/// `text_view.rs::TextView::request_layout`. `changed = true` iff the prior
/// value differed from the incoming value (compute via `state.read(cx)` BEFORE
/// calling `state.update`). No-op when called outside a `render_item` scope —
/// prevents leaks from non-conversation TextView call sites (e.g.
/// `write_viewer.rs` at startup).
#[inline]
pub fn record_state_update(changed: bool) {
    #[cfg(feature = "texture-cache-debug")]
    {
        let _ = with_prepaint_entry(|fp| {
            fp.state_update_writes = fp.state_update_writes.saturating_add(1);
            if changed {
                fp.state_update_changed = fp.state_update_changed.saturating_add(1);
            }
        });
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = changed;
    }
}

/// Record that the active prepaint card's body contains a markdown table.
/// Called from vendor `node.rs::render_table` at function entry.
/// `cell_count = rows × cols`. Multiple tables per card accumulate. No-op
/// when called outside a `render_item` scope.
#[inline]
pub fn record_table(cell_count: u32) {
    #[cfg(feature = "texture-cache-debug")]
    {
        let _ = with_prepaint_entry(|fp| {
            fp.has_table = true;
            fp.cell_count = fp.cell_count.saturating_add(cell_count);
        });
    }
    #[cfg(not(feature = "texture-cache-debug"))]
    {
        let _ = cell_count;
    }
}

/// Get the current paint-side card index (set by `begin_card`, read by the
/// cell-paint emit's fallback path). None when no card paint is active.
/// **For per-cell events, prefer `current_prepaint_card_index()`** since cell
/// paint events fire during prepaint, before begin_card has set this field.
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
