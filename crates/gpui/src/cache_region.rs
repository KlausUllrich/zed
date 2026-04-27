//! GPU texture cache region types and scene extraction.
//!
//! Six responsibilities:
//! 1. Cache region annotations — `CacheRegion`, `CacheRegionId`, and scene helpers
//!    used by the renderer to capture list items to GPU textures.
//! 2. Renderer feedback state — `CACHED_REGION_IDS` thread-local tracks which
//!    regions have valid textures. `has_cached_region` / `clear_cached_region` /
//!    `clear_cached_region_ids` let the list query and invalidate this state.
//! 3. Debug overlay control — `DEBUG_TINT_ENABLED` thread-local and `set_debug_tint` /
//!    `is_debug_tint_enabled`. Set by the app (F9 toggle); read by `draw_cached_regions`.
//! 4. Visibility classification signal — `VISIBLE_REGION_IDS` / `BUFFER_REGION_IDS`
//!    thread-locals with `set_classification_ids` / `take_classification_ids`.
//!    list.rs sets these each paint; the renderer reads them in process_cache_regions
//!    to drive priority-bin classification for eviction ordering.
//! 5. Per-item invalidation signal — `INVALIDATED_REGION_IDS` thread-local with
//!    `invalidate_pool_region` / `take_invalidated_region_ids`. list.rs marks items
//!    for GPU texture purge; the renderer removes them from TexturePool.active.
//! 6. Viewport center index signal — `VIEWPORT_CENTER_INDEX` thread-local with
//!    `set_viewport_center_index` / `take_viewport_center_index`. list.rs sets this
//!    each paint alongside set_classification_ids; texture_cache.rs reads it in
//!    find_priority_victim() to prefer evicting items farthest from scroll position.

use crate::{
    Bounds, ContentMask, Hsla, Point, Primitive, ScaledPixels, Scene,
    point,
    scene::PaintOperation,
};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::Range;

/// Unique identifier for a cache region (typically a list item index).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheRegionId(pub u64);

// --- Renderer → List feedback path ---
// The renderer sets which region IDs have valid textures after each frame.
// The list queries this during paint to decide skip-paint vs Fresh.

thread_local! {
    static CACHED_REGION_IDS: RefCell<HashSet<u64>> = RefCell::new(HashSet::new());
}

/// Called by the renderer after processing cache regions to report
/// which region IDs have valid cached textures.
pub fn set_cached_region_ids(ids: HashSet<u64>) {
    CACHED_REGION_IDS.with(|cell| {
        *cell.borrow_mut() = ids;
    });
}

/// Check if the renderer has a valid cached texture for a region.
/// Used by list.rs to decide whether to skip paint (cache hit).
pub fn has_cached_region(id: CacheRegionId) -> bool {
    CACHED_REGION_IDS.with(|cell| cell.borrow().contains(&id.0))
}

/// FR-2.2: Remove a single region from the cached set.
/// On the next paint, `has_cached_region()` returns false for this ID,
/// causing a cache MISS and a Fresh re-render of the item.
pub fn clear_cached_region(id: CacheRegionId) {
    CACHED_REGION_IDS.with(|cell| cell.borrow_mut().remove(&id.0));
}

/// Clear all cached region IDs to prevent stale IDs from causing
/// skip-paint on items without textures.
///
/// Callers:
///   - `WgpuRenderer::invalidate_texture_cache()` — DPI/width change
///   - `WgpuRenderer::recover()` — GPU device lost (EC-11)
pub fn clear_cached_region_ids() {
    CACHED_REGION_IDS.with(|cell| {
        cell.borrow_mut().clear();
        // Ungated — this function is called by non-feature paths too (e.g. EC-11 recovery).
        log::debug!("event=clear_all_region_ids");
    });
}

// --- Texture dump request ---
// The app sets this flag (e.g., via hotkey) to request a one-shot PNG dump of all
// active textures in the cache pool. The renderer checks this at the start of each
// frame, performs the readback, and clears the flag.

thread_local! {
    static TEXTURE_DUMP_REQUESTED: Cell<bool> = const { Cell::new(false) };
}

/// Request a one-shot dump of all active cached textures to PNG files.
/// The renderer will write images to `/tmp/cs-texture-dump/` on the next frame
/// and clear the flag automatically.
///
/// IMPORTANT: Must be called from the main UI thread. The flag is thread-local;
/// calling from a background thread sets a flag the renderer will never see.
///
/// Called by: CS app hotkey handler (Ctrl+Shift+D) in cs-app/src/app.rs
pub fn request_texture_dump() {
    TEXTURE_DUMP_REQUESTED.with(|cell| cell.set(true));
}

/// Check and clear the texture dump request flag. Returns `true` if a dump
/// was requested since the last call.
///
/// Consumed by: `WgpuRenderer::dump_active_textures_if_requested()` in gpui_wgpu
pub fn take_texture_dump_request() -> bool {
    TEXTURE_DUMP_REQUESTED.with(|cell| cell.replace(false))
}

// --- List → Renderer invalidation signal ---
// The list sets a flag when all cached textures should be released (DPI, theme, font).
// The renderer checks this at the start of each frame and flushes the texture pool.
// This keeps gpui (list.rs) decoupled from gpui_wgpu (TexturePool).

thread_local! {
    static PENDING_POOL_INVALIDATION: Cell<bool> = const { Cell::new(false) };
}

/// Signal the renderer to flush all GPU texture pool resources on the next frame.
/// Called by `ListState::invalidate_all_caches()` for global visual changes
/// (DPI, theme, font size). The renderer checks this via `take_pending_pool_invalidation()`.
pub fn request_pool_invalidation() {
    PENDING_POOL_INVALIDATION.with(|cell| cell.set(true));
}

/// Check and clear the pool invalidation flag. Returns `true` if invalidation
/// was requested since the last call. Called by the renderer at frame start.
pub fn take_pending_pool_invalidation() -> bool {
    PENDING_POOL_INVALIDATION.with(|cell| cell.replace(false))
}

// --- Phase B feature toggles ---
// Runtime toggles for Phase B features that can cause blank cards.
// Each defaults to ON (Phase B behavior). Toggle OFF for Phase A behavior.
// Set by F9 debug tab; read by list.rs and texture_cache.rs.

thread_local! {
    /// EC-6: When ON, items must be visible for TRANSIENT_SKIP_FRAMES before capture.
    /// OFF = capture on first frame (Phase A behavior).
    static TRANSIENT_SKIP_ENABLED: Cell<bool> = const { Cell::new(true) };
    /// When ON, 64MB memory budget triggers LRU eviction.
    /// OFF = no eviction, textures persist forever (Phase A behavior).
    static BUDGET_EVICTION_ENABLED: Cell<bool> = const { Cell::new(true) };
    /// EC-12: When ON, items beyond viewport are laid out and cached.
    /// OFF = only cache items inside viewport (Phase A behavior).
    static OVERDRAW_CACHING_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Toggle transient skip (EC-6). OFF = capture on first frame (Phase A).
pub fn set_transient_skip_enabled(enabled: bool) {
    TRANSIENT_SKIP_ENABLED.with(|cell| cell.set(enabled));
}
/// Check if transient skip is enabled.
pub fn is_transient_skip_enabled() -> bool {
    TRANSIENT_SKIP_ENABLED.with(|cell| cell.get())
}

/// Toggle budget eviction. OFF = no eviction (Phase A).
pub fn set_budget_eviction_enabled(enabled: bool) {
    BUDGET_EVICTION_ENABLED.with(|cell| cell.set(enabled));
}
/// Check if budget eviction is enabled.
pub fn is_budget_eviction_enabled() -> bool {
    BUDGET_EVICTION_ENABLED.with(|cell| cell.get())
}

/// Toggle overdraw caching (EC-12). OFF = only cache viewport items (Phase A).
pub fn set_overdraw_caching_enabled(enabled: bool) {
    OVERDRAW_CACHING_ENABLED.with(|cell| cell.set(enabled));
}
/// Check if overdraw caching is enabled.
pub fn is_overdraw_caching_enabled() -> bool {
    OVERDRAW_CACHING_ENABLED.with(|cell| cell.get())
}

// --- Phase B feature toggles (batch 2) ---

thread_local! {
    /// When ON, composites cached textures via `pipelines.composite` (Phase B).
    /// OFF = use `pipelines.paths` (Phase A behavior).
    static COMPOSITE_PIPELINE_ENABLED: Cell<bool> = const { Cell::new(true) };
    /// When ON, SubpixelSprites in cached textures fall back to mono_sprites pipeline (Phase B).
    /// OFF = use original subpixel_sprites pipeline (Phase A behavior).
    static MONO_FALLBACK_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Toggle composite pipeline. OFF = use paths pipeline (Phase A).
pub fn set_composite_pipeline_enabled(enabled: bool) {
    COMPOSITE_PIPELINE_ENABLED.with(|cell| cell.set(enabled));
}
/// Check if composite pipeline is enabled.
pub fn is_composite_pipeline_enabled() -> bool {
    COMPOSITE_PIPELINE_ENABLED.with(|cell| cell.get())
}

/// Toggle mono sprite fallback. OFF = use subpixel pipeline (Phase A).
pub fn set_mono_fallback_enabled(enabled: bool) {
    MONO_FALLBACK_ENABLED.with(|cell| cell.set(enabled));
}
/// Check if mono sprite fallback is enabled.
pub fn is_mono_fallback_enabled() -> bool {
    MONO_FALLBACK_ENABLED.with(|cell| cell.get())
}

// --- List → Renderer per-item invalidation signal ---
// When a specific item's cache is invalidated (e.g. user interaction on a permission card),
// the old GPU texture must also be purged from TexturePool.active to prevent a 1-frame
// stale HIT from the feedback delay.

thread_local! {
    static INVALIDATED_REGION_IDS: RefCell<HashSet<u64>> = RefCell::new(HashSet::new());
}

/// Mark a region for GPU texture purge. The renderer will remove the matching entry
/// from `TexturePool.active` (moving the texture to the free list for reuse).
/// Called by: `ListState::invalidate_item_cache()` in list.rs.
pub fn invalidate_pool_region(id: CacheRegionId) {
    INVALIDATED_REGION_IDS.with(|cell| cell.borrow_mut().insert(id.0));
}

/// Take the set of region IDs that need GPU texture purging. Clears the thread-local.
/// Called by: `WgpuRenderer::process_cache_regions()` in gpui_wgpu.
pub fn take_invalidated_region_ids() -> HashSet<u64> {
    INVALIDATED_REGION_IDS.with(|cell| std::mem::take(&mut *cell.borrow_mut()))
}

// --- List → Renderer visibility classification ---
// The list sets visible/buffer region ID sets each frame during paint.
// The renderer reads these in process_cache_regions to classify cache entries
// into priority bins (Visible, Buffer, Recent, Distant) for eviction ordering.

thread_local! {
    static VISIBLE_REGION_IDS: RefCell<HashSet<u64>> = RefCell::new(HashSet::new());
    static BUFFER_REGION_IDS: RefCell<HashSet<u64>> = RefCell::new(HashSet::new());
}

/// Set the visible (in-viewport) and buffer (overdraw) region ID sets for this frame.
/// Called by: `List::paint()` in list.rs, before the item paint loop.
pub fn set_classification_ids(visible_ids: HashSet<u64>, buffer_ids: HashSet<u64>) {
    VISIBLE_REGION_IDS.with(|cell| *cell.borrow_mut() = visible_ids);
    BUFFER_REGION_IDS.with(|cell| *cell.borrow_mut() = buffer_ids);
}

/// Take the visible and buffer region ID sets. Returns (visible_ids, buffer_ids).
/// Clears the thread-local state.
/// Called by: `WgpuRenderer::process_cache_regions()` in gpui_wgpu.
pub fn take_classification_ids() -> (HashSet<u64>, HashSet<u64>) {
    let visible = VISIBLE_REGION_IDS.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
    let buffer = BUFFER_REGION_IDS.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
    (visible, buffer)
}

// --- Viewport center index for priority-bin eviction ---
// The list sets this each frame during paint. The renderer uses it in
// find_priority_victim() to prefer evicting items farthest from the viewport center.

thread_local! {
    static VIEWPORT_CENTER_INDEX: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Set the viewport center item index for this frame.
/// Called by: `List::paint()` in list.rs, alongside set_classification_ids.
pub fn set_viewport_center_index(center: usize) {
    VIEWPORT_CENTER_INDEX.with(|cell| cell.set(Some(center)));
}

/// Take the viewport center item index. Returns `None` if not set this frame.
/// Clears the thread-local state.
/// Called by: `WgpuRenderer::process_cache_regions()` in gpui_wgpu.
pub fn take_viewport_center_index() -> Option<usize> {
    VIEWPORT_CENTER_INDEX.with(|cell| cell.take())
}

// --- Debug tint overlay ---
// The app sets this flag (e.g., via F9 toggle) to enable a red tint overlay
// on all composited cached textures. The renderer reads this in draw_cached_regions.

thread_local! {
    static DEBUG_TINT_ENABLED: Cell<bool> = const { Cell::new(false) };
}

/// Enable or disable the debug tint overlay on composited cached textures.
/// When enabled, the renderer draws a semi-transparent red overlay on every
/// composited texture so users can visually distinguish cached items from
/// fresh-rendered ones during scroll (see `draw_cached_regions` for alpha value).
/// Call site: cs-debug F9 tab tint toggle button.
pub fn set_debug_tint(enabled: bool) {
    DEBUG_TINT_ENABLED.with(|cell| cell.set(enabled));
}

/// Check whether the debug tint overlay is enabled.
/// Called by the renderer in `draw_cached_regions`.
pub fn is_debug_tint_enabled() -> bool {
    DEBUG_TINT_ENABLED.with(|cell| cell.get())
}

// --- S513 trace enrichment: active conversation owner + per-region card type ---
// Threaded from CS each render pass so cache events fired in the renderer can
// stamp `agent=<id>` (e.g. agent=max) and `type=<card_type>` (e.g. type=ToolCall).
// CS calls `set_active_agent(None)` on conversation switch so events fired
// before the next render don't leak the prior owner. Stale reads return "unknown".

thread_local! {
    static ACTIVE_AGENT_ID: RefCell<Option<String>> = const { RefCell::new(None) };
    // Values are &'static str from `ConversationItem::card_type_name()` —
    // zero-allocation per-frame map rebuild on the CS render side.
    static CARD_TYPES: RefCell<HashMap<u64, &'static str>> = RefCell::new(HashMap::new());
}

/// Set the agent ID owning the active conversation. `None` clears the slot
/// (use this on conversation switch before the new render call site populates it).
/// Call sites: `ConversationView::render()` in cs-ui (per-frame).
pub fn set_active_agent(agent: Option<String>) {
    ACTIVE_AGENT_ID.with(|cell| *cell.borrow_mut() = agent);
}

/// Snapshot the active agent ID, or "unknown" when unset. Allocates a String —
/// callers fold this into a `format!` arg, not a hot inner loop.
pub fn active_agent_str() -> String {
    ACTIVE_AGENT_ID
        .with(|cell| cell.borrow().clone())
        .unwrap_or_else(|| "unknown".to_string())
}

// --- S513 Move 2: chosen wgpu PresentMode at startup ---
// Decided in `WgpuRenderer::new()` based on what the platform surface
// advertises. Stamped once, read on every paint_timing_breakdown emission so
// Klaus can see whether the build is using Mailbox (no vsync cliff) or Fifo
// (cliff -> 144/3 = 48 FPS at 16-17ms frame work). Process-global because the
// renderer chooses once per process and never changes.
//
// Use of `OnceLock` (cross-thread safe; one-shot write semantics) avoids the
// thread_local re-read-on-each-thread surprise: paint and present run on
// different threads in the wgpu pipeline.
static PRESENT_MODE_NAME: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();

/// Stamp the chosen wgpu present mode at app startup. First call wins; later
/// calls (e.g. on monitor reconfiguration) are ignored — present_mode rarely
/// changes mid-session and Klaus only needs the startup decision recorded.
/// Call site: `WgpuRenderer::new()` in gpui_wgpu/src/wgpu_renderer.rs.
pub fn set_present_mode_name(name: &'static str) {
    let _ = PRESENT_MODE_NAME.set(name);
}

/// Snapshot the chosen present mode, or `"Unknown"` if `set_present_mode_name`
/// wasn't called yet (which can only happen on frames painted before the
/// renderer finished construction — should be impossible in practice).
pub fn present_mode_str() -> &'static str {
    PRESENT_MODE_NAME.get().copied().unwrap_or("Unknown")
}

/// Replace the per-region card-type map. Map keys are list region_ids
/// (CS item index + LIST_SPACER_OFFSET). Values are stable string literals
/// from `ConversationItem::card_type_name()` so they're zero-allocation.
/// Call sites: `ConversationView::render()` in cs-ui (per-frame).
pub fn set_card_types(types: HashMap<u64, &'static str>) {
    CARD_TYPES.with(|cell| *cell.borrow_mut() = types);
}

/// Lookup the card type for a region_id, falling back to "unknown" if
/// the map hasn't been populated this frame (e.g. classic Zed list, or
/// CS render hasn't run yet). Cheap — single HashMap read.
pub fn card_type_for_region(region_id: u64) -> &'static str {
    CARD_TYPES.with(|cell| cell.borrow().get(&region_id).copied().unwrap_or("unknown"))
}

// --- S513 perf instrumentation: per-frame timing accumulator + paint_timing_breakdown ---
// Each existing per-phase event (layout_phase_end, frame_cache_summary, etc.)
// records its numbers via `frame_perf_record_*`. WgpuRenderer::draw consolidates
// the accumulator into one `event=paint_timing_breakdown` per painted frame
// after `frame.present()` returns. Klaus reads this single line to compare
// caching ON vs OFF without correlating four separate events.
//
// FPS rolling window: 60 samples (~0.4 s at 144 Hz, 1.0 s at 60 Hz). Chosen for
// responsiveness while smoothing single-frame jitter; documented in delivery-notes.

#[derive(Default, Clone, Copy)]
struct FramePerf {
    paint_frame: u64,
    layout_ms: f32,
    capture_ms: f32,
    composite_ms: f32,
    hit_count: u32,
    miss_count: u32,
    plain_count: u32,
    /// True once a layout or paint phase has populated the accumulator this frame.
    /// Suppresses spurious zero events on idle frames (e.g. presentation_feedback
    /// firing without a corresponding paint).
    have_data: bool,
}

thread_local! {
    static FRAME_PERF: Cell<FramePerf> = const {
        Cell::new(FramePerf {
            paint_frame: 0, layout_ms: 0.0, capture_ms: 0.0, composite_ms: 0.0,
            hit_count: 0, miss_count: 0, plain_count: 0, have_data: false,
        })
    };
    static FRAME_PRESENT_TIMES_MS: RefCell<VecDeque<u128>> =
        RefCell::new(VecDeque::with_capacity(64));
}

/// Record layout-phase duration for the consolidated paint_timing_breakdown.
/// Called from `List::layout_items()` after the existing `layout_phase_end`
/// event so caching-OFF baselines (where layout is the dominant cost) are
/// still measured even when no cache events fire.
///
/// NOTE: A layout pass without a following paint pass will still emit the
/// consolidated event with zeroed hit/miss/plain counts — that's correct
/// (layout-cost-only frame). Distinct from idle frames (no paint at all),
/// which are suppressed by `have_data` because nothing recorded anything.
pub fn frame_perf_record_layout(paint_frame: u64, layout_ms: f32) {
    FRAME_PERF.with(|cell| {
        let mut perf = cell.get();
        perf.paint_frame = paint_frame;
        perf.layout_ms = layout_ms;
        perf.have_data = true;
        cell.set(perf);
    });
}

/// Record per-frame HIT / MISS / PLAIN counts for the consolidated event.
/// Called from `List::paint()` alongside the existing `frame_cache_summary`.
/// With caching disabled, hit/miss are 0 and plain reflects every painted
/// item — the "OFF baseline" comparison Klaus needs.
pub fn frame_perf_record_counts(paint_frame: u64, hit: u32, miss: u32, plain: u32) {
    FRAME_PERF.with(|cell| {
        let mut perf = cell.get();
        perf.paint_frame = paint_frame;
        perf.hit_count = hit;
        perf.miss_count = miss;
        perf.plain_count = plain;
        perf.have_data = true;
        cell.set(perf);
    });
}

/// Record `process_cache_regions` capture-pass duration.
///
/// Does NOT set `have_data` — capture-only with no preceding layout/counts
/// would mean the renderer ran but the list never laid out or painted, which
/// is impossible in practice. Only `record_layout` and `record_counts` gate
/// emission; capture/composite are timing details on an already-painted frame.
pub fn frame_perf_record_capture(capture_ms: f32) {
    FRAME_PERF.with(|cell| {
        let mut perf = cell.get();
        perf.capture_ms = capture_ms;
        cell.set(perf);
    });
}

/// Record `draw_cached_regions` composite-pass duration.
/// Same `have_data` semantics as `frame_perf_record_capture` — see its docs.
pub fn frame_perf_record_composite(composite_ms: f32) {
    FRAME_PERF.with(|cell| {
        let mut perf = cell.get();
        perf.composite_ms = composite_ms;
        cell.set(perf);
    });
}

/// Stamp present-end timestamp into the rolling FPS window and emit the
/// consolidated `event=paint_timing_breakdown` line. `present_ms` is the
/// wall-clock `frame.present()` duration measured by the renderer.
///
/// Call site: `WgpuRenderer::draw()` immediately after `frame.present()` returns.
/// No-op for frames where neither `record_layout` nor `record_counts` ran
/// (idle / presentation-feedback-only frames) — avoids spurious zero events.
pub fn frame_perf_emit_with_present(present_ms: f32) {
    let perf = FRAME_PERF.with(|cell| {
        let snapshot = cell.get();
        cell.set(FramePerf::default());
        snapshot
    });
    if !perf.have_data {
        return;
    }
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let fps_rolling = FRAME_PRESENT_TIMES_MS.with(|cell| {
        let mut queue = cell.borrow_mut();
        if queue.len() >= 60 {
            queue.pop_front();
        }
        queue.push_back(now_ms);
        if queue.len() < 2 {
            0.0
        } else {
            let span_ms =
                queue.back().unwrap().saturating_sub(*queue.front().unwrap()) as f32;
            if span_ms <= 0.0 {
                0.0
            } else {
                (queue.len() as f32 - 1.0) * 1000.0 / span_ms
            }
        }
    });
    let agent = active_agent_str();
    let present_mode = present_mode_str();
    log::info!(
        "event=paint_timing_breakdown paint_frame={} agent={} present_mode={} layout_ms={:.1} capture_ms={:.1} composite_ms={:.1} present_ms={:.1} hit_count={} miss_count={} plain_count={} fps_rolling={:.1}",
        perf.paint_frame, agent, present_mode,
        perf.layout_ms, perf.capture_ms, perf.composite_ms, present_ms,
        perf.hit_count, perf.miss_count, perf.plain_count,
        fps_rolling
    );
}

/// A completed cache region annotation in the scene.
/// Identifies a contiguous range of paint operations that can be rendered
/// to an offscreen GPU texture and composited as a single quad.
#[derive(Clone, Debug)]
pub struct CacheRegion {
    /// Unique identifier for this cache region.
    pub id: CacheRegionId,
    /// Bounds of the region in window coordinates (used for texture sizing).
    pub bounds: Bounds<ScaledPixels>,
    /// Background color to clear the offscreen texture with before rendering.
    /// Required for correct subpixel text anti-aliasing.
    pub clear_color: Hsla,
    /// Range of indices in `paint_operations` belonging to this region
    /// (excludes the Begin/End markers themselves).
    pub paint_op_range: Range<usize>,
    /// Draw order assigned during scene construction. Used by the renderer
    /// to composite cached textures at the correct z-position (after content
    /// batches, before overlay layers).
    pub composite_order: u32,
    /// Clip rectangle for compositing (the list viewport in window coordinates).
    /// Composite quads are clipped to this rect to prevent overflow outside
    /// the list bounds.
    pub viewport_clip: Bounds<ScaledPixels>,
}

impl Primitive {
    /// Translate bounds for texture capture (window→local coordinates).
    /// Unlike `translate()`, this skips `transformation.translation` for
    /// MonochromeSprite/SubpixelSprite to avoid the GPU double-offset bug
    /// (shader computes: position = bounds.origin + transformation.translation).
    pub fn translate_for_capture(&mut self, offset: Point<ScaledPixels>) {
        match self {
            Primitive::Shadow(s) => s.bounds.origin += offset,
            Primitive::Quad(q) => q.bounds.origin += offset,
            Primitive::Path(p) => {
                p.bounds.origin += offset;
                for vertex in &mut p.vertices {
                    vertex.xy_position += offset;
                }
            }
            Primitive::Underline(u) => u.bounds.origin += offset,
            Primitive::MonochromeSprite(s) => {
                s.bounds.origin += offset;
                // CRITICAL: Do NOT update transformation.translation.
                // GPU shader double-counts: position = bounds.origin + transformation.translation.
            }
            Primitive::SubpixelSprite(s) => {
                s.bounds.origin += offset;
                // CRITICAL: Same double-offset fix as MonochromeSprite.
            }
            Primitive::PolychromeSprite(s) => s.bounds.origin += offset,
            Primitive::Surface(s) => s.bounds.origin += offset,
        }
    }

    /// Set the content mask on this primitive, expanding clipping bounds.
    /// Used during texture capture to prevent edge clipping in offscreen textures.
    pub fn set_content_mask(&mut self, mask: ContentMask<ScaledPixels>) {
        match self {
            Primitive::Shadow(s) => s.content_mask = mask,
            Primitive::Quad(q) => q.content_mask = mask,
            Primitive::Path(p) => {
                p.content_mask = mask.clone();
                for vertex in &mut p.vertices {
                    vertex.content_mask = mask.clone();
                }
            }
            Primitive::Underline(u) => u.content_mask = mask,
            Primitive::MonochromeSprite(s) => s.content_mask = mask,
            Primitive::SubpixelSprite(s) => s.content_mask = mask,
            Primitive::PolychromeSprite(s) => s.content_mask = mask,
            Primitive::Surface(s) => s.content_mask = mask,
        }
    }
}

impl Scene {
    /// Build a mini-scene from a cache region's paint operations,
    /// with coordinates translated to texture-local space and content masks
    /// expanded to cover the full item bounds (preventing edge clipping).
    pub fn extract_region_as_mini_scene(&self, region: &CacheRegion) -> Scene {
        let mut mini = Scene::default();
        let offset = point(
            ScaledPixels(-region.bounds.origin.x.0),
            ScaledPixels(-region.bounds.origin.y.0),
        );
        let local_bounds = Bounds {
            origin: point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size: region.bounds.size,
        };
        let expanded_mask = ContentMask {
            bounds: local_bounds,
        };

        for op in &self.paint_operations[region.paint_op_range.clone()] {
            match op {
                PaintOperation::Primitive(prim) => {
                    let mut translated = prim.clone();
                    translated.translate_for_capture(offset);
                    translated.set_content_mask(expanded_mask.clone());
                    mini.insert_primitive(translated);
                }
                PaintOperation::StartLayer(bounds) => {
                    let mut translated = *bounds;
                    translated.origin += offset;
                    mini.push_layer(translated);
                }
                PaintOperation::EndLayer => mini.pop_layer(),
                PaintOperation::BeginCacheRegion(_) | PaintOperation::EndCacheRegion(_) => {}
            }
        }
        mini.finish();
        mini
    }
}
