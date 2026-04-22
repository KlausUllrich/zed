//! GPU texture caching for list item scroll compositing.
//!
//! Renders list items to offscreen GPU textures and composites them as quads
//! during scroll. Textures persist across frames and are reused until invalidated.
//!
//! Phase B: Memory budget enforcement, size-class bucketed free lists,
//! dynamic globals buffer. Phase 2: Priority-bin eviction — VISIBLE items never evicted,
//! DISTANT evicted first, within-bin tiebreak by distance from viewport center.
//! Textures allocated at exact content dimensions (the path
//! compositing pipeline maps UV via screen_position / viewport_size, which requires
//! textures to match content bounds exactly). Size classes organize the free list
//! for fast lookup; reuse requires exact (width, height) match.

use super::*;
use gpui::{CacheRegionId, Hsla, clear_cached_region};
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[cfg(feature = "texture-cache")]
use std::path::Path as StdPath;

// ── Constants ────────────────────────────────────────────────────────────────

/// Default memory budget: 128MB (was 64MB; increased to reduce eviction thrash
/// during fast scroll through 100+ cards — see GH #76).
const DEFAULT_BUDGET_BYTES: u64 = 128 * 1024 * 1024;

/// Frames an item can be outside viewport+buffer before becoming Distant (~0.5s at 60fps).
const RECENT_WINDOW: u64 = 30;

/// Bytes per pixel for RGBA8/BGRA8 surface formats.
const BYTES_PER_PIXEL: u64 = 4;

/// Percentage threshold for degraded-capture warning. If a recapture produces
/// fewer primitives than this fraction of the baseline, a warn log fires.
const DEGRADED_CAPTURE_THRESHOLD_PCT: u32 = 30;

/// Initial globals buffer capacity (items per frame that can be rendered fresh).
/// Replaces Phase A's hard-coded 16-item cap.
const INITIAL_GLOBALS_CAPACITY: u32 = 64;

// ── Card timeline PNG dump ───────────────────────────────────────────────────

/// A staged GPU readback for the card timeline diagnostic.
/// Created during process_cache_regions, flushed after queue.submit().
#[cfg(feature = "texture-cache")]
pub(crate) struct PendingTimelineDump {
    pub staging: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    pub padded_row_bytes: u32,
    pub unpadded_row_bytes: u32,
    pub region_id: u32,
    pub primitive_count: u32,
    pub sequence: u64,
    pub format: wgpu::TextureFormat,
}

// ── Debug callback types ─────────────────────────────────────────────────────

/// Per-frame debug data emitted by the texture cache for the F9 Cache tab.
/// Emitted during process_cache_regions (pre-pass). Fields:
/// - `fresh_ms`: this frame's texture rendering cost
/// - `composite_ms`: PREVIOUS frame's compositing cost (one-frame delay;
///   compositing happens after this callback fires)
pub struct TextureCacheDebugFrame {
    /// Items rendered fresh this frame (cache miss).
    pub fresh_count: u32,
    /// Items composited from cached textures (cache hit).
    pub cached_count: u32,
    /// Total textures currently in the pool (active + free).
    pub texture_count: u32,
    /// Estimated GPU memory used by texture pool (MB).
    pub memory_mb: f32,
    /// Time spent compositing cached textures in draw_cached_regions (ms).
    pub composite_ms: f32,
    /// Time spent rendering fresh items in process_cache_regions (ms).
    pub fresh_ms: f32,
    /// Per-item cache state for all visible regions.
    pub items: Vec<TextureCacheDebugItem>,
    /// Texture pool statistics.
    pub pool: TextureCacheDebugPool,
    /// Lifecycle events this frame.
    pub events: Vec<TextureCacheDebugLifecycle>,
}

/// Per-item cache state within a frame.
pub struct TextureCacheDebugItem {
    /// CacheRegionId value.
    pub index: u32,
    /// Cache state: "cached", "fresh", "too_large", "over_budget".
    pub state: &'static str,
    /// Texture width (0 if no texture).
    pub texture_width: u32,
    /// Texture height (0 if no texture).
    pub texture_height: u32,
    /// Frames since last used.
    pub age_frames: u32,
    /// Per-item render time (ms).
    pub last_render_ms: f32,
    /// Reason for state (e.g. "first_appearance", "size_change", "reused_from_pool").
    pub reason: Option<String>,
    /// Primitive counts from the mini-scene (only populated for fresh-rendered items).
    pub quads: u32,
    pub mono_sprites: u32,
    pub subpixel_sprites: u32,
    pub paths: u32,
}

/// Texture pool statistics.
pub struct TextureCacheDebugPool {
    /// Textures actively assigned to cached items.
    pub allocated: u32,
    /// Textures in the free list awaiting reuse.
    pub free: u32,
    /// Total textures (allocated + free).
    pub total: u32,
    /// Total GPU memory (active + free) in MB.
    pub memory_mb: f32,
    /// Memory budget in MB.
    pub budget_mb: f32,
    /// Cumulative eviction count this session.
    pub eviction_count: u32,
    /// Per-size-class breakdown.
    pub size_classes: Vec<TextureCacheDebugSizeClass>,
    /// Priority bin counts (from classify_entries). All zero if classification didn't run.
    pub bin_visible: u32,
    pub bin_buffer: u32,
    pub bin_recent: u32,
    pub bin_distant: u32,
}

/// Per-size-class pool statistics.
pub struct TextureCacheDebugSizeClass {
    /// Human-readable size class name.
    pub name: &'static str,
    /// Maximum height for this class (px).
    pub max_height: u32,
    /// Textures actively assigned to items in this class.
    pub active_count: u32,
    /// Textures in the free list for this class.
    pub free_count: u32,
    /// Total memory in this class (MB).
    pub memory_mb: f32,
}

/// Texture lifecycle event.
pub struct TextureCacheDebugLifecycle {
    /// "create", "evict", "reuse", "destroy", "invalidate".
    pub event_type: &'static str,
    pub index: u32,
    pub width: u32,
    pub height: u32,
    pub render_ms: f32,
    pub age_frames: u32,
    pub pool_bucket: Option<String>,
    pub reason: Option<String>,
}

// ── Debug callback registration ──────────────────────────────────────────────

type TextureCacheDebugCallback = Box<dyn Fn(TextureCacheDebugFrame) + Send + 'static>;

// Registration and usage both happen on the main thread — GPUI's renderer
// (WgpuRenderer::draw) runs on the main event loop, not a separate render thread.
// Cell<Option<*const _>> is sufficient — no lock needed, no Arc overhead.
// Box::leak ensures the callback lives for the program's lifetime.
thread_local! {
    static TEXTURE_CACHE_DEBUG_CB: Cell<Option<*const TextureCacheDebugCallback>> = const { Cell::new(None) };
}

/// Register a callback to receive per-frame GPU texture cache debug data.
/// Call once at app startup. Formats data for the F9 Cache tab in cs-debug.
/// Calling twice leaks the first callback (intentional: avoids a global lock).
pub fn set_texture_cache_debug_callback(callback: TextureCacheDebugCallback) {
    let leaked = Box::leak(Box::new(callback));
    TEXTURE_CACHE_DEBUG_CB.with(|cell| cell.set(Some(leaked as *const TextureCacheDebugCallback)));
}

fn has_debug_callback() -> bool {
    TEXTURE_CACHE_DEBUG_CB.with(|cell| cell.get().is_some())
}

fn emit_texture_cache_debug(frame: TextureCacheDebugFrame) {
    TEXTURE_CACHE_DEBUG_CB.with(|cell| {
        if let Some(ptr) = cell.get() {
            let cb = unsafe { &*ptr };
            cb(frame);
        }
    });
}

// ── Eviction callback ────────────────────────────────────────────────────────
// Feeds cs-debug Rule 3 (eviction_storm) via registration in cs-app/src/main.rs.
// Called from ensure_budget() on the render thread only.

/// Data passed to the eviction callback when a texture is evicted from the cache.
#[derive(Debug, Clone)]
pub struct TextureEvictionEvent {
    /// The region ID of the evicted texture.
    pub region_id: u64,
    /// Priority bin the evicted texture was in.
    pub priority_bin: &'static str,
    /// Item index of the evicted texture.
    pub item_index: usize,
    /// Reason for eviction.
    pub reason: &'static str,
}

type TextureEvictionCallback = Box<dyn Fn(TextureEvictionEvent) + Send + 'static>;

// Thread-local mirrors set_texture_cache_debug_callback's design.
// ensure_budget() is only called from the render thread, so thread-local
// avoids lock overhead on the hot path. Registered once at startup.
thread_local! {
    static TEXTURE_EVICTION_CB: Cell<Option<*const TextureEvictionCallback>> = const { Cell::new(None) };
}

/// Register a callback for GPU texture eviction events.
/// Call once at app startup. Follows the same pattern as `set_texture_cache_debug_callback`.
pub fn set_eviction_callback(callback: TextureEvictionCallback) {
    let leaked = Box::leak(Box::new(callback));
    TEXTURE_EVICTION_CB.with(|cell| cell.set(Some(leaked as *const TextureEvictionCallback)));
}

fn emit_eviction_event(event: TextureEvictionEvent) {
    TEXTURE_EVICTION_CB.with(|cell| {
        if let Some(ptr) = cell.get() {
            // SAFETY: Pointer was created by Box::leak in set_eviction_callback,
            // lives for program lifetime, and is only read (never mutated).
            let cb = unsafe { &*ptr };
            cb(event);
        }
    });
}

// ── Quality guard callback ────────────────────────────────────────────────────
// Fires when a re-capture produces significantly fewer primitives than baseline.
// Feeds cs-debug anomaly detection via registration in cs-app/src/main.rs.
// Called from process_cache_regions() on the render thread only.

/// Data passed to the quality guard callback when a degraded capture is detected.
#[derive(Debug, Clone)]
pub struct QualityGuardEvent {
    /// The region ID of the degraded texture.
    pub region_id: u64,
    /// Total primitives in the degraded capture.
    pub total: u32,
    /// Baseline primitive count from first successful capture.
    pub baseline: u32,
    /// Ratio of total/baseline (0.0–1.0).
    pub ratio: f32,
}

type QualityGuardCallback = Box<dyn Fn(QualityGuardEvent) + Send + 'static>;

// Thread-local mirrors set_eviction_callback's design.
// process_cache_regions() is only called from the render thread.
thread_local! {
    static QUALITY_GUARD_CB: Cell<Option<*const QualityGuardCallback>> = const { Cell::new(None) };
}

/// Register a callback for quality guard warnings (degraded primitive captures).
/// Call once at app startup. Follows the same pattern as `set_eviction_callback`.
pub fn set_quality_guard_callback(callback: QualityGuardCallback) {
    let leaked = Box::leak(Box::new(callback));
    QUALITY_GUARD_CB.with(|cell| cell.set(Some(leaked as *const QualityGuardCallback)));
}

fn emit_quality_guard_event(event: QualityGuardEvent) {
    QUALITY_GUARD_CB.with(|cell| {
        if let Some(ptr) = cell.get() {
            // SAFETY: Pointer was created by Box::leak in set_quality_guard_callback,
            // lives for program lifetime, and is only read (never mutated).
            let cb = unsafe { &*ptr };
            cb(event);
        }
    });
}

// ── Size classes ─────────────────────────────────────────────────────────────

/// Height-based size classes for free-list organization.
/// Textures are allocated at exact content dimensions; size classes determine
/// which free-list bucket a texture is stored in for O(1) lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SizeClass {
    /// ≤256px height — compact cards, tool summaries.
    Small,
    /// ≤512px — standard conversation items.
    Medium,
    /// ≤1024px — expanded code blocks, diffs.
    Large,
    /// ≤2048px — very tall items.
    XLarge,
    /// >2048px — exceptional items (up to max_texture_size).
    Oversize,
}

impl SizeClass {
    fn from_height(height: u32) -> Self {
        match height {
            0..=256 => SizeClass::Small,
            257..=512 => SizeClass::Medium,
            513..=1024 => SizeClass::Large,
            1025..=2048 => SizeClass::XLarge,
            _ => SizeClass::Oversize,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            SizeClass::Small => "small",
            SizeClass::Medium => "medium",
            SizeClass::Large => "large",
            SizeClass::XLarge => "xlarge",
            SizeClass::Oversize => "oversize",
        }
    }

    fn max_height(&self) -> u32 {
        match self {
            SizeClass::Small => 256,
            SizeClass::Medium => 512,
            SizeClass::Large => 1024,
            SizeClass::XLarge => 2048,
            SizeClass::Oversize => u32::MAX,
        }
    }

    const ALL: [SizeClass; 5] = [
        SizeClass::Small,
        SizeClass::Medium,
        SizeClass::Large,
        SizeClass::XLarge,
        SizeClass::Oversize,
    ];
}

// ── Priority bins for visibility-aware eviction ─────────────────────────────

/// Eviction priority based on visibility. Lower bins are evicted first.
/// Ord derive gives natural eviction ordering: Distant < Recent < Buffer < Visible.
/// INVARIANT: Declaration order = eviction order. New variants must be inserted
/// at the correct position to preserve this.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PriorityBin {
    /// Far from viewport — evicted FIRST.
    Distant,
    /// Left viewport within RECENT_WINDOW frames — brief grace period.
    Recent,
    /// In overdraw zone — evicted reluctantly.
    Buffer,
    /// In viewport — NEVER evicted.
    Visible,
}

// ── Pool data structures ─────────────────────────────────────────────────────

/// An active cached texture assigned to a specific list item.
struct CacheEntry {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    size_class: SizeClass,
    last_used_frame: u64,
    memory_bytes: u64,
    /// True when the texture was captured with actual primitives (total > 0).
    /// False for blank captures. HIT path rejects entries with has_content=false
    /// to break the stale-feedback → blank-texture cycle.
    has_content: bool,
    /// Visibility-based eviction priority, recomputed each frame by classify_entries().
    priority_bin: PriorityBin,
    /// Frame when this entry first left the viewport+buffer zone.
    /// `None` means the entry is currently inside viewport/buffer, or was just inserted
    /// (classify_entries sets it on the first outside-viewport frame).
    exit_frame: Option<u64>,
    /// Conversation item index (list index). Used for distance calculations
    /// in find_priority_victim(). INVARIANT: equals region_id (see list.rs paint loop).
    item_index: usize,
    /// Primitive count from first successful capture. process_cache_regions compares
    /// subsequent captures against this baseline to detect degraded renders.
    /// `None` until first capture with has_content == true.
    baseline_total: Option<u32>,
}

/// A recycled texture in the free list, available for reuse.
struct FreeTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    memory_bytes: u64,
}

fn texture_memory_bytes(width: u32, height: u32) -> u64 {
    width as u64 * height as u64 * BYTES_PER_PIXEL
}

/// Production-quality texture pool with LRU eviction, memory budget, and
/// size-class bucketed free lists.
///
/// Public API for other streams (flux, axle):
/// - `invalidate(region_id)` — remove a specific cached texture
/// - `evict_offscreen(visible_ids)` — evict textures for non-visible items
/// - `memory_stats()` — return current memory statistics
pub(crate) struct TexturePool {
    /// Active textures keyed by CacheRegionId.
    active: HashMap<u64, CacheEntry>,
    /// Free textures bucketed by size class, available for reuse.
    free_list: HashMap<SizeClass, Vec<FreeTexture>>,
    /// Current frame counter (incremented each frame).
    current_frame: u64,
    /// Total GPU memory across active + free textures (bytes).
    total_memory_bytes: u64,
    /// Memory budget (bytes). Allocations exceeding this trigger eviction.
    budget_bytes: u64,
    /// Cumulative eviction count (for debug reporting).
    eviction_count: u32,
    /// Region IDs that had blank textures rejected by the guard.
    /// Used for proof logging: when a guarded ID gets a successful recapture,
    /// we log `event=guard_heal` to confirm the self-heal cycle works.
    guarded_ids: HashSet<u64>,
    // Debug tint moved to thread-local in gpui::cache_region (set_debug_tint / is_debug_tint_enabled).
    /// Composite GPU pass duration from the previous frame (ms).
    /// Always zero until draw_cached_regions completes at least once.
    /// Debug callback reads this value one frame after it is written
    /// (process_cache_regions emits the report before draw_cached_regions runs).
    prev_composite_ms: f32,
    /// Uniform buffer for per-item viewport globals (one entry per fresh render).
    item_globals_buffer: wgpu::Buffer,
    /// Stride between entries in item_globals_buffer (alignment-padded).
    globals_entry_stride: u64,
    /// Current capacity of the globals buffer (number of entries).
    globals_capacity: u32,
    /// IP-2 V1: every `region_id` that entered `process_cache_regions` this
    /// frame. Cleared at the top of the function, pushed per-region in the
    /// loop, checked pre-submit to detect mid-frame destroys (RW1 / RW3).
    /// Only present under `texture-cache-debug` — zero cost in release.
    #[cfg(feature = "texture-cache-debug")]
    frame_regions_processed: Vec<u64>,
}

/// Memory statistics returned by `TexturePool::memory_stats()`.
#[allow(dead_code)] // Public API for other streams (flux, axle)
pub(crate) struct TextureCacheMemoryStats {
    pub active_count: u32,
    pub free_count: u32,
    pub total_memory_bytes: u64,
    pub budget_bytes: u64,
    pub eviction_count: u32,
}

impl TexturePool {
    /// Mark the start of a new frame. Advances the frame counter for LRU tracking.
    fn begin_frame(&mut self) {
        self.current_frame += 1;
    }

    /// IP-2 V1: return every `region_id` that entered this frame's
    /// `process_cache_regions` but is no longer in `active`. A non-empty
    /// vector means a cache entry was destroyed after the encoder already
    /// bound its view — exactly the RW1 / RW3 race. Caller emits an
    /// `INVARIANT_VIOLATION rule=V1` event per id and skips `queue.submit`.
    ///
    /// The one *intentional* removal this frame is the `empty_recapture`
    /// guard at `texture_cache.rs:1231`. That ALSO emits an
    /// `event=texture_destroy reason=stale_hit_cleanup` (Gap-NEW-1 site 4)
    /// paired with the V1 violation — the two together confirm the race.
    #[cfg(feature = "texture-cache-debug")]
    pub(crate) fn check_pre_submit_liveness(&self) -> Vec<u64> {
        self.frame_regions_processed
            .iter()
            .copied()
            .filter(|id| !self.active.contains_key(id))
            .collect()
    }

    /// IP-2 V1: drop any frame-scoped V1 state after submit. Called
    /// unconditionally at the end of the draw function so the next frame
    /// starts with an empty processed set regardless of whether submit fired.
    #[cfg(feature = "texture-cache-debug")]
    pub(crate) fn clear_frame_regions_processed(&mut self) {
        self.frame_regions_processed.clear();
    }

    /// Reclassify all active entries into priority bins based on current viewport state.
    /// Called by: `WgpuRenderer::process_cache_regions()` at the start of each frame,
    /// after `begin_frame()` and before any capture decisions.
    fn classify_entries(
        &mut self,
        visible_ids: &HashSet<u64>,
        buffer_ids: &HashSet<u64>,
    ) {
        let current_frame = self.current_frame;
        for (region_id, entry) in self.active.iter_mut() {
            if visible_ids.contains(region_id) {
                entry.priority_bin = PriorityBin::Visible;
                entry.exit_frame = None;
            } else if buffer_ids.contains(region_id) {
                entry.priority_bin = PriorityBin::Buffer;
                entry.exit_frame = None;
            } else {
                // Outside viewport+buffer — classify by recency
                match entry.exit_frame {
                    None => {
                        // First frame outside viewport+buffer
                        entry.exit_frame = Some(current_frame);
                        entry.priority_bin = PriorityBin::Recent;
                    }
                    Some(f) if current_frame.saturating_sub(f) < RECENT_WINDOW => {
                        entry.priority_bin = PriorityBin::Recent;
                    }
                    Some(_) => {
                        entry.priority_bin = PriorityBin::Distant;
                    }
                }
            }

            #[cfg(feature = "texture-cache-debug")]
            log::debug!(
                "classify region={} bin={:?} exit_frame={:?}",
                region_id,
                entry.priority_bin,
                entry.exit_frame,
            );
        }
    }

    /// Try to find a reusable texture from the free list with exact dimensions.
    fn try_reuse(&mut self, width: u32, height: u32) -> Option<FreeTexture> {
        let size_class = SizeClass::from_height(height);
        let bucket = self.free_list.get_mut(&size_class)?;
        let idx = bucket
            .iter()
            .position(|t| t.width == width && t.height == height)?;
        Some(bucket.swap_remove(idx))
    }

    /// Move an active entry to the free list (eviction without GPU deallocation).
    fn release_to_free_list(&mut self, region_id: u64) {
        if let Some(entry) = self.active.remove(&region_id) {
            // Gap-NEW-1 site 5: active→free_list transition. Paired with any
            // future texture_destroy that drains the free list (site 1).
            #[cfg(feature = "texture-cache-debug")]
            log::info!(
                "event=texture_invalidate ix={} destination=free_list size={}x{}",
                region_id, entry.width, entry.height
            );
            let sc = entry.size_class;
            let free = FreeTexture {
                texture: entry.texture,
                view: entry.view,
                width: entry.width,
                height: entry.height,
                memory_bytes: entry.memory_bytes,
            };
            self.free_list.entry(sc).or_default().push(free);
            // total_memory_bytes unchanged — texture moved, not destroyed
        }
    }

    /// Destroy a single free-list texture, reclaiming GPU memory.
    /// Prefers destroying from the largest size class first (frees more memory).
    /// Returns bytes freed, or 0 if free list is empty.
    fn destroy_any_free(&mut self) -> u64 {
        for class in SizeClass::ALL.iter().rev() {
            if let Some(bucket) = self.free_list.get_mut(class) {
                if let Some(freed) = bucket.pop() {
                    self.total_memory_bytes -= freed.memory_bytes;
                    // Gap-NEW-1 site 1: free-list texture destroyed under budget
                    // pressure. `ix=none` because free-list entries are identity-
                    // less once released from active.
                    #[cfg(feature = "texture-cache-debug")]
                    log::info!(
                        "event=texture_destroy ix=none reason=budget_free_drained class={:?} size={}x{} bytes={}",
                        class, freed.width, freed.height, freed.memory_bytes
                    );
                    drop(freed.texture);
                    return freed.memory_bytes;
                }
            }
        }
        0
    }

    /// Destroy an active entry, reclaiming GPU memory.
    fn destroy_active(&mut self, region_id: u64) {
        if let Some(entry) = self.active.remove(&region_id) {
            self.total_memory_bytes -= entry.memory_bytes;
            // Gap-NEW-1 site 2: active-slot victim under budget pressure.
            // Paired with the `event=eviction` emit that fires before this
            // call in ensure_budget (see line ~605).
            #[cfg(feature = "texture-cache-debug")]
            log::info!(
                "event=texture_destroy ix={} reason=budget_victim_priority bin={:?} item_index={} size={}x{}",
                region_id, entry.priority_bin, entry.item_index, entry.width, entry.height
            );
            drop(entry.texture);
        }
    }

    /// Find the best eviction victim using priority-bin ordering.
    /// VISIBLE items are never evicted. Within the same bin, items farthest
    /// from the viewport center are preferred (less likely to be scrolled to).
    fn find_priority_victim(&self, viewport_center_index: usize) -> Option<u64> {
        let mut best: Option<(u64, PriorityBin, usize)> = None; // (region_id, bin, distance)

        for (&region_id, entry) in &self.active {
            if entry.priority_bin == PriorityBin::Visible {
                continue; // NEVER evict visible
            }
            let dist =
                (entry.item_index as isize - viewport_center_index as isize).unsigned_abs();
            match &best {
                None => best = Some((region_id, entry.priority_bin, dist)),
                Some((_, best_bin, best_dist)) => {
                    // Lower bin = evict first. Same bin = farthest from viewport first.
                    if entry.priority_bin < *best_bin
                        || (entry.priority_bin == *best_bin && dist > *best_dist)
                    {
                        best = Some((region_id, entry.priority_bin, dist));
                    }
                }
            }
        }
        best.map(|(id, _, _)| id)
    }

    /// Ensure memory budget allows `needed_bytes` of new allocation.
    /// Two-phase eviction: (1) destroy free-list textures, (2) evict by priority bin
    /// (Distant first, then Recent, then Buffer — VISIBLE items are never evicted).
    /// Returns true if budget allows the allocation, false if only VISIBLE items remain.
    fn ensure_budget(&mut self, needed_bytes: u64, viewport_center_index: usize) -> bool {
        if self.total_memory_bytes + needed_bytes <= self.budget_bytes {
            return true;
        }

        // Phase 1: destroy free-list textures (cheapest — no re-render needed)
        while self.total_memory_bytes + needed_bytes > self.budget_bytes {
            if self.destroy_any_free() == 0 {
                break;
            }
        }

        // Phase 2: evict by priority bin — lowest bin first, farthest from viewport within bin.
        // VISIBLE items are never evicted; if only VISIBLE remain, return false → render Fresh.
        while self.total_memory_bytes + needed_bytes > self.budget_bytes {
            if let Some(victim_id) = self.find_priority_victim(viewport_center_index) {
                // Log + callback before destroy removes the entry from the map.
                if let Some(entry) = self.active.get(&victim_id) {
                    let bin_str = match entry.priority_bin {
                        PriorityBin::Distant => "Distant",
                        PriorityBin::Recent => "Recent",
                        PriorityBin::Buffer => "Buffer",
                        PriorityBin::Visible => "Visible",
                    };
                    log::info!(
                        "event=eviction region={} bin={:?} item_index={} reason=budget_pressure",
                        victim_id, entry.priority_bin, entry.item_index,
                    );
                    emit_eviction_event(TextureEvictionEvent {
                        region_id: victim_id,
                        priority_bin: bin_str,
                        item_index: entry.item_index,
                        reason: "budget_pressure",
                    });
                }
                self.destroy_active(victim_id);
                self.eviction_count += 1;
            } else {
                log::info!("event=eviction_blocked reason=all_visible");
                emit_eviction_event(TextureEvictionEvent {
                    region_id: 0,
                    priority_bin: "none",
                    item_index: 0,
                    reason: "eviction_blocked_all_visible",
                });
                return false;
            }
        }

        true
    }

    /// Insert a newly rendered texture into the active map.
    fn insert(
        &mut self,
        region_id: u64,
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        width: u32,
        height: u32,
        has_content: bool,
    ) {
        let memory_bytes = texture_memory_bytes(width, height);
        let size_class = SizeClass::from_height(height);

        // If replacing an existing entry, account for memory
        if let Some(old) = self.active.remove(&region_id) {
            self.total_memory_bytes -= old.memory_bytes;
            // Gap-NEW-1 site 3: silent replace-path drop. RW3 smoking gun —
            // if a bind group still references `old.view`, submit validates
            // against a dropped texture after this line.
            #[cfg(feature = "texture-cache-debug")]
            log::info!(
                "event=texture_destroy ix={} reason=replace_active old_size={}x{} old_bytes={}",
                region_id, old.width, old.height, old.memory_bytes
            );
            drop(old.texture);
        }

        self.total_memory_bytes += memory_bytes;
        self.active.insert(
            region_id,
            CacheEntry {
                texture,
                view,
                width,
                height,
                size_class,
                last_used_frame: self.current_frame,
                memory_bytes,
                has_content,
                priority_bin: PriorityBin::Distant,
                exit_frame: None,
                item_index: region_id as usize,
                baseline_total: None,
            },
        );
    }

    /// Insert a reused free-list texture into the active map.
    /// Memory accounting: no change — texture was already counted when in the free list.
    fn reactivate(
        &mut self,
        region_id: u64,
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        width: u32,
        height: u32,
        has_content: bool,
    ) {
        let size_class = SizeClass::from_height(height);
        let memory_bytes = texture_memory_bytes(width, height);
        self.active.insert(
            region_id,
            CacheEntry {
                texture,
                view,
                width,
                height,
                size_class,
                last_used_frame: self.current_frame,
                memory_bytes,
                has_content,
                priority_bin: PriorityBin::Distant,
                exit_frame: None,
                item_index: region_id as usize,
                baseline_total: None,
            },
        );
    }

    // ── Public API (for flux, axle, obedi) ───────────────────────────────────

    /// Invalidate a specific cached region. Moves the texture to the free list
    /// for potential reuse (does not destroy GPU resources).
    #[allow(dead_code)] // Public API for Stream 2 (flux) and Stream 4 (axle)
    pub fn invalidate(&mut self, region_id: u64) {
        self.release_to_free_list(region_id);
    }

    /// Evict all textures for items not in the visible set.
    /// Moves evicted textures to the free list. Returns the number evicted.
    ///
    /// Callers should include BOTH viewport AND overdraw item IDs in `visible_ids`.
    /// Overdraw items (those just outside the viewport, tracked by `ItemLayout.is_overdraw`
    /// in list.rs) should be kept cached for scroll readiness — they are lower priority
    /// than viewport items but higher than fully off-screen items.
    #[allow(dead_code)] // Public API for Stream 2 (flux) and Stream 4 (axle)
    pub fn evict_offscreen(&mut self, visible_ids: &HashSet<u64>) -> u32 {
        let to_evict: Vec<u64> = self
            .active
            .keys()
            .filter(|id| !visible_ids.contains(id))
            .cloned()
            .collect();
        let count = to_evict.len() as u32;
        for id in to_evict {
            self.release_to_free_list(id);
        }
        count
    }

    /// Return current memory statistics.
    #[allow(dead_code)] // Public API for Stream 2 (flux) and Stream 4 (axle)
    pub fn memory_stats(&self) -> TextureCacheMemoryStats {
        let free_count: u32 = self
            .free_list
            .values()
            .map(|bucket| bucket.len() as u32)
            .sum();
        TextureCacheMemoryStats {
            active_count: self.active.len() as u32,
            free_count,
            total_memory_bytes: self.total_memory_bytes,
            budget_bytes: self.budget_bytes,
            eviction_count: self.eviction_count,
        }
    }

    /// Collect active region IDs that have real content (for skip-paint decisions).
    /// Entries with has_content=false (empty captures) are excluded so that
    /// has_cached_region() returns false, forcing a full render on the next frame.
    fn active_region_ids(&self) -> HashSet<u64> {
        self.active.iter()
            .filter(|(_, entry)| entry.has_content)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Build debug pool statistics.
    fn debug_pool_stats(&self) -> TextureCacheDebugPool {
        let active_count = self.active.len() as u32;
        let free_count: u32 = self
            .free_list
            .values()
            .map(|b| b.len() as u32)
            .sum();

        let size_classes: Vec<TextureCacheDebugSizeClass> = SizeClass::ALL
            .iter()
            .map(|class| {
                let ac = self
                    .active
                    .values()
                    .filter(|e| e.size_class == *class)
                    .count() as u32;
                let fc = self
                    .free_list
                    .get(class)
                    .map(|b| b.len() as u32)
                    .unwrap_or(0);
                let active_mem: u64 = self
                    .active
                    .values()
                    .filter(|e| e.size_class == *class)
                    .map(|e| e.memory_bytes)
                    .sum();
                let free_mem: u64 = self
                    .free_list
                    .get(class)
                    .map(|b| b.iter().map(|t| t.memory_bytes).sum())
                    .unwrap_or(0);
                TextureCacheDebugSizeClass {
                    name: class.name(),
                    max_height: class.max_height(),
                    active_count: ac,
                    free_count: fc,
                    memory_mb: (active_mem + free_mem) as f32 / (1024.0 * 1024.0),
                }
            })
            .collect();

        let (mut bin_vis, mut bin_buf, mut bin_rec, mut bin_dist) = (0u32, 0u32, 0u32, 0u32);
        for entry in self.active.values() {
            match entry.priority_bin {
                PriorityBin::Visible => bin_vis += 1,
                PriorityBin::Buffer => bin_buf += 1,
                PriorityBin::Recent => bin_rec += 1,
                PriorityBin::Distant => bin_dist += 1,
            }
        }

        TextureCacheDebugPool {
            allocated: active_count,
            free: free_count,
            total: active_count + free_count,
            memory_mb: self.total_memory_bytes as f32 / (1024.0 * 1024.0),
            budget_mb: self.budget_bytes as f32 / (1024.0 * 1024.0),
            eviction_count: self.eviction_count,
            size_classes,
            bin_visible: bin_vis,
            bin_buffer: bin_buf,
            bin_recent: bin_rec,
            bin_distant: bin_dist,
        }
    }
}

// ── Color conversion ─────────────────────────────────────────────────────────

fn hsla_to_wgpu_color(color: Hsla) -> wgpu::Color {
    let h = color.h;
    let s = color.s;
    let l = color.l;
    let a = color.a;
    let (r, g, b) = if s == 0.0 {
        (l, l, l)
    } else {
        let hue_to_rgb = |p: f32, q: f32, mut t: f32| -> f32 {
            if t < 0.0 {
                t += 1.0;
            }
            if t > 1.0 {
                t -= 1.0;
            }
            if t < 1.0 / 6.0 {
                return p + (q - p) * 6.0 * t;
            }
            if t < 1.0 / 2.0 {
                return q;
            }
            if t < 2.0 / 3.0 {
                return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
            }
            p
        };
        let q = if l < 0.5 {
            l * (1.0 + s)
        } else {
            l + s - l * s
        };
        let p = 2.0 * l - q;
        (
            hue_to_rgb(p, q, h + 1.0 / 3.0),
            hue_to_rgb(p, q, h),
            hue_to_rgb(p, q, h - 1.0 / 3.0),
        )
    };
    wgpu::Color {
        r: r as f64,
        g: g as f64,
        b: b as f64,
        a: a as f64,
    }
}

// ── WgpuRenderer integration ─────────────────────────────────────────────────

impl WgpuRenderer {
    fn ensure_texture_pool(&mut self) {
        if self.texture_pool.is_some() {
            return;
        }
        let resources = self.resources();
        let alignment = resources.device.limits().min_uniform_buffer_offset_alignment as u64;
        let globals_size = std::mem::size_of::<GlobalParams>() as u64;
        let entry_stride = globals_size.next_multiple_of(alignment);
        let capacity = INITIAL_GLOBALS_CAPACITY;
        let item_globals_buffer = resources.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("item_texture_globals"),
            size: entry_stride * capacity as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.texture_pool = Some(TexturePool {
            active: HashMap::new(),
            free_list: HashMap::new(),
            current_frame: 0,
            total_memory_bytes: 0,
            budget_bytes: DEFAULT_BUDGET_BYTES,
            eviction_count: 0,
            guarded_ids: HashSet::new(),
            prev_composite_ms: 0.0,
            item_globals_buffer,
            globals_entry_stride: entry_stride,
            globals_capacity: capacity,
            #[cfg(feature = "texture-cache-debug")]
            frame_regions_processed: Vec::new(),
        });
    }

    /// Grow the globals buffer if the current capacity is insufficient.
    fn ensure_globals_capacity(&mut self, needed: u32) {
        let current_capacity = self
            .texture_pool
            .as_ref()
            .map(|p| p.globals_capacity)
            .unwrap_or(0);
        if needed <= current_capacity {
            return;
        }
        let stride = self
            .texture_pool
            .as_ref()
            .expect("texture pool must exist")
            .globals_entry_stride;
        let new_capacity = (needed * 2).max(INITIAL_GLOBALS_CAPACITY);
        let resources = self.resources();
        let new_buffer = resources.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("item_texture_globals"),
            size: stride * new_capacity as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pool = self
            .texture_pool
            .as_mut()
            .expect("texture pool must exist");
        pool.item_globals_buffer = new_buffer;
        pool.globals_capacity = new_capacity;
    }

    fn create_item_texture(
        &self,
        region_id: u32,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let resources = self.resources();
        let texture = resources.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("item_cache_texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        // IP-1: Named view so wgpu validation errors identify the specific
        // cache region. An empty label surfaces as `''` in panic messages and
        // was the central blind spot in S506 (PLAN.md Option B, V2 invariant).
        let view_label = format!("cache_item_{region_id}_{width}x{height}");
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&view_label),
            ..Default::default()
        });
        (texture, view)
    }

    /// Pre-pass: render dirty cache regions to offscreen textures.
    /// Textures persist across frames — only re-rendered on cache miss or size change.
    /// Phase B: LRU tracking, memory budget enforcement, free-list reuse.
    pub(crate) fn process_cache_regions(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &Scene,
        instance_offset: &mut u64,
    ) -> bool {
        // IP-2 V1: reset per-frame processed set unconditionally. Even on
        // empty-regions frames we must clear so stale entries from a prior
        // frame don't cause false V1 violations later. `if let Some` avoids
        // forcing pool creation on frames where there's nothing to do.
        #[cfg(feature = "texture-cache-debug")]
        if let Some(pool) = self.texture_pool.as_mut() {
            pool.frame_regions_processed.clear();
        }

        let regions: Vec<_> = scene.cache_regions().to_vec();
        if regions.is_empty() {
            return true;
        }

        self.ensure_texture_pool();

        // Check if list.rs requested a full pool flush (DPI, theme, font change)
        if gpui::take_pending_pool_invalidation() {
            self.invalidate_texture_cache();
        }

        // Purge specific items invalidated by list.rs (e.g. user interaction on a card).
        // Moves textures to the free list for reuse rather than destroying GPU resources.
        let invalidated = gpui::take_invalidated_region_ids();
        if !invalidated.is_empty() {
            let pool = self.texture_pool.as_mut().unwrap();
            for region_id in &invalidated {
                pool.invalidate(*region_id);
            }
        }

        // Advance frame counter for LRU tracking
        self.texture_pool.as_mut().unwrap().begin_frame();

        // Classify active cache entries into priority bins based on viewport state.
        // Must run before any capture decisions so eviction uses current-frame data.
        // Skip when list.rs reported no items (hidden panel, not yet laid out) —
        // entries keep their previous-frame bin until budget eviction clears them.
        let (visible_ids, buffer_ids) = gpui::take_classification_ids();
        if !visible_ids.is_empty() || !buffer_ids.is_empty() {
            self.texture_pool
                .as_mut()
                .unwrap()
                .classify_entries(&visible_ids, &buffer_ids);
        }
        // Viewport center for distance-based eviction within priority bins.
        // Falls back to 0 when list.rs didn't report (hidden panel, first frame).
        let viewport_center_index = gpui::take_viewport_center_index().unwrap_or(0);

        // Grow globals buffer if needed (removes Phase A's 16-item cap)
        self.ensure_globals_capacity(regions.len() as u32);

        let debug = has_debug_callback();
        let start = if debug { Some(Instant::now()) } else { None };
        let mut debug_items: Vec<TextureCacheDebugItem> = Vec::new();
        let mut debug_events: Vec<TextureCacheDebugLifecycle> = Vec::new();
        let mut fresh_count: u32 = 0;
        let mut cached_count: u32 = 0;

        let globals_size = std::mem::size_of::<GlobalParams>() as u64;
        let mut render_idx: usize = 0;

        // CS S499: snapshot eviction counter before per-region processing so we can
        // emit a single aggregated "burst" event at end-of-loop rather than rely on
        // the per-eviction log lines. A burst of 51 evictions tightly clustered right
        // before the silent CACHE->DIRECT window would implicate hypothesis A
        // (mass texture destruction at caching_disable).
        #[cfg(feature = "texture-cache-debug")]
        let eviction_count_before = self
            .texture_pool
            .as_ref()
            .map(|p| p.eviction_count)
            .unwrap_or(0);

        for region in &regions {
            let tex_width = (region.bounds.size.width.0.ceil() as u32).max(1);
            let tex_height = (region.bounds.size.height.0.ceil() as u32).max(1);
            let region_id = region.id.0 as u32;

            // IP-2 V1: remember this region entered the pass. If it is NOT
            // in `pool.active` at submit time, something removed it mid-frame.
            #[cfg(feature = "texture-cache-debug")]
            self.texture_pool
                .as_mut()
                .unwrap()
                .frame_regions_processed
                .push(region.id.0);

            // FR-5.1/EC-14: Skip items exceeding GPU max texture dimension.
            // Falls back to Fresh rendering (no caching attempt).
            if tex_width > self.max_texture_size || tex_height > self.max_texture_size {
                log::debug!(
                    "texture cache: item {} ({}x{}) exceeds max texture dimension {} — fallback to Fresh",
                    region_id, tex_width, tex_height, self.max_texture_size
                );
                if debug {
                    debug_items.push(TextureCacheDebugItem {
                        index: region_id,
                        state: "too_large",
                        texture_width: tex_width,
                        texture_height: tex_height,
                        age_frames: 0,
                        last_render_ms: 0.0,
                        reason: Some(format!(
                            "exceeds max_texture_dimension_2d ({})",
                            self.max_texture_size
                        )),
                        quads: 0,
                        mono_sprites: 0,
                        subpixel_sprites: 0,
                        paths: 0,
                    });
                }
                continue;
            }

            // Cache hit: texture exists with matching dimensions — touch and reuse
            {
                let pool = self.texture_pool.as_mut().unwrap();
                if let Some(cached) = pool.active.get_mut(&region.id.0) {
                    if cached.width == tex_width && cached.height == tex_height && cached.has_content {
                        cached.last_used_frame = pool.current_frame;
                        cached_count += 1;
                        if debug {
                            debug_items.push(TextureCacheDebugItem {
                                index: region_id,
                                state: "cached",
                                texture_width: cached.width,
                                texture_height: cached.height,
                                age_frames: 0,
                                last_render_ms: 0.0,
                                reason: None,
                                quads: 0,
                                mono_sprites: 0,
                                subpixel_sprites: 0,
                                paths: 0,
                            });
                        }
                        continue;
                    }
                    // S504: split emit — dim-mismatch and has_content=false are
                    // distinct cases. Prior single-emit path misrepresented 99.7%
                    // of fires as dimension mismatches when dims matched exactly
                    // and only has_content differed (flux S504 investigation).
                    if cached.width != tex_width || cached.height != tex_height {
                        log::debug!(
                            "event=dimension_mismatch ix={} cached_w={} cached_h={} tex_w={} tex_h={} action=recapture",
                            region_id, cached.width, cached.height, tex_width, tex_height
                        );
                        // S498 INV-X4: dimension mismatch triggers recapture loop.
                        #[cfg(feature = "texture-cache-debug")]
                        log::warn!(
                            "event=INVARIANT_VIOLATION rule=X4 region={} cached_w={} cached_h={} actual_w={} actual_h={}",
                            region_id, cached.width, cached.height, tex_width, tex_height
                        );
                    } else {
                        // Dims match, has_content=false — re-capture is driven by
                        // an empty prior capture (total=0 primitives). Not an
                        // X4 violation; no anomaly rule consumes this event.
                        log::debug!(
                            "event=empty_capture_retain ix={} tex_w={} tex_h={} action=recapture",
                            region_id, tex_width, tex_height
                        );
                    }
                }
            }

            // Try reusing a free-list texture BEFORE checking budget.
            // Reuse costs zero new memory — ensure_budget might needlessly evict
            // the very texture we'd reuse if called first.
            let (texture, view, reused) = {
                let pool = self.texture_pool.as_mut().unwrap();
                if let Some(free) = pool.try_reuse(tex_width, tex_height) {
                    (free.texture, free.view, true)
                } else {
                    // No reusable texture — check memory budget before allocating new
                    let needed_bytes = texture_memory_bytes(tex_width, tex_height);
                    if gpui::is_budget_eviction_enabled() && !pool.ensure_budget(needed_bytes, viewport_center_index) {
                        if debug {
                            debug_items.push(TextureCacheDebugItem {
                                index: region_id,
                                state: "over_budget",
                                texture_width: 0,
                                texture_height: 0,
                                age_frames: 0,
                                last_render_ms: 0.0,
                                reason: Some("over_budget".into()),
                                quads: 0,
                                mono_sprites: 0,
                                subpixel_sprites: 0,
                                paths: 0,
                            });
                        }
                        continue;
                    }
                    // NLL ends the &mut pool borrow here — create_item_texture can borrow &self
                    let (t, v) = self.create_item_texture(region_id, tex_width, tex_height);
                    (t, v, false)
                }
            };

            // Cache miss or dimension change — render to texture
            let render_start = if debug { Some(Instant::now()) } else { None };
            let mini_scene = scene.extract_region_as_mini_scene(region);

            // Per-type sprite counts — `prim_*` copies feed the F9 debug panel below.
            let q = mini_scene.quads.len() as u32;
            let m = mini_scene.monochrome_sprites.len() as u32;
            let s = mini_scene.subpixel_sprites.len() as u32;
            let p = mini_scene.paths.len() as u32;
            let total = q
                + m
                + s
                + p
                + mini_scene.polychrome_sprites.len() as u32
                + mini_scene.shadows.len() as u32
                + mini_scene.underlines.len() as u32;
            #[cfg(feature = "texture-cache-debug")]
            {
                let poly = mini_scene.polychrome_sprites.len() as u32;
                let shadows = mini_scene.shadows.len() as u32;
                let underlines = mini_scene.underlines.len() as u32;
                log::info!(
                    "event=capture_detail ix={} texture={}x{} total={} quads={} mono={} subpixel={} paths={} polychrome={} shadows={} underlines={} reused={}",
                    region_id, tex_width, tex_height, total, q, m, s, p, poly, shadows, underlines, reused
                );
                log::info!(
                    "event=mini_scene ix={} region_y={:.1} region_h={:.1} quads={} shadows={} mono={} subpixel={} poly={} underlines={} paths={} total={}",
                    region_id,
                    region.bounds.origin.y.0,
                    region.bounds.size.height.0,
                    q, shadows, m, s, poly, underlines, p, total,
                );
                if let Some(first_mono) = mini_scene.monochrome_sprites.first() {
                    log::info!("event=mini_scene_sprite ix={} type=mono first_y={:.1} trans_y={:.1}",
                        region_id, first_mono.bounds.origin.y.0, first_mono.transformation.translation[1]);
                }
                if let Some(first_quad) = mini_scene.quads.first() {
                    log::info!("event=mini_scene_sprite ix={} type=quad first_y={:.1}",
                        region_id, first_quad.bounds.origin.y.0);
                }
            }

            // Guard: reject empty mini-scenes caused by stale HIT feedback.
            // When list.rs skips paint (cache HIT) but the renderer evicted the texture,
            // we get total=0. Don't create a blank texture — remove the stale feedback
            // so list.rs paints on the next frame (self-healing in 1 frame).
            if total == 0 && tex_height > 16 {
                log::warn!(
                    "event=empty_recapture ix={} size={}x{} — stale HIT feedback, skipping",
                    region_id, tex_width, tex_height
                );
                // Return the reused texture to free list if we grabbed one
                if reused {
                    let sc = SizeClass::from_height(tex_height);
                    let pool = self.texture_pool.as_mut().unwrap();
                    pool.free_list.entry(sc).or_default().push(FreeTexture {
                        texture, view, width: tex_width, height: tex_height,
                        memory_bytes: texture_memory_bytes(tex_width, tex_height),
                    });
                }
                // Ensure this region is NOT in active pool — on next frame,
                // has_cached_region() returns false → list.rs paints fresh
                let pool = self.texture_pool.as_mut().unwrap();
                // Gap-NEW-1 site 4: RW1 smoking gun. The mid-loop remove that
                // sage's code-path analysis named as the race — drops texture
                // + view via CacheEntry Drop while the encoder may still hold
                // an Arc to the view.
                #[cfg(feature = "texture-cache-debug")]
                log::info!(
                    "event=texture_destroy ix={} reason=stale_hit_cleanup size={}x{}",
                    region.id.0, tex_width, tex_height
                );
                pool.active.remove(&region.id.0);
                pool.guarded_ids.insert(region.id.0);
                // Also clear the thread-local feedback so list.rs sees MISS immediately
                clear_cached_region(CacheRegionId(region.id.0));
                continue;
            }

            // Capture counts for F9 debug item (available to debug callback below).
            let (prim_quads, prim_mono, prim_subpixel, prim_paths) = (q, m, s, p);

            // Write per-item viewport globals at a unique offset
            let pool = self.texture_pool.as_ref().unwrap();
            let entry_offset = (render_idx as u64) * pool.globals_entry_stride;
            let item_globals = GlobalParams {
                viewport_size: [tex_width as f32, tex_height as f32],
                premultiplied_alpha: 0,
                // S502: capture-pass globals — fade alpha is never applied to
                // capture (texture ingestion, not composite). Set to 1.0 to
                // keep the field's "no-op" default explicit.
                composite_fade_alpha: 1.0,
            };
            let resources = self.resources();
            resources.queue.write_buffer(
                &pool.item_globals_buffer,
                entry_offset,
                bytemuck::bytes_of(&item_globals),
            );

            // Create bind group pointing to this item's globals + shared gamma
            let item_bind_group =
                resources
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("item_globals_bind_group"),
                        layout: &resources.bind_group_layouts.globals,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &pool.item_globals_buffer,
                                    offset: entry_offset,
                                    size: Some(NonZeroU64::new(globals_size).unwrap()),
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &resources.globals_buffer,
                                    offset: self.gamma_offset,
                                    size: Some(
                                        NonZeroU64::new(
                                            std::mem::size_of::<GammaParams>() as u64,
                                        )
                                        .unwrap(),
                                    ),
                                }),
                            },
                        ],
                    });

            // Render mini-scene to offscreen texture.
            // When debug tint is active, override the clear color to opaque magenta.
            // If magenta cards appear during scroll, the composite pipeline works —
            // the issue is in the texture content. If still invisible, the composite
            // pipeline itself is broken.
            let clear_color = if gpui::is_debug_tint_enabled() {
                wgpu::Color { r: 1.0, g: 0.0, b: 1.0, a: 1.0 }
            } else {
                hsla_to_wgpu_color(region.clear_color)
            };
            if !self.render_mini_scene_to_texture(
                encoder,
                &mini_scene,
                &view,
                clear_color,
                &item_bind_group,
                instance_offset,
            ) {
                // Return reused texture to free list to avoid orphaning GPU resource
                if reused {
                    let sc = SizeClass::from_height(tex_height);
                    let pool = self.texture_pool.as_mut().unwrap();
                    pool.free_list.entry(sc).or_default().push(FreeTexture {
                        texture,
                        view,
                        width: tex_width,
                        height: tex_height,
                        memory_bytes: texture_memory_bytes(tex_width, tex_height),
                    });
                }
                return false;
            }

            // Store in pool
            let render_ms = render_start
                .map(|s| s.elapsed().as_secs_f32() * 1000.0)
                .unwrap_or(0.0);
            let reason = if reused {
                "reused_from_pool"
            } else {
                "first_appearance"
            };
            let pool = self.texture_pool.as_mut().unwrap();
            let has_content = total > 0;
            if reused {
                pool.reactivate(region.id.0, texture, view, tex_width, tex_height, has_content);
            } else {
                pool.insert(region.id.0, texture, view, tex_width, tex_height, has_content);
            }

            // Guard baseline tracking: record first successful primitive count,
            // warn on subsequent captures that drop below 30% of baseline.
            if let Some(entry) = pool.active.get_mut(&region.id.0) {
                if has_content {
                    match entry.baseline_total {
                        None => {
                            entry.baseline_total = Some(total);
                        }
                        Some(baseline)
                            if total * 100
                                < baseline * DEGRADED_CAPTURE_THRESHOLD_PCT =>
                        {
                            let ratio = total as f32 / baseline as f32;
                            log::warn!(
                                "event=low_primitive_capture region={} total={} baseline={} ratio={:.1}%",
                                region_id,
                                total,
                                baseline,
                                ratio * 100.0,
                            );
                            emit_quality_guard_event(QualityGuardEvent {
                                region_id: region.id.0,
                                total,
                                baseline,
                                ratio,
                            });
                        }
                        _ => {}
                    }
                }
            }

            // Card timeline: log capture event + stage PNG dump for watched item
            if gpui::card_timeline::is_watched(region_id as usize) {
                let pool_frame = pool.current_frame;
                gpui::card_timeline::log_event(&format!(
                    "[capture] item={} primitive_count={} texture_size={}x{} pool_frame={}",
                    region_id, total, tex_width, tex_height, pool_frame,
                ));
                if gpui::card_timeline::should_dump_capture(total) {
                    // Stage a GPU readback for this texture (flushed after queue.submit)
                    let entry = pool.active.get(&region.id.0);
                    if let Some(entry) = entry {
                        let bytes_per_pixel = 4u32;
                        let unpadded_row_bytes = tex_width * bytes_per_pixel;
                        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
                        let padded_row_bytes = (unpadded_row_bytes + align - 1) / align * align;
                        let buffer_size = padded_row_bytes as u64 * tex_height as u64;
                        let resources = self.resources.as_ref().unwrap();
                        let staging = resources.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("timeline_dump_staging"),
                            size: buffer_size,
                            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                            mapped_at_creation: false,
                        });
                        encoder.copy_texture_to_buffer(
                            wgpu::TexelCopyTextureInfo {
                                texture: &entry.texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::TexelCopyBufferInfo {
                                buffer: &staging,
                                layout: wgpu::TexelCopyBufferLayout {
                                    offset: 0,
                                    bytes_per_row: Some(padded_row_bytes),
                                    rows_per_image: Some(tex_height),
                                },
                            },
                            wgpu::Extent3d {
                                width: tex_width,
                                height: tex_height,
                                depth_or_array_layers: 1,
                            },
                        );
                        let seq = gpui::card_timeline::next_capture_sequence();
                        let format = self.surface_config.format;
                        self.pending_timeline_dumps.push(PendingTimelineDump {
                            staging,
                            width: tex_width,
                            height: tex_height,
                            padded_row_bytes,
                            unpadded_row_bytes,
                            region_id,
                            primitive_count: total,
                            sequence: seq,
                            format,
                        });
                    }
                }
            }

            // Clear guard state on successful fresh capture.
            if has_content {
                pool.guarded_ids.remove(&region.id.0);
            }
            fresh_count += 1;
            if debug {
                debug_items.push(TextureCacheDebugItem {
                    index: region_id,
                    state: "fresh",
                    texture_width: tex_width,
                    texture_height: tex_height,
                    age_frames: 0,
                    last_render_ms: render_ms,
                    reason: Some(reason.into()),
                    quads: prim_quads,
                    mono_sprites: prim_mono,
                    subpixel_sprites: prim_subpixel,
                    paths: prim_paths,
                });
                debug_events.push(TextureCacheDebugLifecycle {
                    event_type: if reused { "reuse" } else { "create" },
                    index: region_id,
                    width: tex_width,
                    height: tex_height,
                    render_ms,
                    age_frames: 0,
                    pool_bucket: Some(SizeClass::from_height(tex_height).name().into()),
                    reason: None,
                });
            }
            render_idx += 1;
        }

        // Report all cached region IDs back to the list for skip-paint decisions
        let pool = self.texture_pool.as_ref().unwrap();
        let cached_ids = pool.active_region_ids();
        gpui::set_cached_region_ids(cached_ids);

        // CS S499: emit aggregated eviction-burst event if any evictions happened
        // this frame. Paired with eviction_count_before snapshot above.
        #[cfg(feature = "texture-cache-debug")]
        {
            let burst = pool.eviction_count.saturating_sub(eviction_count_before);
            if burst > 0 {
                log::info!(
                    "event=texture_eviction_burst count={} pool_active={} paint_frame={}",
                    burst,
                    pool.active.len(),
                    pool.current_frame
                );
            }
        }

        // Emit debug callback with per-frame stats
        if debug {
            let fresh_ms = start
                .map(|s| s.elapsed().as_secs_f32() * 1000.0)
                .unwrap_or(0.0);
            let pool = self.texture_pool.as_ref().unwrap();
            let pool_stats = pool.debug_pool_stats();

            emit_texture_cache_debug(TextureCacheDebugFrame {
                fresh_count,
                cached_count,
                texture_count: pool_stats.total,
                memory_mb: pool_stats.memory_mb,
                composite_ms: pool.prev_composite_ms,
                fresh_ms,
                items: debug_items,
                pool: pool_stats,
                events: debug_events,
            });
        }

        true
    }

    /// Render a complete mini-scene to an offscreen texture.
    /// Runs the full batch pipeline targeting the given texture view.
    fn render_mini_scene_to_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        mini_scene: &Scene,
        target_view: &wgpu::TextureView,
        clear_color: wgpu::Color,
        globals_bind_group: &wgpu::BindGroup,
        instance_offset: &mut u64,
    ) -> bool {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("item_texture_render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear_color),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        for batch in mini_scene.batches() {
            let ok = match batch {
                PrimitiveBatch::Quads(range) => {
                    let items = &mini_scene.quads[range];
                    let data = unsafe { Self::instance_bytes(items) };
                    self.draw_instances_with_custom_globals(
                        data,
                        items.len() as u32,
                        &self.resources().pipelines.quads,
                        globals_bind_group,
                        instance_offset,
                        &mut pass,
                    )
                }
                PrimitiveBatch::Shadows(range) => {
                    let items = &mini_scene.shadows[range];
                    let data = unsafe { Self::instance_bytes(items) };
                    self.draw_instances_with_custom_globals(
                        data,
                        items.len() as u32,
                        &self.resources().pipelines.shadows,
                        globals_bind_group,
                        instance_offset,
                        &mut pass,
                    )
                }
                PrimitiveBatch::Paths(range) => {
                    let paths = &mini_scene.paths[range];
                    if paths.is_empty() {
                        true
                    } else {
                        // Two-pass path rendering: rasterize to intermediate,
                        // then composite onto item texture.
                        drop(pass);

                        let did_rasterize = self.render_paths_to_intermediate_with_globals(
                            encoder,
                            paths,
                            globals_bind_group,
                            instance_offset,
                        );

                        // Restart item texture pass (LoadOp::Load preserves content)
                        pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("item_texture_render_continued"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: target_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: None,
                            ..Default::default()
                        });

                        if did_rasterize {
                            let first_path = &paths[0];
                            // Content mask covers the full viewport (paths intermediate is viewport-sized).
                            // No clipping needed — the path rasterization already clips content.
                            let capture_mask = gpui::Bounds {
                                origin: gpui::Point { x: ScaledPixels(0.0), y: ScaledPixels(0.0) },
                                size: gpui::Size {
                                    width: ScaledPixels(self.surface_config.width as f32),
                                    height: ScaledPixels(self.surface_config.height as f32),
                                },
                            };
                            let sprites: Vec<PathSprite> =
                                if paths.last().map(|p| &p.order) == Some(&first_path.order) {
                                    paths.iter().map(|p| PathSprite { bounds: p.clipped_bounds(), content_mask: capture_mask }).collect()
                                } else {
                                    let mut bounds = first_path.clipped_bounds();
                                    for path in &paths[1..] {
                                        bounds = bounds.union(&path.clipped_bounds());
                                    }
                                    vec![PathSprite { bounds, content_mask: capture_mask }]
                                };
                            let sprite_data = unsafe { Self::instance_bytes(&sprites) };
                            if let Some(intermediate_view) =
                                self.resources().path_intermediate_view.as_ref()
                            {
                                self.draw_instances_with_texture_and_globals(
                                    sprite_data,
                                    sprites.len() as u32,
                                    intermediate_view,
                                    &self.resources().pipelines.paths,
                                    globals_bind_group,
                                    instance_offset,
                                    &mut pass,
                                )
                            } else {
                                true
                            }
                        } else {
                            false
                        }
                    }
                }
                PrimitiveBatch::Underlines(range) => {
                    let items = &mini_scene.underlines[range];
                    let data = unsafe { Self::instance_bytes(items) };
                    self.draw_instances_with_custom_globals(
                        data,
                        items.len() as u32,
                        &self.resources().pipelines.underlines,
                        globals_bind_group,
                        instance_offset,
                        &mut pass,
                    )
                }
                PrimitiveBatch::MonochromeSprites { texture_id, range } => {
                    let items = &mini_scene.monochrome_sprites[range];
                    let tex_info = self.atlas.get_texture_info(texture_id);
                    let data = unsafe { Self::instance_bytes(items) };
                    self.draw_instances_with_texture_and_globals(
                        data,
                        items.len() as u32,
                        &tex_info.view,
                        &self.resources().pipelines.mono_sprites,
                        globals_bind_group,
                        instance_offset,
                        &mut pass,
                    )
                }
                PrimitiveBatch::SubpixelSprites { texture_id, range } => {
                    let items = &mini_scene.subpixel_sprites[range];
                    let tex_info = self.atlas.get_texture_info(texture_id);
                    let data = unsafe { Self::instance_bytes(items) };
                    // Phase B: mono fallback (dual-source blending can't round-trip through RGBA).
                    // Phase A: used subpixel pipeline with mono fallback.
                    let resources = self.resources();
                    let pipeline = if gpui::is_mono_fallback_enabled() {
                        &resources.pipelines.mono_sprites
                    } else {
                        resources.pipelines.subpixel_sprites.as_ref()
                            .unwrap_or(&resources.pipelines.mono_sprites)
                    };
                    self.draw_instances_with_texture_and_globals(
                        data,
                        items.len() as u32,
                        &tex_info.view,
                        pipeline,
                        globals_bind_group,
                        instance_offset,
                        &mut pass,
                    )
                }
                PrimitiveBatch::PolychromeSprites { texture_id, range } => {
                    let items = &mini_scene.polychrome_sprites[range];
                    let tex_info = self.atlas.get_texture_info(texture_id);
                    let data = unsafe { Self::instance_bytes(items) };
                    self.draw_instances_with_texture_and_globals(
                        data,
                        items.len() as u32,
                        &tex_info.view,
                        &self.resources().pipelines.poly_sprites,
                        globals_bind_group,
                        instance_offset,
                        &mut pass,
                    )
                }
                PrimitiveBatch::Surfaces(_) => true,
            };
            if !ok {
                return false;
            }
        }

        true
    }

    /// Rasterize path triangles to the intermediate texture using custom globals.
    /// Used by render_mini_scene_to_texture for path rendering in offscreen items.
    /// The custom globals set viewport_size to item dimensions so path vertices
    /// (in item-local coordinates) map correctly to NDC.
    fn render_paths_to_intermediate_with_globals(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        paths: &[Path<ScaledPixels>],
        globals_bind_group: &wgpu::BindGroup,
        instance_offset: &mut u64,
    ) -> bool {
        let mut vertices = Vec::new();
        for path in paths {
            let bounds = path.clipped_bounds();
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds,
            }));
        }

        if vertices.is_empty() {
            return true;
        }

        let vertex_data = unsafe { Self::instance_bytes(&vertices) };
        let Some((vertex_offset, vertex_size)) =
            self.write_to_instance_buffer(instance_offset, vertex_data)
        else {
            return false;
        };

        let resources = self.resources();
        let data_bind_group =
            resources
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("item_path_rasterization_bind_group"),
                    layout: &resources.bind_group_layouts.instances,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.instance_binding(vertex_offset, vertex_size),
                    }],
                });

        let Some(path_intermediate_view) = resources.path_intermediate_view.as_ref() else {
            return true;
        };

        let (target_view, resolve_target) = if let Some(ref msaa_view) = resources.path_msaa_view {
            (msaa_view, Some(path_intermediate_view))
        } else {
            (path_intermediate_view, None)
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("item_path_rasterization_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&resources.pipelines.path_rasterization);
            pass.set_bind_group(0, globals_bind_group, &[]);
            pass.set_bind_group(1, &data_bind_group, &[]);
            pass.draw(0..vertices.len() as u32, 0..1);
        }

        true
    }

    /// Draw instances using a custom globals bind group (for offscreen rendering).
    fn draw_instances_with_custom_globals(
        &self,
        data: &[u8],
        instance_count: u32,
        pipeline: &wgpu::RenderPipeline,
        globals_bind_group: &wgpu::BindGroup,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        if instance_count == 0 {
            return true;
        }
        let Some((offset, size)) = self.write_to_instance_buffer(instance_offset, data) else {
            return false;
        };
        let resources = self.resources();
        let bind_group = resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &resources.bind_group_layouts.instances,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.instance_binding(offset, size),
                }],
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, globals_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.draw(0..4, 0..instance_count);
        true
    }

    /// Draw textured instances using a custom globals bind group (for offscreen rendering).
    fn draw_instances_with_texture_and_globals(
        &self,
        data: &[u8],
        instance_count: u32,
        texture_view: &wgpu::TextureView,
        pipeline: &wgpu::RenderPipeline,
        globals_bind_group: &wgpu::BindGroup,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        if instance_count == 0 {
            return true;
        }
        let Some((offset, size)) = self.write_to_instance_buffer(instance_offset, data) else {
            return false;
        };
        let resources = self.resources();
        let bind_group = resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &resources.bind_group_layouts.instances_with_texture,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.instance_binding(offset, size),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&resources.atlas_sampler),
                    },
                ],
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, globals_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.draw(0..4, 0..instance_count);
        true
    }

    /// Composite all cached textures as quads into the current main render pass.
    /// Uses the `composite` pipeline which maps UV as unit_vertex [0,1],
    /// giving correct sampling for standalone item textures and enabling
    /// sub-pixel scroll positioning (the quad carries fractional Y from bounds).
    pub(crate) fn draw_cached_regions(
        &mut self,
        scene: &Scene,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        // Time the full compositing pass. Written to pool.prev_composite_ms via
        // record_composite_elapsed() at every exit point — including early returns.
        // This value is read by process_cache_regions on the NEXT frame.
        let started_at = Instant::now();
        let pool = match &self.texture_pool {
            Some(p) => p,
            None => return true,
        };
        let tint_enabled = gpui::is_debug_tint_enabled();
        // Pipeline choice is per-frame, not per-region. Hoist outside the loop.
        let use_composite = gpui::is_composite_pipeline_enabled();
        let pipeline = if use_composite {
            &self.resources().pipelines.composite
        } else {
            &self.resources().pipelines.paths
        };
        if tint_enabled {
            log::info!(
                "event=tint_draw_start regions={} pool_active={} pipeline={}",
                scene.cache_regions().len(), pool.active.len(),
                if use_composite { "composite" } else { "paths" },
            );
        }

        let mut composite_count: u32 = 0;
        let mut skip_count: u32 = 0;

        for region in scene.cache_regions() {
            let entry = match pool.active.get(&region.id.0) {
                Some(e) => e,
                None => {
                    skip_count += 1;
                    continue;
                }
            };

            if tint_enabled {
                log::info!(
                    "event=composite_draw ix={} bounds=({:.0},{:.0},{:.0},{:.0}) tex={}x{} has_content={}",
                    region.id.0,
                    region.bounds.origin.x.0, region.bounds.origin.y.0,
                    region.bounds.size.width.0, region.bounds.size.height.0,
                    entry.width, entry.height,
                    entry.has_content,
                );
            }

            let sprite = PathSprite {
                bounds: region.bounds,
                content_mask: region.viewport_clip,
            };
            let sprite_data = unsafe { Self::instance_bytes(std::slice::from_ref(&sprite)) };
            if !self.draw_instances_with_texture(
                sprite_data,
                1,
                &entry.view,
                pipeline,
                instance_offset,
                pass,
            ) {
                self.record_composite_elapsed(started_at);
                return false;
            }

            composite_count += 1;

            // Debug tint: draw a semi-transparent red overlay on composited textures
            // so users can visually identify which items are cached vs fresh.
            if tint_enabled {
                let tint_quad = gpui::Quad {
                    order: 0,
                    border_style: gpui::BorderStyle::default(),
                    bounds: region.bounds,
                    content_mask: gpui::ContentMask { bounds: region.bounds },
                    background: gpui::solid_background(gpui::hsla(0.0, 1.0, 0.5, 0.15)),
                    border_color: gpui::Hsla::default(),
                    corner_radii: gpui::Corners::default(),
                    border_widths: gpui::Edges::default(),
                };
                if !self.draw_quads(
                    std::slice::from_ref(&tint_quad),
                    instance_offset,
                    pass,
                ) {
                    self.record_composite_elapsed(started_at);
                    return false;
                }
            }
        }

        if tint_enabled {
            log::info!(
                "event=composite_summary composited={} skipped={} total_regions={}",
                composite_count, skip_count, scene.cache_regions().len(),
            );
        }

        self.record_composite_elapsed(started_at);
        true
    }

    /// Record the elapsed composite time from the given start instant.
    /// Called at every exit point of draw_cached_regions. The stored value
    /// is read by process_cache_regions on the NEXT frame's debug callback.
    fn record_composite_elapsed(&mut self, started_at: Instant) {
        if let Some(pool) = &mut self.texture_pool {
            pool.prev_composite_ms = started_at.elapsed().as_secs_f32() * 1000.0;
        }
    }

    /// Invalidate all cached textures and free-list textures.
    /// Called on DPI change, window resize, or theme change.
    pub(crate) fn invalidate_texture_cache(&mut self) {
        if let Some(pool) = &mut self.texture_pool {
            // Gap-NEW-1 Gate-B follow-up: mass-drop emit (sage's missed site).
            // Fires on DPI change, window resize, theme change (RW5 territory).
            // Paired with V1 so every destroy-class event that could invalidate
            // a pending bind group is visible in the crash dump.
            #[cfg(feature = "texture-cache-debug")]
            {
                let active_count = pool.active.len();
                let free_count: usize = pool.free_list.values().map(|v| v.len()).sum();
                log::info!(
                    "event=texture_destroy ix=all reason=pool_purge active_count={} free_count={} bytes={}",
                    active_count, free_count, pool.total_memory_bytes
                );
            }
            pool.active.clear();
            pool.free_list.clear();
            pool.total_memory_bytes = 0;
        }
    }

    /// Dump all active cached textures to PNG files in `/tmp/cs-texture-dump/`.
    /// Reads back GPU textures synchronously via staging buffer + map_async.
    /// Performs one synchronous GPU readback per active texture. Each readback
    /// creates a staging buffer and blocks the main thread until the GPU copy
    /// completes. With 40 cached items this may stall for ~40-100ms total —
    /// acceptable for one-shot diagnostics triggered by hotkey.
    ///
    /// Each file is named `item-{region_id}-{width}x{height}.png`.
    /// If text is missing from the PNG, the capture is broken.
    /// If text IS there, the compositing is broken.
    #[cfg(feature = "texture-cache")]
    pub(crate) fn dump_active_textures_if_requested(&self) {
        if !gpui::take_texture_dump_request() {
            return;
        }

        let pool = match &self.texture_pool {
            Some(p) => p,
            None => {
                log::info!("event=texture_dump status=no_pool hint=no_list_items_rendered_yet");
                return;
            }
        };

        if pool.active.is_empty() {
            log::info!("event=texture_dump status=empty count=0");
            return;
        }

        let dump_dir = StdPath::new("/tmp/cs-texture-dump");
        if let Err(e) = std::fs::create_dir_all(dump_dir) {
            log::error!("event=texture_dump status=mkdir_failed error={}", e);
            return;
        }

        let resources = self.resources();
        let format = self.surface_config.format;
        debug_assert!(
            matches!(
                format,
                wgpu::TextureFormat::Bgra8Unorm
                    | wgpu::TextureFormat::Bgra8UnormSrgb
                    | wgpu::TextureFormat::Rgba8Unorm
                    | wgpu::TextureFormat::Rgba8UnormSrgb
            ),
            "texture dump assumes 4-byte format, got {:?}",
            format
        );
        let bytes_per_pixel = 4u32;

        log::info!(
            "event=texture_dump status=start count={} format={:?}",
            pool.active.len(),
            format
        );

        for (&region_id, entry) in &pool.active {
            let width = entry.width;
            let height = entry.height;
            // wgpu requires rows padded to COPY_BYTES_PER_ROW_ALIGNMENT (256 bytes)
            let unpadded_row_bytes = width * bytes_per_pixel;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let padded_row_bytes = (unpadded_row_bytes + align - 1) / align * align;
            let buffer_size = (padded_row_bytes * height) as u64;

            let staging = resources.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("texture_dump_staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut encoder =
                resources
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("texture_dump_encoder"),
                    });

            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &entry.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row_bytes),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );

            let sub_index = resources
                .queue
                .submit(std::iter::once(encoder.finish()));

            // Synchronous readback — blocks until this specific copy completes
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            if let Err(e) = resources.device.poll(wgpu::PollType::Wait {
                submission_index: Some(sub_index),
                timeout: Some(std::time::Duration::from_secs(5)),
            }) {
                log::error!(
                    "event=texture_dump status=gpu_timeout region={} error={:?}",
                    region_id, e
                );
                continue;
            }

            match rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    log::error!(
                        "event=texture_dump status=map_failed region={} error={}",
                        region_id, e
                    );
                    continue;
                }
                Err(e) => {
                    log::error!(
                        "event=texture_dump status=recv_failed region={} error={}",
                        region_id, e
                    );
                    continue;
                }
            }

            // SAFETY: Both error arms above `continue`; reaching here guarantees map succeeded.
            let mapped = slice.get_mapped_range();

            // Convert BGRA→RGBA if needed, strip row padding
            let is_bgra = matches!(
                format,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
            );
            let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
            for row in 0..height {
                let row_start = (row * padded_row_bytes) as usize;
                let row_end = row_start + (unpadded_row_bytes) as usize;
                let row_data = &mapped[row_start..row_end];
                if is_bgra {
                    // Swap B and R channels: BGRA → RGBA
                    for pixel in row_data.chunks_exact(4) {
                        rgba_data.push(pixel[2]); // R (was B)
                        rgba_data.push(pixel[1]); // G
                        rgba_data.push(pixel[0]); // B (was R)
                        rgba_data.push(pixel[3]); // A
                    }
                } else {
                    // Assumes RGBA byte order (Rgba8Unorm, Rgba8UnormSrgb).
                    // If a new surface format produces corrupt colours in PNGs, add a conversion here.
                    rgba_data.extend_from_slice(row_data);
                }
            }

            drop(mapped);
            staging.unmap();

            // Write PNG
            let filename = format!("item-{}-{}x{}.png", region_id, width, height);
            let path = dump_dir.join(&filename);

            match write_png(&path, width, height, &rgba_data) {
                Ok(()) => {
                    log::info!(
                        "event=texture_dump status=ok region={} file={} size={}x{}",
                        region_id,
                        path.display(),
                        width,
                        height
                    );
                }
                Err(e) => {
                    log::error!(
                        "event=texture_dump status=write_failed region={} error={}",
                        region_id, e
                    );
                }
            }
        }

        log::info!(
            "event=texture_dump status=complete count={} dir={}",
            pool.active.len(),
            dump_dir.display()
        );
    }

    /// Flush pending card-timeline PNG dumps (called after queue.submit).
    /// Reads back staged GPU buffers and writes PNGs to /tmp/.
    #[cfg(feature = "texture-cache")]
    pub(crate) fn flush_timeline_dumps(&mut self) {
        if self.pending_timeline_dumps.is_empty() {
            return;
        }
        let resources = self.resources.as_ref().unwrap();
        let dumps: Vec<PendingTimelineDump> = self.pending_timeline_dumps.drain(..).collect();
        for dump in &dumps {
            let slice = dump.staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            if let Err(e) = resources.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            }) {
                gpui::card_timeline::log_event(&format!(
                    "[capture_dump] item={} error=gpu_timeout {:?}",
                    dump.region_id, e,
                ));
                continue;
            }
            match rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    gpui::card_timeline::log_event(&format!(
                        "[capture_dump] item={} error=map_failed {}",
                        dump.region_id, e,
                    ));
                    continue;
                }
                Err(e) => {
                    gpui::card_timeline::log_event(&format!(
                        "[capture_dump] item={} error=recv_failed {}",
                        dump.region_id, e,
                    ));
                    continue;
                }
            }
            let mapped = slice.get_mapped_range();
            let is_bgra = matches!(
                dump.format,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
            );
            let mut rgba_data = Vec::with_capacity((dump.width * dump.height * 4) as usize);
            for row in 0..dump.height {
                let row_start = row as usize * dump.padded_row_bytes as usize;
                let row_end = row_start + dump.unpadded_row_bytes as usize;
                let row_data = &mapped[row_start..row_end];
                if is_bgra {
                    for pixel in row_data.chunks_exact(4) {
                        rgba_data.push(pixel[2]);
                        rgba_data.push(pixel[1]);
                        rgba_data.push(pixel[0]);
                        rgba_data.push(pixel[3]);
                    }
                } else {
                    rgba_data.extend_from_slice(row_data);
                }
            }
            drop(mapped);
            dump.staging.unmap();

            let path = format!(
                "/tmp/cs-texture-capture-{}-{}.png",
                dump.region_id, dump.sequence,
            );
            match write_png(StdPath::new(&path), dump.width, dump.height, &rgba_data) {
                Ok(()) => {
                    gpui::card_timeline::log_event(&format!(
                        "[capture_dump] item={} seq={} file={} size={}x{} primitives={}",
                        dump.region_id, dump.sequence, path, dump.width, dump.height, dump.primitive_count,
                    ));
                }
                Err(e) => {
                    gpui::card_timeline::log_event(&format!(
                        "[capture_dump] item={} error=write_failed {}",
                        dump.region_id, e,
                    ));
                }
            }
        }
    }
}

/// Write RGBA pixel data as a PNG file.
#[cfg(feature = "texture-cache")]
fn write_png(
    path: &StdPath,
    width: u32,
    height: u32,
    rgba_data: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::create(path)?;
    let ref mut w = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(rgba_data)?;
    Ok(())
}
