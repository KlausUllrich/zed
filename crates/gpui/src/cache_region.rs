//! GPU texture cache region types and scene extraction.
//!
//! Two responsibilities:
//! 1. Cache region annotations — `CacheRegion`, `CacheRegionId`, and scene helpers
//!    used by the renderer to capture list items to GPU textures.
//! 2. Renderer feedback state — `CACHED_REGION_IDS` thread-local tracks which
//!    regions have valid textures. `has_cached_region` / `clear_cached_region` /
//!    `clear_cached_region_ids` let the list query and invalidate this state.

use crate::{
    Bounds, ContentMask, Hsla, Point, Primitive, ScaledPixels, Scene,
    point,
    scene::PaintOperation,
};
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
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
    CACHED_REGION_IDS.with(|cell| cell.borrow_mut().clear());
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
