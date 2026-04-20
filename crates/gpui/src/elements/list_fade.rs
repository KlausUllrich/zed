//! Fade-composite helper for the texture→layout transition crossfade.
//!
//! S502 — minimal encapsulation to keep `list.rs`'s paint-loop hunks small
//! and reduce per-upstream-bump merge cost. Upstream Zed has no equivalent
//! file, so this sibling has zero rebase conflict surface on its own.
//!
//! Called from `List::paint` ONLY when:
//!   (a) `ListState::caching_enabled == false` (DIRECT path), AND
//!   (b) `ListState::fade_alpha < 1.0`          (fade is in-progress), AND
//!   (c) `has_cached_region(region_id) == true` (the prior cached texture
//!       from the just-ended scroll is still valid).
//!
//! Under these conditions the controller on the CS side has armed the fade
//! and is driving `ListState::fade_alpha` toward 0.0 over ~100ms. The CS
//! owner is `cs_ui::conversation::fade_compose::FadeComposeController` —
//! `set_fade_alpha` on `ListState` is the only fork-side surface. This
//! helper emits the two paint primitives needed to crossfade the cached
//! texture over the fresh DIRECT paint:
//!
//!   1. `element.paint(..)`               — DIRECT primitives at normal z
//!   2. `begin_cache_region / end_cache_region` — empty bracket whose
//!      recorded Scene position is AFTER the DIRECT primitives → the cached
//!      texture composites ON TOP. The GPU fragment shader multiplies the
//!      sample by `globals.composite_fade_alpha` (written from
//!      `Scene::composite_fade_alpha`, set by List::paint from
//!      ListState::fade_alpha). As alpha decays 1.0 → 0.0, the cached
//!      texture fades out and the DIRECT paint shows through.

#![cfg(feature = "texture-cache")]

use crate::{AnyElement, App, Bounds, CacheRegionId, Hsla, Pixels, Window};

/// Emit the fade-overlay paint sequence for one list item.
///
/// See the module doc for invocation preconditions. Callers MUST have
/// already confirmed `has_cached_region(region_id) == true`.
///
/// Emits an `event=fade_overlay_emit` Info log entry so cs-debug (via its
/// log-forwarding layer) can confirm the mechanism is firing during a
/// transition. `alpha` is passed by the caller from the snapshot taken at
/// paint entry so the log matches what the shader will actually use.
pub(crate) fn emit_fade_overlay(
    window: &mut Window,
    cx: &mut App,
    element: &mut AnyElement,
    region_id: CacheRegionId,
    item_bounds: Bounds<Pixels>,
    viewport_bounds: Bounds<Pixels>,
    cache_clear_color: Hsla,
    alpha: f32,
) {
    log::info!(
        "event=fade_overlay_emit region={} alpha={:.2}",
        region_id.0,
        alpha
    );
    // DIRECT paint: primitives at the card's normal z-order.
    element.paint(window, cx);
    // Empty cache-region bracket: the renderer composites the already-cached
    // texture into this region at `Scene::composite_fade_alpha` (set from
    // ListState::fade_alpha by `List::paint`). Emitted after `element.paint`
    // so the region's recorded Scene order is strictly greater than the
    // DIRECT primitives → composite renders on top, not below.
    window.begin_cache_region(region_id, item_bounds, cache_clear_color, viewport_bounds);
    window.end_cache_region(region_id);
}
