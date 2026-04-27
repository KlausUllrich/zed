//! A list element that can be used to render a large number of differently sized elements
//! efficiently. Clients of this API need to ensure that elements outside of the scrolled
//! area do not change their height for this element to function correctly. If your elements
//! do change height, notify the list element via [`ListState::splice`] or [`ListState::reset`].
//! In order to minimize re-renders, this element's state is stored intrusively
//! on your own views, so that your code can coordinate directly with the list element's cached state.
//!
//! If all of your elements are the same height, see [`crate::UniformList`] for a simpler API

use crate::{
    AnyElement, App, AvailableSpace, Bounds, ContentMask, DispatchPhase, Edges, Element, EntityId,
    FocusHandle, GlobalElementId, Hitbox, HitboxBehavior, InspectorElementId, IntoElement,
    Overflow, Pixels, Point, ScrollDelta, ScrollWheelEvent, Size, Style, StyleRefinement, Styled,
    Window, point, px, size,
};
#[cfg(feature = "texture-cache")]
use crate::{CacheRegionId, Hsla, has_cached_region};
#[cfg(feature = "texture-cache")]
use std::any::TypeId;
#[cfg(feature = "texture-cache")]
use std::collections::{HashMap, HashSet};
use collections::VecDeque;
use refineable::Refineable as _;
use std::{cell::Cell, cell::RefCell, ops::Range, rc::Rc, time::Instant};
use sum_tree::{Bias, Dimensions, SumTree};

type RenderItemFn = dyn FnMut(usize, &mut Window, &mut App) -> AnyElement + 'static;

/// Telemetry data emitted each frame during Tail mode inertia scroll.
/// Consumed by the app's debug viewer (e.g., cs-debug Scroll tab).
#[derive(Clone, Debug)]
pub struct ScrollTelemetry {
    /// Current velocity in px/s.
    pub velocity: f32,
    /// Frame delta time in seconds.
    pub dt: f32,
    /// Distance from current position to scroll_max in px.
    pub delta: f32,
    /// Content growth this frame in px.
    pub growth: f32,
    /// Total item count.
    pub item_count: usize,
    /// Which branch fired: true = inertia, false = snap.
    pub inertia_active: bool,
}

/// Callback type for scroll telemetry. Registered by the app at startup.
type ScrollTelemetryCallback = Box<dyn Fn(&ScrollTelemetry) + 'static>;

/// Performance data emitted after layout_items() or prepaint_items().
/// Helps diagnose which items are expensive to render.
#[derive(Clone, Debug)]
pub struct LayoutPerfTelemetry {
    /// Which phase: "layout" (render_item + layout_as_root) or "prepaint" (prepaint_at).
    pub phase: &'static str,
    /// Total wall time in seconds.
    pub total_secs: f32,
    /// Number of items processed (rendered or prepainted).
    pub rendered_count: usize,
    /// Number of items skipped (used cached size). Only meaningful for layout phase.
    pub cached_count: usize,
    /// Total item count in the list.
    pub item_count: usize,
    /// Scroll top item index.
    pub scroll_top_ix: usize,
    /// Slowest single item time in seconds (0 if none processed).
    pub slowest_item_secs: f32,
    /// Index of the slowest item.
    pub slowest_item_ix: usize,
}

type LayoutPerfCallback = Box<dyn Fn(&LayoutPerfTelemetry) + 'static>;

/// Telemetry emitted when scroll offset compensation fires.
/// Tracks hidden scroll adjustments caused by item re-measurement.
#[derive(Clone, Debug)]
pub struct OffsetCompensationEvent {
    /// Index of the item that triggered compensation.
    pub item_ix: usize,
    /// Old height from size_hint (estimated or previously measured).
    pub old_height: f32,
    /// New measured height.
    pub new_height: f32,
    /// Height delta applied to scroll offset (positive = item grew, negative = shrank).
    /// Equal to `new_height - old_height`.
    pub height_delta: f32,
    /// offset_in_item before adjustment.
    pub offset_before: f32,
    /// offset_in_item after adjustment.
    pub offset_after: f32,
}

type OffsetCompensationCallback = Box<dyn Fn(&OffsetCompensationEvent) + 'static>;

/// Telemetry emitted when a visible item's measured height changes between frames
/// during active scroll with texture caching enabled. Used to diagnose height
/// oscillation that causes perpetual cache MISSes and blank cards.
#[cfg(feature = "texture-cache")]
#[derive(Clone, Debug)]
pub struct HeightChangeEvent {
    /// Absolute list index of the item whose height changed.
    pub item_index: usize,
    /// Height from the previous frame (px). Zero if first observation.
    pub old_height: f32,
    /// Height measured this frame (px).
    pub new_height: f32,
    /// Cache path for this item this frame: "HIT", "MISS", "STREAMING", or "TRANSIENT".
    pub cache_path: &'static str,
    /// Monotonic frame counter (incremented each paint pass).
    pub frame_count: u64,
}

#[cfg(feature = "texture-cache")]
type HeightChangeCallback = Box<dyn Fn(&HeightChangeEvent) + 'static>;

thread_local! {
    static SCROLL_TELEMETRY_CB: Cell<Option<*const ScrollTelemetryCallback>> = const { Cell::new(None) };
    static LAYOUT_PERF_CB: Cell<Option<*const LayoutPerfCallback>> = const { Cell::new(None) };
    static OFFSET_COMP_CB: Cell<Option<*const OffsetCompensationCallback>> = const { Cell::new(None) };
    #[cfg(feature = "texture-cache")]
    static HEIGHT_CHANGE_CB: Cell<Option<*const HeightChangeCallback>> = const { Cell::new(None) };
}

/// Register a callback to receive scroll telemetry during Tail mode inertia.
/// Call once at app startup. The callback should forward to the debug logging system.
///
/// # Safety
/// The callback must outlive all List elements. In practice, register a `Box::leak`'d
/// callback at startup and never unregister.
pub fn set_scroll_telemetry_callback(callback: ScrollTelemetryCallback) {
    let leaked = Box::leak(Box::new(callback));
    SCROLL_TELEMETRY_CB.with(|cell| cell.set(Some(leaked as *const ScrollTelemetryCallback)));
}

/// Register a callback to receive layout performance telemetry.
/// Fires after every `layout_items()` call. Only reports when total time > 8ms
/// to avoid noise during fast frames.
pub fn set_layout_perf_callback(callback: LayoutPerfCallback) {
    let leaked = Box::leak(Box::new(callback));
    LAYOUT_PERF_CB.with(|cell| cell.set(Some(leaked as *const LayoutPerfCallback)));
}

fn emit_scroll_telemetry(telemetry: &ScrollTelemetry) {
    SCROLL_TELEMETRY_CB.with(|cell| {
        if let Some(ptr) = cell.get() {
            let cb = unsafe { &*ptr };
            cb(telemetry);
        }
    });
}

fn emit_layout_perf(telemetry: &LayoutPerfTelemetry) {
    LAYOUT_PERF_CB.with(|cell| {
        if let Some(ptr) = cell.get() {
            let cb = unsafe { &*ptr };
            cb(telemetry);
        }
    });
}

/// Register a callback to receive offset compensation events.
/// Fires when item re-measurement adjusts the scroll offset.
pub fn set_offset_compensation_callback(callback: OffsetCompensationCallback) {
    let leaked = Box::leak(Box::new(callback));
    OFFSET_COMP_CB.with(|cell| cell.set(Some(leaked as *const OffsetCompensationCallback)));
}

fn emit_offset_compensation(event: &OffsetCompensationEvent) {
    OFFSET_COMP_CB.with(|cell| {
        if let Some(ptr) = cell.get() {
            let cb = unsafe { &*ptr };
            cb(event);
        }
    });
}

/// Register a callback to receive height change events during cached scroll.
/// Fires when a visible item's measured height differs from its previous frame height
/// while texture caching is active. Used to diagnose height oscillation.
#[cfg(feature = "texture-cache")]
pub fn set_height_change_callback(callback: HeightChangeCallback) {
    let leaked = Box::leak(Box::new(callback));
    HEIGHT_CHANGE_CB.with(|cell| cell.set(Some(leaked as *const HeightChangeCallback)));
}

#[cfg(feature = "texture-cache")]
fn emit_height_change(event: &HeightChangeEvent) {
    HEIGHT_CHANGE_CB.with(|cell| {
        if let Some(ptr) = cell.get() {
            let cb = unsafe { &*ptr };
            cb(event);
        }
    });
}

/// Minimum velocity threshold for Tail mode inertia (px/s).
/// Below this, movement is imperceptible and we snap to scroll_max.
/// Referenced by both `layout_items()` and `is_smooth_scrolling()`.
const MIN_VELOCITY_PX_PER_SEC: f32 = 12.0;

/// EC-6: Minimum frames an item must be visible before texture creation.
/// Items visible for fewer frames during rapid scroll skip texture capture
/// to avoid wasting GPU budget on imperceptible items.
#[cfg(feature = "texture-cache")]
const TRANSIENT_SKIP_FRAMES: u8 = 2;

/// Construct a new list element
pub fn list(
    state: ListState,
    render_item: impl FnMut(usize, &mut Window, &mut App) -> AnyElement + 'static,
) -> List {
    List {
        state,
        render_item: Box::new(render_item),
        style: StyleRefinement::default(),
        sizing_behavior: ListSizingBehavior::default(),
    }
}

/// A list element
pub struct List {
    state: ListState,
    render_item: Box<RenderItemFn>,
    style: StyleRefinement,
    sizing_behavior: ListSizingBehavior,
}

impl List {
    /// Set the sizing behavior for the list.
    pub fn with_sizing_behavior(mut self, behavior: ListSizingBehavior) -> Self {
        self.sizing_behavior = behavior;
        self
    }
}

/// The list state that views must hold on behalf of the list element.
#[derive(Clone)]
pub struct ListState(Rc<RefCell<StateInner>>);

impl std::fmt::Debug for ListState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ListState")
    }
}

struct StateInner {
    last_layout_bounds: Option<Bounds<Pixels>>,
    last_padding: Option<Edges<Pixels>>,
    items: SumTree<ListItem>,
    logical_scroll_top: Option<ListOffset>,
    alignment: ListAlignment,
    overdraw: Pixels,
    reset: bool,
    #[allow(clippy::type_complexity)]
    scroll_handler: Option<Box<dyn FnMut(&ListScrollEvent, &mut Window, &mut App)>>,
    scrollbar_drag_start_height: Option<Pixels>,
    /// Smoothed content height for scrollbar — prevents thumb jumps during auto-scroll.
    /// Lerps toward live height each frame (factor 0.3). None until first scrollbar query.
    smoothed_scrollbar_height: Option<Pixels>,
    measuring_behavior: ListMeasuringBehavior,
    pending_scroll: Option<PendingScrollFraction>,
    /// When true, the built-in scroll wheel handler is suppressed.
    /// Used by ConversationView to handle wheel events with smooth pixel animation.
    suppress_wheel_scroll: bool,
    /// Follow mode state — controls auto-scroll to end behavior.
    follow_state: FollowState,
    /// Velocity for inertia-based smooth scroll in Tail mode (px/second).
    /// Content growth adds impulses; friction decays velocity each frame.
    /// Delta-time-based: animation speed is constant regardless of frame rate.
    tail_scroll_velocity: f32,
    /// Previous frame's item count when in Tail mode. Used to detect new card insertion.
    prev_tail_item_count: usize,
    /// Previous frame's scroll_max in Tail mode. Used to compute per-frame growth.
    prev_tail_scroll_max: Pixels,
    /// Timestamp of the last inertia physics update. Used for delta-time calculation.
    last_inertia_time: Option<Instant>,
    // === Texture Cache Edge Cases (Phase B Stream 4) ===
    // EC-6:  Transient skip — items visible <2 frames skip texture creation (rapid scroll)
    // EC-7:  Stale dimension removal — size-changed textures evicted before hit check
    // EC-10: Drag parity — texture caching enabled during scrollbar drag (matches wheel)
    // EC-11: GPU recovery — cached_region_ids cleared on device lost
    // EC-12: Overdraw caching — trailing overdraw items included in layout when caching active
    // EC-13: Jump reset — visible_frames cleared on programmatic scroll jumps
    /// When true, items are annotated for GPU texture caching during paint.
    #[cfg(feature = "texture-cache")]
    caching_enabled: bool,
    /// Background color for offscreen texture clear (subpixel text needs opaque bg).
    #[cfg(feature = "texture-cache")]
    cache_clear_color: Hsla,
    /// EC-6: Per-item frame visibility counter. Items visible for fewer than
    /// TRANSIENT_SKIP_FRAMES frames skip texture creation during rapid scroll.
    /// Key: item index, Value: consecutive frames visible.
    #[cfg(feature = "texture-cache")]
    visible_frames: HashMap<usize, u8>,
    /// FR-4: Item indices that are actively streaming content and must NOT be cached.
    /// These items render Fresh every frame. Set by the host app via
    /// `set_streaming_items()`. Cleared when streaming stops.
    #[cfg(feature = "texture-cache")]
    streaming_items: HashSet<usize>,
    /// Debug: item indices that should emit detailed trace logs during the cache
    /// pipeline (paint → capture → composite). Set by the host app via
    /// `set_trace_items()` to diagnose card-specific rendering issues.
    #[cfg(feature = "texture-cache")]
    trace_items: HashSet<usize>,
    /// Per-item height from the previous paint frame. Compared against current
    /// frame to detect height oscillation during cached scroll.
    #[cfg(feature = "texture-cache")]
    prev_item_heights: HashMap<usize, Pixels>,
    /// Monotonic paint frame counter for height change event timestamps.
    #[cfg(feature = "texture-cache")]
    paint_frame_count: u64,
    /// When true, scroll_to_max() smooths large jumps (>50px) over multiple
    /// frames instead of snapping. Set by the host app during follow_output mode.
    #[cfg(feature = "texture-cache")]
    smooth_scroll_active: bool,
    /// S500: Element state keys captured at MISS/TRANSIENT paint per cached
    /// item. During HIT frames the item's `element.paint` is skipped entirely,
    /// so its `with_element_state` calls never re-register their keys in
    /// `next_frame.accessed_element_states`. Without re-registration,
    /// `Frame::finish` drops the states on the next tick — the first DIRECT
    /// paint after transition then re-creates `TextViewState` with empty
    /// `parsed_content`, producing a 1-frame shell-only render while the async
    /// parse task repopulates content. Re-extending these keys each HIT frame
    /// (via `Window::keep_element_states_alive`) keeps the states alive across
    /// arbitrary cached-scroll durations.
    #[cfg(feature = "texture-cache")]
    cached_item_state_keys: HashMap<usize, Vec<(GlobalElementId, TypeId)>>,
    /// S502: Fade alpha applied to cached-texture composites during the
    /// texture→layout transition. Default 1.0 = no fade (composite as opaque).
    /// Driven from CS via `set_fade_alpha()` on each frame the controller's
    /// `tick()` returns < 1.0. Written into `Scene::composite_fade_alpha` at
    /// paint entry so the GPU composite shader applies the multiply. Reset to
    /// 1.0 in `set_item_caching_enabled(true, ...)` so scroll resume snaps the
    /// cached path back to full opacity immediately (RS-FADE-B).
    #[cfg(feature = "texture-cache")]
    fade_alpha: f32,
}

/// Keeps track of a fractional scroll position within an item for restoration
/// after remeasurement.
struct PendingScrollFraction {
    /// The index of the item to scroll within.
    item_ix: usize,
    /// Fractional offset (0.0 to 1.0) within the item's height.
    fraction: f32,
}

/// Whether the list is scrolling from top to bottom or bottom to top.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ListAlignment {
    /// The list is scrolling from top to bottom, like most lists.
    Top,
    /// The list is scrolling from bottom to top, like a chat log.
    Bottom,
}

/// Controls the auto-scroll behavior of the list.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FollowMode {
    /// No auto-scrolling — the user controls the scroll position.
    Normal,
    /// Automatically scroll to the end of the list when new content is added
    /// or items are remeasured. If the user scrolls away, following is suspended
    /// but re-engages automatically when they scroll back to the bottom.
    Tail,
}

/// Internal state tracking for follow mode.
#[derive(Clone, Copy, Debug, Default)]
enum FollowState {
    /// Not following — user controls scroll position.
    #[default]
    Normal,
    /// Following the tail of the list.
    /// `is_following` is false when the user has scrolled away (suspended).
    Tail { is_following: bool },
}

/// A scroll event that has been converted to be in terms of the list's items.
pub struct ListScrollEvent {
    /// The range of items currently visible in the list, after applying the scroll event.
    pub visible_range: Range<usize>,

    /// The number of items that are currently visible in the list, after applying the scroll event.
    pub count: usize,

    /// Whether the list has been scrolled.
    pub is_scrolled: bool,

    /// Whether the list is currently auto-following the tail (end) of the list.
    pub is_following_tail: bool,
}

/// The sizing behavior to apply during layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ListSizingBehavior {
    /// The list should calculate its size based on the size of its items.
    Infer,
    /// The list should not calculate a fixed size.
    #[default]
    Auto,
}

/// The measuring behavior to apply during layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ListMeasuringBehavior {
    /// Measure all items in the list.
    /// Note: This can be expensive for the first frame in a large list.
    Measure(bool),
    /// Only measure visible items
    #[default]
    Visible,
}

impl ListMeasuringBehavior {
    fn reset(&mut self) {
        match self {
            ListMeasuringBehavior::Measure(has_measured) => *has_measured = false,
            ListMeasuringBehavior::Visible => {}
        }
    }
}

/// The horizontal sizing behavior to apply during layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ListHorizontalSizingBehavior {
    /// List items' width can never exceed the width of the list.
    #[default]
    FitList,
    /// List items' width may go over the width of the list, if any item is wider.
    Unconstrained,
}

struct LayoutItemsResponse {
    max_item_width: Pixels,
    scroll_top: ListOffset,
    item_layouts: VecDeque<ItemLayout>,
}

struct ItemLayout {
    index: usize,
    element: AnyElement,
    size: Size<Pixels>,
    #[cfg(feature = "texture-cache")]
    origin: Point<Pixels>,
    /// EC-12: True for items in the overdraw zone (outside viewport).
    /// These are included in layout only when texture caching is active,
    /// so their textures are ready when scrolled into view.
    #[cfg(feature = "texture-cache")]
    is_overdraw: bool,
}

/// Frame state used by the [List] element after layout.
pub struct ListPrepaintState {
    hitbox: Hitbox,
    layout: LayoutItemsResponse,
}

#[derive(Clone)]
enum ListItem {
    Unmeasured {
        /// Hint from the last measured size, used to keep SumTree height stable
        /// while the item is unmeasured. `None` for truly new items.
        size_hint: Option<Size<Pixels>>,
        focus_handle: Option<FocusHandle>,
    },
    Measured {
        size: Size<Pixels>,
        focus_handle: Option<FocusHandle>,
    },
}

impl ListItem {
    fn size(&self) -> Option<Size<Pixels>> {
        if let ListItem::Measured { size, .. } = self {
            Some(*size)
        } else {
            None
        }
    }

    /// Returns the best known size: measured size for Measured items,
    /// or the preserved size hint for Unmeasured items.
    fn size_hint(&self) -> Option<Size<Pixels>> {
        match self {
            ListItem::Measured { size, .. } => Some(*size),
            ListItem::Unmeasured { size_hint, .. } => *size_hint,
        }
    }

    fn focus_handle(&self) -> Option<FocusHandle> {
        match self {
            ListItem::Unmeasured { focus_handle, .. } | ListItem::Measured { focus_handle, .. } => {
                focus_handle.clone()
            }
        }
    }

    fn contains_focused(&self, window: &Window, cx: &App) -> bool {
        match self {
            ListItem::Unmeasured { focus_handle, .. } | ListItem::Measured { focus_handle, .. } => {
                focus_handle
                    .as_ref()
                    .is_some_and(|handle| handle.contains_focused(window, cx))
            }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct ListItemSummary {
    count: usize,
    rendered_count: usize,
    unrendered_count: usize,
    height: Pixels,
    has_focus_handles: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Count(usize);

#[derive(Clone, Debug, Default)]
struct Height(Pixels);

impl ListState {
    /// Construct a new list state, for storage on a view.
    ///
    /// The overdraw parameter controls how much extra space is rendered
    /// above and below the visible area. Elements within this area will
    /// be measured even though they are not visible. This can help ensure
    /// that the list doesn't flicker or pop in when scrolling.
    pub fn new(item_count: usize, alignment: ListAlignment, overdraw: Pixels) -> Self {
        let this = Self(Rc::new(RefCell::new(StateInner {
            last_layout_bounds: None,
            last_padding: None,
            items: SumTree::default(),
            logical_scroll_top: None,
            alignment,
            overdraw,
            scroll_handler: None,
            reset: false,
            scrollbar_drag_start_height: None,
            smoothed_scrollbar_height: None,
            measuring_behavior: ListMeasuringBehavior::default(),
            pending_scroll: None,
            suppress_wheel_scroll: false,
            follow_state: FollowState::Normal,
            tail_scroll_velocity: 0.0,
            prev_tail_item_count: 0,
            prev_tail_scroll_max: px(0.),
            last_inertia_time: None,
            #[cfg(feature = "texture-cache")]
            caching_enabled: false,
            #[cfg(feature = "texture-cache")]
            cache_clear_color: Hsla::default(),
            #[cfg(feature = "texture-cache")]
            visible_frames: HashMap::new(),
            #[cfg(feature = "texture-cache")]
            streaming_items: HashSet::new(),
            #[cfg(feature = "texture-cache")]
            trace_items: HashSet::new(),
            #[cfg(feature = "texture-cache")]
            prev_item_heights: HashMap::new(),
            #[cfg(feature = "texture-cache")]
            paint_frame_count: 0,
            #[cfg(feature = "texture-cache")]
            smooth_scroll_active: false,
            #[cfg(feature = "texture-cache")]
            cached_item_state_keys: HashMap::new(),
            #[cfg(feature = "texture-cache")]
            fade_alpha: 1.0,
        })));
        this.splice(0..0, item_count);
        this
    }

    /// Set the list to measure all items in the list in the first layout phase.
    ///
    /// This is useful for ensuring that the scrollbar size is correct instead of based on only rendered elements.
    pub fn measure_all(self) -> Self {
        self.0.borrow_mut().measuring_behavior = ListMeasuringBehavior::Measure(false);
        self
    }

    /// Reset this instantiation of the list state.
    ///
    /// Note that this will cause scroll events to be dropped until the next paint.
    pub fn reset(&self, element_count: usize) {
        let old_count = {
            let state = &mut *self.0.borrow_mut();
            state.reset = true;
            state.measuring_behavior.reset();
            state.logical_scroll_top = None;
            state.scrollbar_drag_start_height = None;
            state.smoothed_scrollbar_height = None;
            state.items.summary().count
        };

        self.splice(0..old_count, element_count);
    }

    /// Remeasure all items while preserving proportional scroll position.
    ///
    /// Use this when item heights may have changed (e.g., font size changes)
    /// but the number and identity of items remains the same.
    pub fn remeasure(&self) {
        let state = &mut *self.0.borrow_mut();

        let new_items = state.items.iter().map(|item| ListItem::Unmeasured {
            focus_handle: item.focus_handle(),
            // Preserve last known size as hint — keeps SumTree height stable
            // during remeasure cycle.
            size_hint: item.size_hint(),
        });

        // If there's a `logical_scroll_top`, we need to keep track of it as a
        // `PendingScrollFraction`, so we can later preserve that scroll
        // position proportionally to the item, in case the item's height
        // changes.
        if let Some(scroll_top) = state.logical_scroll_top {
            let mut cursor = state.items.cursor::<Count>(());
            cursor.seek(&Count(scroll_top.item_ix), Bias::Right);

            if let Some(item) = cursor.item() {
                if let Some(size) = item.size() {
                    let fraction = if size.height.0 > 0.0 {
                        (scroll_top.offset_in_item.0 / size.height.0).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };

                    state.pending_scroll = Some(PendingScrollFraction {
                        item_ix: scroll_top.item_ix,
                        fraction,
                    });
                }
            }
        }

        state.items = SumTree::from_iter(new_items, ());
        state.measuring_behavior.reset();
    }

    /// Mark a range of items for remeasure without changing the item count.
    /// Unlike `splice()`, this preserves the last measured size as a `size_hint`,
    /// keeping the SumTree height stable during the remeasure cycle.
    /// Use for content updates (e.g., streaming tokens) where items change height
    /// but the number of items stays the same.
    pub fn remeasure_items(&self, range: Range<usize>) {
        let state = &mut *self.0.borrow_mut();

        // Save scroll position fraction if scroll_top item is in the range
        if let Some(scroll_top) = state.logical_scroll_top {
            if range.contains(&scroll_top.item_ix) {
                let mut scroll_cursor = state.items.cursor::<Count>(());
                scroll_cursor.seek(&Count(scroll_top.item_ix), Bias::Right);
                if let Some(item) = scroll_cursor.item() {
                    if let Some(size) = item.size() {
                        let fraction = if size.height.0 > 0.0 {
                            (scroll_top.offset_in_item.0 / size.height.0).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        state.pending_scroll = Some(PendingScrollFraction {
                            item_ix: scroll_top.item_ix,
                            fraction,
                        });
                    }
                }
            }
        }

        let mut cursor = state.items.cursor::<Count>(());
        let mut new_items = cursor.slice(&Count(range.start), Bias::Right);

        // Mark items in range as unmeasured, preserving their last known size
        while cursor.start() < &Count(range.end) {
            if let Some(item) = cursor.item() {
                new_items.push(
                    ListItem::Unmeasured {
                        size_hint: item.size_hint(),
                        focus_handle: item.focus_handle(),
                    },
                    (),
                );
                cursor.next();
            } else {
                break;
            }
        }

        new_items.append(cursor.suffix(), ());
        drop(cursor);
        state.items = new_items;
    }

    /// The number of items in this list.
    pub fn item_count(&self) -> usize {
        self.0.borrow().items.summary().count
    }

    /// Inform the list state that the items in `old_range` have been replaced
    /// by `count` new items that must be recalculated.
    pub fn splice(&self, old_range: Range<usize>, count: usize) {
        self.splice_focusable(old_range, (0..count).map(|_| None))
    }

    /// Register with the list state that the items in `old_range` have been replaced
    /// by new items. As opposed to [`Self::splice`], this method allows an iterator of optional focus handles
    /// to be supplied to properly integrate with items in the list that can be focused. If a focused item
    /// is scrolled out of view, the list will continue to render it to allow keyboard interaction.
    pub fn splice_focusable(
        &self,
        old_range: Range<usize>,
        focus_handles: impl IntoIterator<Item = Option<FocusHandle>>,
    ) {
        let state = &mut *self.0.borrow_mut();
        // Splice shifts item indices — all CacheRegionIds after splice point become stale.
        // Invalidate entire cache to prevent compositing wrong textures.
        #[cfg(feature = "texture-cache")]
        if state.caching_enabled {
            state.caching_enabled = false;
            state.visible_frames.clear();
            state.cached_item_state_keys.clear();
            crate::clear_cached_region_ids();
            crate::request_pool_invalidation();
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            log::info!("event=caching_state enabled=false prev=true source=splice_focusable range={}..{} timestamp_ms={}", old_range.start, old_range.end, ts);
        }

        let mut old_items = state.items.cursor::<Count>(());
        let mut new_items = old_items.slice(&Count(old_range.start), Bias::Right);
        old_items.seek_forward(&Count(old_range.end), Bias::Right);

        let mut spliced_count = 0;
        new_items.extend(
            focus_handles.into_iter().map(|focus_handle| {
                spliced_count += 1;
                ListItem::Unmeasured { size_hint: None, focus_handle }
            }),
            (),
        );
        new_items.append(old_items.suffix(), ());
        drop(old_items);
        state.items = new_items;

        if let Some(ListOffset {
            item_ix,
            offset_in_item,
        }) = state.logical_scroll_top.as_mut()
        {
            if old_range.contains(item_ix) {
                *item_ix = old_range.start;
                *offset_in_item = px(0.);
            } else if old_range.end <= *item_ix {
                *item_ix = *item_ix - (old_range.end - old_range.start) + spliced_count;
            }
        }
    }

    /// Like [`Self::splice`], but each new item carries an estimated height for
    /// the SumTree summary. This allows the scrollbar to reflect realistic content
    /// size before items are measured. Pass `px(0.)` if no estimate is available.
    pub fn splice_with_heights(
        &self,
        old_range: Range<usize>,
        items: impl IntoIterator<Item = (Option<FocusHandle>, Pixels)>,
    ) {
        let state = &mut *self.0.borrow_mut();
        #[cfg(feature = "texture-cache")]
        if state.caching_enabled {
            state.caching_enabled = false;
            state.visible_frames.clear();
            state.cached_item_state_keys.clear();
            crate::clear_cached_region_ids();
            crate::request_pool_invalidation();
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            log::info!("event=caching_state enabled=false prev=true source=splice_with_heights range={}..{} timestamp_ms={}", old_range.start, old_range.end, ts);
        }

        let mut old_items = state.items.cursor::<Count>(());
        let mut new_items = old_items.slice(&Count(old_range.start), Bias::Right);
        old_items.seek_forward(&Count(old_range.end), Bias::Right);

        let mut spliced_count = 0;
        new_items.extend(
            items.into_iter().map(|(focus_handle, estimated_height)| {
                spliced_count += 1;
                ListItem::Unmeasured {
                    size_hint: Some(size(px(0.), estimated_height)),
                    focus_handle,
                }
            }),
            (),
        );
        new_items.append(old_items.suffix(), ());
        drop(old_items);
        state.items = new_items;

        if let Some(ListOffset {
            item_ix,
            offset_in_item,
        }) = state.logical_scroll_top.as_mut()
        {
            if old_range.contains(item_ix) {
                *item_ix = old_range.start;
                *offset_in_item = px(0.);
            } else if old_range.end <= *item_ix {
                *item_ix = *item_ix - (old_range.end - old_range.start) + spliced_count;
            }
        }
    }

    /// Set a handler that will be called when the list is scrolled.
    pub fn set_scroll_handler(
        &self,
        handler: impl FnMut(&ListScrollEvent, &mut Window, &mut App) + 'static,
    ) {
        self.0.borrow_mut().scroll_handler = Some(Box::new(handler))
    }

    /// Get the current scroll offset, in terms of the list's items.
    pub fn logical_scroll_top(&self) -> ListOffset {
        self.0.borrow().logical_scroll_top()
    }

    /// Scroll the list by the given offset
    pub fn scroll_by(&self, distance: Pixels) {
        if distance == px(0.) {
            return;
        }

        let current_offset = self.logical_scroll_top();
        let state = &mut *self.0.borrow_mut();
        let mut cursor = state.items.cursor::<ListItemSummary>(());
        cursor.seek(&Count(current_offset.item_ix), Bias::Right);

        let start_pixel_offset = cursor.start().height + current_offset.offset_in_item;
        // Clamp at both boundaries: 0 (top) and scroll_max (bottom).
        // Without bottom clamp, inertia animation can overshoot past content,
        // creating a visible gap between last item and viewport bottom.
        let bounds = state.last_layout_bounds.unwrap_or_default();
        let padding = state.last_padding.unwrap_or_default();
        let scroll_max = (state.items.summary().height + padding.top + padding.bottom
            - bounds.size.height)
            .max(px(0.));
        let new_pixel_offset = (start_pixel_offset + distance)
            .max(px(0.))
            .min(scroll_max);
        if new_pixel_offset > start_pixel_offset {
            cursor.seek_forward(&Height(new_pixel_offset), Bias::Right);
        } else {
            cursor.seek(&Height(new_pixel_offset), Bias::Right);
        }

        state.logical_scroll_top = Some(ListOffset {
            item_ix: cursor.start().count,
            offset_in_item: new_pixel_offset - cursor.start().height,
        });

        // Scrolling up suspends follow-tail (but stays in Tail mode for re-engagement)
        if distance < px(0.) {
            if let FollowState::Tail { ref mut is_following } = state.follow_state {
                *is_following = false;
            }
            // Cancel smooth scroll animation — user is scrolling away
            state.tail_scroll_velocity = 0.0;
        }
    }

    /// Suppress or enable the built-in scroll wheel handler.
    /// When suppressed, the caller is responsible for registering their own scroll
    /// handler (e.g., via `on_scroll_wheel` on a parent element).
    pub fn set_suppress_wheel_scroll(&self, suppress: bool) {
        self.0.borrow_mut().suppress_wheel_scroll = suppress;
    }

    /// Scroll the list to the given offset
    pub fn scroll_to(&self, mut scroll_top: ListOffset) {
        let state = &mut *self.0.borrow_mut();
        let item_count = state.items.summary().count;
        if scroll_top.item_ix >= item_count {
            scroll_top.item_ix = item_count;
            scroll_top.offset_in_item = px(0.);
        }

        // EC-13: Clear frame counters on programmatic scroll jump. Textures remain
        // valid (has_cached_region still returns true) so cached items composite
        // immediately. Only genuinely new items (first appearance after jump)
        // go through the transient skip (EC-6), spreading texture creation
        // across 2+ frames instead of a single-frame burst.
        #[cfg(feature = "texture-cache")]
        state.visible_frames.clear();

        state.logical_scroll_top = Some(scroll_top);

        // Explicit scroll-to suspends follow-tail
        if let FollowState::Tail { ref mut is_following } = state.follow_state {
            *is_following = false;
        }
    }

    /// Scroll the list to the given item, such that the item is fully visible.
    pub fn scroll_to_reveal_item(&self, ix: usize) {
        let state = &mut *self.0.borrow_mut();
        // EC-13: Clear frame counters — programmatic jump resets visibility.
        #[cfg(feature = "texture-cache")]
        state.visible_frames.clear();

        let mut scroll_top = state.logical_scroll_top();
        let height = state
            .last_layout_bounds
            .map_or(px(0.), |bounds| bounds.size.height);
        let padding = state.last_padding.unwrap_or_default();

        if ix <= scroll_top.item_ix {
            scroll_top.item_ix = ix;
            scroll_top.offset_in_item = px(0.);
        } else {
            let mut cursor = state.items.cursor::<ListItemSummary>(());
            cursor.seek(&Count(ix + 1), Bias::Right);
            let bottom = cursor.start().height + padding.top;
            let goal_top = px(0.).max(bottom - height + padding.bottom);

            cursor.seek(&Height(goal_top), Bias::Left);
            let start_ix = cursor.start().count;
            let start_item_top = cursor.start().height;

            if start_ix >= scroll_top.item_ix {
                scroll_top.item_ix = start_ix;
                scroll_top.offset_in_item = goal_top - start_item_top;
            }
        }

        state.logical_scroll_top = Some(scroll_top);
    }

    /// Get the bounds for the given item in window coordinates, if it's
    /// been rendered.
    pub fn bounds_for_item(&self, ix: usize) -> Option<Bounds<Pixels>> {
        let state = &*self.0.borrow();

        let bounds = state.last_layout_bounds.unwrap_or_default();
        let scroll_top = state.logical_scroll_top();
        if ix < scroll_top.item_ix {
            return None;
        }

        let mut cursor = state.items.cursor::<Dimensions<Count, Height>>(());
        cursor.seek(&Count(scroll_top.item_ix), Bias::Right);

        let scroll_top = cursor.start().1.0 + scroll_top.offset_in_item;

        cursor.seek_forward(&Count(ix), Bias::Right);
        if let Some(&ListItem::Measured { size, .. }) = cursor.item() {
            let &Dimensions(Count(count), Height(top), _) = cursor.start();
            if count == ix {
                let top = bounds.top() + top - scroll_top;
                return Some(Bounds::from_corners(
                    point(bounds.left(), top),
                    point(bounds.right(), top + size.height),
                ));
            }
        }
        None
    }

    /// Call this method when the user starts dragging the scrollbar.
    ///
    /// This will prevent the height reported to the scrollbar from changing during the drag
    /// as items in the overdraw get measured, and help offset scroll position changes accordingly.
    pub fn scrollbar_drag_started(&self) {
        let mut state = self.0.borrow_mut();
        state.scrollbar_drag_start_height = Some(state.items.summary().height);
    }

    /// Called when the user stops dragging the scrollbar.
    /// Unfreezes height and preserves the current scroll position. The user
    /// continues seeing the same content; only the scrollbar thumb adjusts
    /// to reflect the live content height (smooth transition vs instant jump).
    pub fn scrollbar_drag_ended(&self) {
        let mut state = self.0.borrow_mut();
        let frozen = state.scrollbar_drag_start_height.take();
        // Re-initialize smoothed height from live so it doesn't jump on first
        // non-drag scrollbar query after drag ends.
        state.smoothed_scrollbar_height = None;
        // Preserve scroll position through the unfreeze. The logical_scroll_top
        // was computed against frozen height — it points to the correct item/offset.
        // No adjustment needed: logical_scroll_top is item-index + offset-in-item,
        // which is independent of total height. The scrollbar thumb will smoothly
        // adjust to the live height on next paint.
        //
        // Only case needing care: if we were pinned to bottom (logical_scroll_top = None)
        // during drag, keep it pinned — the bottom-alignment semantics handle this.
        if let (Some(_frozen_h), Some(scroll_top)) = (frozen, state.logical_scroll_top) {
            let live_h = state.items.summary().height;
            // If heights differ significantly and we were near the bottom,
            // re-pin to bottom to avoid appearing stuck above new content.
            let bounds = state.last_layout_bounds.unwrap_or_default();
            let padding = state.last_padding.unwrap_or_default();
            let scroll_max = (live_h + padding.top + padding.bottom - bounds.size.height).max(px(0.));
            let scroll_pos = {
                let mut cursor = state.items.cursor::<ListItemSummary>(());
                let summary: ListItemSummary =
                    cursor.summary(&Count(scroll_top.item_ix), Bias::Right);
                summary.height + scroll_top.offset_in_item
            };
            if state.alignment == ListAlignment::Bottom && scroll_pos >= scroll_max {
                state.logical_scroll_top = None;
            }
        }
    }

    /// Pin the list to the bottom.
    /// For bottom-aligned lists, setting logical_scroll_top to None means "pinned to bottom" —
    /// GPUI will keep the view anchored to the last item.
    /// Note: Unused by CS (which uses ListAlignment::Top + scroll_to_max());
    /// retained for GPUI API compatibility with ListAlignment::Bottom callers.
    pub fn pin_to_bottom(&self) {
        self.0.borrow_mut().logical_scroll_top = None;
    }

    /// Returns true if the list is currently pinned to the bottom.
    /// Only meaningful for ListAlignment::Bottom lists.
    /// Note: Unused by CS (which uses is_at_bottom()); retained for GPUI API compatibility.
    pub fn is_pinned_to_bottom(&self) -> bool {
        let state = self.0.borrow();
        state.alignment == ListAlignment::Bottom && state.logical_scroll_top.is_none()
    }

    /// Set the offset from the scrollbar
    pub fn set_offset_from_scrollbar(&self, point: Point<Pixels>) {
        self.0.borrow_mut().set_offset_from_scrollbar(point);
    }

    /// Returns the maximum scroll offset according to the items we have measured.
    /// During drag, uses frozen height but allows it to grow if content has grown —
    /// prevents getting stuck at stale max while allowing access to new content.
    pub fn max_offset_for_scrollbar(&self) -> Point<Pixels> {
        let mut state = self.0.borrow_mut();
        let bounds = state.last_layout_bounds.unwrap_or_default();
        let live_height = state.items.summary().height;

        let height = match state.scrollbar_drag_start_height {
            Some(frozen) if live_height > frozen => {
                // Content grew during drag — update frozen to live so new content
                // is reachable. Only grows, never shrinks — thumb position stays stable.
                state.scrollbar_drag_start_height = Some(live_height);
                live_height
            }
            Some(frozen) => frozen,
            None => {
                // Not dragging — apply capped-lerp smoothing to prevent thumb
                // jumps when items are re-measured with large height deltas
                // (e.g., scroll-up through items with estimated heights).
                // The lerp (30%) tracks streaming growth well for small deltas.
                // Proportional cap (0.5% of live height) scales with content size:
                //   5000px content → 25px/frame cap (tracks 5× faster than old 5px cap)
                //   50000px content → 250px/frame cap
                let smoothed = match state.smoothed_scrollbar_height {
                    Some(prev) => {
                        let delta = live_height - prev;
                        let lerp_step = delta * 0.3;
                        let max_step = Pixels(live_height.0 * 0.005);
                        let step = if lerp_step.0.abs() > max_step.0 {
                            Pixels(max_step.0 * lerp_step.0.signum())
                        } else {
                            lerp_step
                        };
                        prev + step
                    }
                    None => live_height,
                };
                state.smoothed_scrollbar_height = Some(smoothed);
                smoothed
            }
        };

        let padding = state.last_padding.unwrap_or_default();
        let padded = height + padding.top + padding.bottom;
        point(Pixels::ZERO, Pixels::ZERO.max(padded - bounds.size.height))
    }

    /// Scroll to the maximum offset (bottom of content).
    /// Unlike pin_to_bottom() which sets logical_scroll_top=None (a sentinel),
    /// this computes the actual scroll position. Works with any ListAlignment.
    pub fn scroll_to_max(&self) {
        let state = &mut *self.0.borrow_mut();
        #[cfg(feature = "texture-cache")]
        let old_top = state.logical_scroll_top;
        // EC-13: Clear frame counters — programmatic jump resets visibility.
        #[cfg(feature = "texture-cache")]
        state.visible_frames.clear();
        let bounds = state.last_layout_bounds.unwrap_or_default();
        let padding = state.last_padding.unwrap_or_default();
        let total_height = state.items.summary().height;
        let scroll_max =
            (total_height + padding.top + padding.bottom - bounds.size.height).max(px(0.));

        // S497: Smooth scrolling for follow_output mode. When active and the
        // jump to bottom exceeds 50px, spread the movement over ~4 frames
        // instead of snapping. Prevents visual jank from content bursts.
        // Called once per rendered frame by follow_output; each call advances
        // one step, host re-calls on next frame until is_at_bottom() is true.
        #[cfg(feature = "texture-cache")]
        let effective_target = if state.smooth_scroll_active {
            // Compute current scroll position in pixels.
            // For Bottom-aligned pinned lists, logical_scroll_top() returns
            // item_ix=count → current_px = total_height → delta negative → skip.
            let current = state.logical_scroll_top();
            let mut cursor = state.items.cursor::<ListItemSummary>(());
            let summary: ListItemSummary =
                cursor.summary(&Count(current.item_ix), Bias::Right);
            let current_px = summary.height + current.offset_in_item;
            let delta = scroll_max - current_px;
            let delta_f32 = f32::from(delta);

            // delta < 0 means content shrank — fall through to scroll_max.
            if delta_f32 > 50.0 {
                // Spread over ~4 frames, but never step less than 40px
                // to avoid falling behind content growth.
                let step_px = (delta_f32 / 4.0).max(40.0);
                let frame_target = (current_px + px(step_px)).min(scroll_max);
                log::info!(
                    "event=scroll_to_max_smoothed paint_frame={} delta_px={:.1} step_px={:.1} frame_target_px={:.1} scroll_max_px={:.1}",
                    state.paint_frame_count, delta_f32, step_px, f32::from(frame_target), f32::from(scroll_max)
                );
                frame_target
            } else {
                scroll_max
            }
        } else {
            scroll_max
        };
        #[cfg(not(feature = "texture-cache"))]
        let effective_target = scroll_max;

        let (start, ..) =
            state
                .items
                .find::<ListItemSummary, _>((), &Height(effective_target), Bias::Right);
        state.logical_scroll_top = Some(ListOffset {
            item_ix: start.count,
            offset_in_item: effective_target - start.height,
        });

        // Log completion when smoothing reaches the target
        #[cfg(feature = "texture-cache")]
        if state.smooth_scroll_active && effective_target == scroll_max {
            log::info!(
                "event=scroll_smooth_complete paint_frame={} final_px={:.1}",
                state.paint_frame_count, f32::from(scroll_max)
            );
        }

        #[cfg(feature = "texture-cache")]
        log::info!(
            "event=scroll_to_max paint_frame={} old_ix={} old_off={:.1} new_ix={} new_off={:.1} scroll_max_px={:.1} total_height_px={:.1} viewport_px={:.1}",
            state.paint_frame_count,
            old_top.map_or(0, |o| o.item_ix),
            old_top.map_or(0.0, |o| f32::from(o.offset_in_item)),
            start.count,
            f32::from(effective_target - start.height),
            f32::from(scroll_max),
            f32::from(total_height),
            f32::from(bounds.size.height)
        );
    }

    /// Returns true if the list is scrolled to (or very near) the bottom.
    /// Works with any alignment by comparing current scroll position to max.
    pub fn is_at_bottom(&self) -> bool {
        let state = self.0.borrow();
        let bounds = state.last_layout_bounds.unwrap_or_default();
        let padding = state.last_padding.unwrap_or_default();
        let total_height = state.items.summary().height;
        let scroll_max =
            (total_height + padding.top + padding.bottom - bounds.size.height).max(px(0.));

        if scroll_max == px(0.) {
            return true; // Content fits in viewport
        }

        let scroll_top = state.logical_scroll_top();
        let mut cursor = state.items.cursor::<ListItemSummary>(());
        let summary: ListItemSummary =
            cursor.summary(&Count(scroll_top.item_ix), Bias::Right);
        let current_pos = summary.height + scroll_top.offset_in_item;

        (scroll_max - current_pos) < px(1.0)
    }

    /// Returns true if the list is within `tolerance` pixels of the bottom.
    /// Use for re-engagement checks where height estimation oscillation (±25px)
    /// makes the strict 1px `is_at_bottom()` unreliable.
    pub fn is_near_bottom(&self, tolerance: Pixels) -> bool {
        let state = self.0.borrow();
        let bounds = state.last_layout_bounds.unwrap_or_default();
        let padding = state.last_padding.unwrap_or_default();
        let total_height = state.items.summary().height;
        let scroll_max =
            (total_height + padding.top + padding.bottom - bounds.size.height).max(px(0.));

        if scroll_max == px(0.) {
            return true;
        }

        let scroll_top = state.logical_scroll_top();
        let mut cursor = state.items.cursor::<ListItemSummary>(());
        let summary: ListItemSummary =
            cursor.summary(&Count(scroll_top.item_ix), Bias::Right);
        let current_pos = summary.height + scroll_top.offset_in_item;

        (scroll_max - current_pos) < tolerance
    }

    /// Returns true if the scrollbar is currently being dragged.
    /// The Scrollbar component sets this via `scrollbar_drag_started()`/`scrollbar_drag_ended()`.
    /// Useful for detecting mouse-up without relying on position-stable timeouts.
    pub fn is_scrollbar_dragging(&self) -> bool {
        self.0.borrow().scrollbar_drag_start_height.is_some()
    }

    /// Returns the current scroll offset adjusted for the scrollbar.
    /// S450: Removed drag_offset — consistent with set_offset_from_scrollbar
    /// which no longer uses drag_offset during drag.
    pub fn scroll_px_offset_for_scrollbar(&self) -> Point<Pixels> {
        let state = &self.0.borrow();
        let logical_scroll_top = state.logical_scroll_top();

        let mut cursor = state.items.cursor::<ListItemSummary>(());
        let summary: ListItemSummary =
            cursor.summary(&Count(logical_scroll_top.item_ix), Bias::Right);
        let offset = summary.height + logical_scroll_top.offset_in_item;

        Point::new(px(0.), -offset)
    }

    /// Return the bounds of the viewport in pixels.
    pub fn viewport_bounds(&self) -> Bounds<Pixels> {
        self.0.borrow().last_layout_bounds.unwrap_or_default()
    }

    /// Set the follow mode for the list.
    ///
    /// `FollowMode::Tail` causes the list to automatically scroll to the end
    /// whenever items are laid out. If the user scrolls away, following is
    /// suspended but re-engages when they scroll back to the bottom.
    /// Scrollbar drag exits tail mode entirely.
    pub fn set_follow_mode(&self, mode: FollowMode) {
        let mut state = self.0.borrow_mut();
        match mode {
            FollowMode::Normal => {
                state.follow_state = FollowState::Normal;
                state.tail_scroll_velocity = 0.0;
                state.last_inertia_time = None;
            }
            FollowMode::Tail => {
                state.follow_state = FollowState::Tail { is_following: true };
                // Reset growth tracking and velocity for fresh start
                state.tail_scroll_velocity = 0.0;
                state.prev_tail_item_count = 0;
                state.prev_tail_scroll_max = px(0.);
                state.last_inertia_time = None;
            }
        }
    }

    /// Returns true if the list is currently auto-following the tail.
    pub fn is_following_tail(&self) -> bool {
        matches!(
            self.0.borrow().follow_state,
            FollowState::Tail { is_following: true }
        )
    }

    /// Scroll to the end of the list. Alias for `scroll_to_max()`.
    pub fn scroll_to_end(&self) {
        self.scroll_to_max();
    }

    /// Returns true if a smooth scroll animation is in progress (inertia scroll).
    pub fn is_smooth_scrolling(&self) -> bool {
        self.0.borrow().tail_scroll_velocity.abs() > MIN_VELOCITY_PX_PER_SEC
    }

    /// Cancel any in-progress smooth scroll animation.
    pub fn cancel_smooth_scroll(&self) {
        let state = &mut *self.0.borrow_mut();
        state.tail_scroll_velocity = 0.0;
    }

    /// Enable or disable GPU texture caching for list items during scroll.
    /// When enabled, items are annotated for render-to-texture capture.
    /// `clear_color` should be the card's opaque background color (for subpixel text).
    #[cfg(feature = "texture-cache")]
    pub fn set_item_caching_enabled(&self, enabled: bool, clear_color: Hsla) {
        let mut inner = self.0.borrow_mut();
        let prev = inner.caching_enabled;
        inner.caching_enabled = enabled;
        inner.cache_clear_color = clear_color;
        // Clear stale height tracking on disable so re-enable starts with a clean slate.
        // Prevents spurious height_change events from stale index→height associations
        // after splice/invalidation changes item ordering.
        if prev && !enabled {
            inner.prev_item_heights.clear();
        }
        // S502: scroll resumed — fade is stale. Snap cached composites back to
        // full opacity so the HIT path renders as normal. CS's
        // FadeComposeController also calls cancel() at the matching call
        // sites; this reset is the last line of defense when a future
        // caller adds a new `set_item_caching_enabled(true, …)` site and
        // forgets to plumb the CS-side cancel.
        if enabled && !prev {
            inner.fade_alpha = 1.0;
        }
        if prev != enabled {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            log::info!("event=caching_state enabled={} prev={} source=set_item_caching_enabled timestamp_ms={}", enabled, prev, ts);
        }
    }

    /// Check if item caching is currently enabled.
    #[cfg(feature = "texture-cache")]
    pub fn is_item_caching_enabled(&self) -> bool {
        self.0.borrow().caching_enabled
    }

    /// S502: Set the fade alpha applied to cached-texture composites during
    /// the texture→layout transition crossfade. 1.0 = no fade (normal cached
    /// composite). 0.0 = fully transparent (texture invisible, DIRECT paint
    /// shows). CS's `FadeComposeController::tick()` is the sole caller — it
    /// drives this toward 0.0 over the 100 ms fade window and calls cancel()
    /// (alpha back to 1.0) on scroll resume.
    #[cfg(feature = "texture-cache")]
    pub fn set_fade_alpha(&self, alpha: f32) {
        self.0.borrow_mut().fade_alpha = alpha.clamp(0.0, 1.0);
    }

    /// Returns the monotonic paint frame counter.
    /// Incremented once per paint() call.
    /// Useful for correlating events that happen between paints.
    #[cfg(feature = "texture-cache")]
    pub fn paint_frame_count(&self) -> u64 {
        self.0.borrow().paint_frame_count
    }

    /// Returns the current logical scroll position (item index + offset within item).
    #[cfg(feature = "texture-cache")]
    pub fn current_scroll_top(&self) -> ListOffset {
        self.0.borrow().logical_scroll_top()
    }

    /// Disable GPU texture caching and reset per-item frame counters.
    /// Called on scroll stop to restore full hit-test interactivity.
    ///
    /// NOTE: This does NOT purge the renderer feedback set. Textures already
    /// cached remain valid and will be composited if caching is re-enabled.
    /// For full purge (DPI change, theme change), use `invalidate_all_caches()`.
    #[cfg(feature = "texture-cache")]
    pub fn invalidate_all_item_caches(&self) {
        let mut inner = self.0.borrow_mut();
        let prev = inner.caching_enabled;
        inner.caching_enabled = false;
        inner.visible_frames.clear();
        inner.prev_item_heights.clear();
        inner.cached_item_state_keys.clear();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        log::info!("event=caching_state enabled=false prev={} source=invalidate_all_item_caches timestamp_ms={}", prev, ts);
    }

    // --- Selective cache invalidation ---
    // Use `invalidate_item_cache` for single-item changes (content update, collapse).
    // Use `invalidate_all_caches` for global changes (DPI, font, theme).
    // Use `set_streaming_items` to exclude live-updating items from caching.

    /// FR-2.2: Invalidate the cached GPU texture for a single item.
    /// The item re-renders Fresh on the next frame. Does NOT disable caching globally.
    /// Used for: card collapse/expand (EC-1), search highlight changes (EC-2),
    /// explicit content changes.
    ///
    /// `index` must be the list index (including any spacer offsets), not the
    /// items-array index — this is what the paint loop uses as `CacheRegionId`.
    #[cfg(feature = "texture-cache")]
    pub fn invalidate_item_cache(&self, index: usize) {
        let region_id = CacheRegionId(index as u64);
        // Remove this item from the renderer's "valid cache" feedback set.
        // On the next paint, has_cached_region() returns false → cache MISS → re-render.
        crate::clear_cached_region(region_id);
        // Signal the renderer to purge the old GPU texture from TexturePool.active.
        // Without this, the 1-frame feedback delay means the stale texture could
        // be served as HIT on the first scroll frame after invalidation.
        crate::invalidate_pool_region(region_id);
        // Reset frame visibility counter so the item goes through the transient
        // skip check again (avoids caching a single-frame flash). Also drop any
        // captured element-state keys so a subsequent HIT does not re-extend
        // stale keys after content/collapse invalidation (S500 §3.1 cleanup).
        let mut inner = self.0.borrow_mut();
        inner.visible_frames.remove(&index);
        inner.cached_item_state_keys.remove(&index);
    }

    /// EC-3/4/5: Invalidate ALL cached GPU textures and clear the renderer feedback set.
    /// Used for global visual changes: theme, font size, DPI, scale factor.
    /// Unlike `invalidate_all_item_caches()`, this also purges renderer-side state
    /// so stale textures are never composited.
    #[cfg(feature = "texture-cache")]
    pub fn invalidate_all_caches(&self) {
        let mut inner = self.0.borrow_mut();
        let prev = inner.caching_enabled;
        inner.caching_enabled = false;
        inner.visible_frames.clear();
        inner.streaming_items.clear();
        inner.prev_item_heights.clear();
        inner.cached_item_state_keys.clear();
        // Clear the renderer feedback set — all cached textures become stale.
        crate::clear_cached_region_ids();
        // Signal the renderer to flush all GPU texture pool resources on the next frame.
        crate::request_pool_invalidation();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        log::info!("event=caching_state enabled=false prev={} source=invalidate_all_caches timestamp_ms={}", prev, ts);
    }

    /// FR-4: Set the indices of items that are actively streaming content.
    /// Streaming items are excluded from GPU texture caching in the paint loop —
    /// they render Fresh every frame because their content changes each chunk.
    /// Call with an empty set when no items are streaming.
    #[cfg(feature = "texture-cache")]
    pub fn set_streaming_items(&self, items: HashSet<usize>) {
        self.0.borrow_mut().streaming_items = items;
    }

    /// Enable/disable smooth scrolling for scroll_to_max(). When active,
    /// large jumps (>50px) are spread over multiple frames. Set by the host
    /// app during follow_output/tail mode to prevent visual jank from
    /// content arriving in bursts.
    #[cfg(feature = "texture-cache")]
    pub fn set_smooth_scroll(&self, active: bool) {
        self.0.borrow_mut().smooth_scroll_active = active;
    }

    /// Set item indices that should emit detailed trace logs through the cache
    /// pipeline. Used to diagnose specific card types (e.g. AskUserQuestion)
    /// that may have rendering issues in offscreen textures.
    #[cfg(feature = "texture-cache")]
    pub fn set_trace_items(&self, items: HashSet<usize>) {
        self.0.borrow_mut().trace_items = items;
    }
}

impl StateInner {
    fn visible_range(
        items: &SumTree<ListItem>,
        height: Pixels,
        scroll_top: &ListOffset,
    ) -> Range<usize> {
        let mut cursor = items.cursor::<ListItemSummary>(());
        cursor.seek(&Count(scroll_top.item_ix), Bias::Right);
        let start_y = cursor.start().height + scroll_top.offset_in_item;
        cursor.seek_forward(&Height(start_y + height), Bias::Left);
        scroll_top.item_ix..cursor.start().count + 1
    }

    fn scroll(
        &mut self,
        scroll_top: &ListOffset,
        height: Pixels,
        delta: Point<Pixels>,
        current_view: EntityId,
        window: &mut Window,
        cx: &mut App,
    ) {
        // Drop scroll events after a reset, since we can't calculate
        // the new logical scroll top without the item heights
        if self.reset {
            return;
        }

        let padding = self.last_padding.unwrap_or_default();
        let scroll_max =
            (self.items.summary().height + padding.top + padding.bottom - height).max(px(0.));
        let new_scroll_top = (self.scroll_top(scroll_top) - delta.y)
            .max(px(0.))
            .min(scroll_max);

        if self.alignment == ListAlignment::Bottom && new_scroll_top == scroll_max {
            self.logical_scroll_top = None;
        } else {
            let (start, ..) =
                self.items
                    .find::<ListItemSummary, _>((), &Height(new_scroll_top), Bias::Right);
            let item_ix = start.count;
            let offset_in_item = new_scroll_top - start.height;
            self.logical_scroll_top = Some(ListOffset {
                item_ix,
                offset_in_item,
            });
        }

        // Wheel scroll away from bottom suspends follow-tail
        if let FollowState::Tail { ref mut is_following } = self.follow_state {
            if new_scroll_top < scroll_max {
                *is_following = false;
            }
        }

        let is_following_tail = matches!(
            self.follow_state,
            FollowState::Tail { is_following: true }
        );

        if let Some(handler) = self.scroll_handler.as_mut() {
            let visible_range = Self::visible_range(&self.items, height, scroll_top);
            handler(
                &ListScrollEvent {
                    visible_range,
                    count: self.items.summary().count,
                    is_scrolled: self.logical_scroll_top.is_some(),
                    is_following_tail,
                },
                window,
                cx,
            );
        }

        cx.notify(current_view);
    }

    fn logical_scroll_top(&self) -> ListOffset {
        self.logical_scroll_top
            .unwrap_or_else(|| match self.alignment {
                ListAlignment::Top => ListOffset {
                    item_ix: 0,
                    offset_in_item: px(0.),
                },
                ListAlignment::Bottom => ListOffset {
                    item_ix: self.items.summary().count,
                    offset_in_item: px(0.),
                },
            })
    }

    fn scroll_top(&self, logical_scroll_top: &ListOffset) -> Pixels {
        let (start, ..) = self.items.find::<ListItemSummary, _>(
            (),
            &Count(logical_scroll_top.item_ix),
            Bias::Right,
        );
        start.height + logical_scroll_top.offset_in_item
    }

    /// Set `logical_scroll_top` to the ListOffset corresponding to a pixel position.
    fn set_logical_scroll_top_to(&mut self, position: Pixels) {
        let (start, ..) =
            self.items
                .find::<ListItemSummary, _>((), &Height(position), Bias::Right);
        self.logical_scroll_top = Some(ListOffset {
            item_ix: start.count,
            offset_in_item: position - start.height,
        });
    }

    fn layout_all_items(
        &mut self,
        available_width: Pixels,
        render_item: &mut RenderItemFn,
        window: &mut Window,
        cx: &mut App,
    ) {
        match &mut self.measuring_behavior {
            ListMeasuringBehavior::Visible => {
                return;
            }
            ListMeasuringBehavior::Measure(has_measured) => {
                if *has_measured {
                    return;
                }
                *has_measured = true;
            }
        }

        let mut cursor = self.items.cursor::<Count>(());
        let available_item_space = size(
            AvailableSpace::Definite(available_width),
            AvailableSpace::MinContent,
        );

        let mut measured_items = Vec::default();

        for (ix, item) in cursor.enumerate() {
            let size = item.size().unwrap_or_else(|| {
                let mut element = render_item(ix, window, cx);
                element.layout_as_root(available_item_space, window, cx)
            });

            measured_items.push(ListItem::Measured {
                size,
                focus_handle: item.focus_handle(),
            });
        }

        self.items = SumTree::from_iter(measured_items, ());
    }

    fn layout_items(
        &mut self,
        available_width: Option<Pixels>,
        available_height: Pixels,
        padding: &Edges<Pixels>,
        render_item: &mut RenderItemFn,
        window: &mut Window,
        cx: &mut App,
    ) -> LayoutItemsResponse {
        // S498: Layout phase timing — measures full layout_items() cost.
        #[cfg(feature = "texture-cache-debug")]
        let layout_phase_start = std::time::Instant::now();
        #[cfg(feature = "texture-cache-debug")]
        {
            let item_count = self.items.summary().count;
            log::info!(
                "event=layout_phase_start items={} caching={}",
                item_count, self.caching_enabled
            );
        }

        // If following tail, scroll toward end before layout.
        // Inertia-based smooth scroll for Tail mode (delta-time).
        // Content growth adds velocity impulses; friction decays velocity per second.
        // All constants are normalized to 60 FPS feel — animation speed is constant
        // regardless of actual frame rate (Wayland VSync, thermal throttling, etc.).
        //
        // Two impulse factors: IMPULSE_FACTOR (streaming — converges to growth rate)
        // and NEW_CARD_IMPULSE (card insertion — faster response for discrete jumps).

        // Friction per frame at 60 FPS: 0.80. Per second: 0.80^60.
        // At any FPS: friction_per_frame = FRICTION_60.powf(dt * 60.0)
        const FRICTION_60: f32 = 0.80;
        // Impulse factors scaled by (dt * 60): at 60 FPS, dt=1/60, so dt*60=1 → same as before.
        const IMPULSE_FACTOR: f32 = 0.20;
        const NEW_CARD_IMPULSE: f32 = 0.35;
        // Velocity threshold — see module-level MIN_VELOCITY_PX_PER_SEC.
        const MIN_VELOCITY: f32 = MIN_VELOCITY_PX_PER_SEC;

        if let FollowState::Tail { is_following: true } = self.follow_state {
            let total_height = self.items.summary().height;
            let scroll_max =
                (total_height + padding.top + padding.bottom - available_height).max(px(0.));
            let current_item_count = self.items.summary().count;

            if scroll_max > px(0.) {
                let current_pos = self.logical_scroll_top
                    .map(|off| self.scroll_top(&off))
                    .unwrap_or(px(0.));

                // Delta-time: seconds since last inertia update.
                // First frame after follow-start: use 1/60 as default.
                // Cap at 50ms to prevent velocity explosion after pause.
                let now = Instant::now();
                let dt = self.last_inertia_time
                    .map(|last| now.duration_since(last).as_secs_f32())
                    .unwrap_or(1.0 / 60.0)
                    .min(0.05);
                self.last_inertia_time = Some(now);

                // 1. Apply friction (time-normalized: same decay per second at any FPS)
                self.tail_scroll_velocity *= FRICTION_60.powf(dt * 60.0);

                // 2. Add impulse from content growth this frame.
                // scroll_growth is per-frame (px). Convert to velocity impulse (px/s):
                //   growth_rate_px_per_sec = scroll_growth / dt
                //   impulse = growth_rate * IMPULSE_FACTOR
                // Simplified: growth * IMPULSE_FACTOR * 60.0 (normalized to 60fps base).
                // The *60 converts the per-frame factor to per-second: at 60fps, one frame's
                // impulse = growth * factor. At 30fps, growth is 2x larger (same total rate)
                // so impulse per second stays constant.
                if self.prev_tail_scroll_max > px(0.) {
                    let scroll_growth = f32::from(scroll_max - self.prev_tail_scroll_max);
                    if scroll_growth > 0.5 {
                        let factor = if current_item_count > self.prev_tail_item_count {
                            NEW_CARD_IMPULSE
                        } else {
                            IMPULSE_FACTOR
                        };
                        self.tail_scroll_velocity += scroll_growth * factor * 60.0;
                    }
                }

                // 3. Apply velocity or snap
                // Velocity is now px/s. Displacement = velocity * dt.
                if self.tail_scroll_velocity.abs() > MIN_VELOCITY {
                    let displacement = self.tail_scroll_velocity * dt;
                    let new_pos = (current_pos + px(displacement)).min(scroll_max);
                    self.set_logical_scroll_top_to(new_pos);
                } else {
                    self.tail_scroll_velocity = 0.0;
                    self.set_logical_scroll_top_to(scroll_max);
                }

                // Telemetry: emit via registered callback (cs-debug Scroll tab).
                // Only fires when inertia is active or content grew — silent at rest.
                {
                    let growth = f32::from(scroll_max - self.prev_tail_scroll_max);
                    if self.tail_scroll_velocity.abs() > 0.1 || growth > 0.1 {
                        emit_scroll_telemetry(&ScrollTelemetry {
                            velocity: self.tail_scroll_velocity,
                            dt,
                            delta: f32::from(scroll_max - current_pos),
                            growth,
                            item_count: current_item_count,
                            inertia_active: self.tail_scroll_velocity.abs() > MIN_VELOCITY,
                        });
                    }
                }

                self.prev_tail_item_count = current_item_count;
                self.prev_tail_scroll_max = scroll_max;
            }
        }

        let old_items = self.items.clone();
        let mut measured_items = VecDeque::new();
        let mut item_layouts = VecDeque::new();
        let mut rendered_height = padding.top;
        let mut max_item_width = px(0.);
        let mut scroll_top = self.logical_scroll_top();
        let mut rendered_focused_item = false;

        let available_item_space = size(
            available_width.map_or(AvailableSpace::MinContent, |width| {
                AvailableSpace::Definite(width)
            }),
            AvailableSpace::MinContent,
        );

        let mut cursor = old_items.cursor::<Count>(());

        // Perf telemetry: track per-item render cost
        let layout_start = Instant::now();
        let mut perf_rendered_count: usize = 0;
        let mut perf_cached_count: usize = 0;
        let mut perf_slowest_secs: f32 = 0.0;
        // S498: Accumulate total render_item cost for layout breakdown.
        #[cfg(feature = "texture-cache-debug")]
        let mut render_sum_secs: f32 = 0.0;
        let mut perf_slowest_ix: usize = 0;

        // Render items after the scroll top, including those in the trailing overdraw
        cursor.seek(&Count(scroll_top.item_ix), Bias::Right);
        for (ix, item) in cursor.by_ref().enumerate() {
            let visible_height = rendered_height - scroll_top.offset_in_item;
            if visible_height >= available_height + self.overdraw {
                break;
            }

            // Use the previously cached height and focus handle if available
            let mut size = item.size();

            // If we're within the visible area or the height wasn't cached, render and measure the item's element.
            // EC-12: When caching is active, also render overdraw items so they can be texture-cached.
            #[cfg(feature = "texture-cache")]
            let should_render = visible_height < available_height
                || size.is_none()
                || (self.caching_enabled && crate::is_overdraw_caching_enabled() && visible_height < available_height + self.overdraw);
            #[cfg(not(feature = "texture-cache"))]
            let should_render = visible_height < available_height || size.is_none();
            if should_render {
                let item_index = scroll_top.item_ix + ix;
                let item_start = Instant::now();
                let mut element = render_item(item_index, window, cx);
                let element_size = element.layout_as_root(available_item_space, window, cx);
                let item_elapsed = item_start.elapsed().as_secs_f32();
                perf_rendered_count += 1;
                // S498: Accumulate per-item render cost + log slow items.
                #[cfg(feature = "texture-cache-debug")]
                {
                    render_sum_secs += item_elapsed;
                    if item_elapsed > 0.0005 {
                        log::info!(
                            "event=render_item_cost ix={} ms={:.1}",
                            item_index, item_elapsed * 1000.0
                        );
                    }
                }
                if item_elapsed > perf_slowest_secs {
                    perf_slowest_secs = item_elapsed;
                    perf_slowest_ix = item_index;
                }
                size = Some(element_size);

                if ix == 0 {
                    // CS patch: Scroll offset compensation (wheel/trackpad scroll only).
                    // During scrollbar drag, scroll position comes from the scrollbar's
                    // absolute pixel mapping — compensation is not needed and would
                    // compound each frame since we freeze SumTree heights during drag.
                    if self.scrollbar_drag_start_height.is_none() {
                        let old_height = item.size_hint()
                            .map(|s| s.height)
                            .unwrap_or(px(0.));
                        let height_delta = element_size.height - old_height;
                        if height_delta != px(0.) && scroll_top.offset_in_item > px(0.) {
                            let offset_before = f32::from(scroll_top.offset_in_item);
                            scroll_top.offset_in_item = (scroll_top.offset_in_item + height_delta)
                                .max(px(0.))
                                .min(element_size.height);
                            self.logical_scroll_top = Some(scroll_top);
                            emit_offset_compensation(&OffsetCompensationEvent {
                                item_ix: scroll_top.item_ix,
                                old_height: f32::from(old_height),
                                new_height: f32::from(element_size.height),
                                height_delta: f32::from(height_delta),
                                offset_before,
                                offset_after: f32::from(scroll_top.offset_in_item),
                            });
                        }
                    }

                    // If there's a pending scroll adjustment (from remeasure_items),
                    // apply it — proportional preservation takes priority over our
                    // delta correction.
                    if let Some(pending_scroll) = self.pending_scroll.take() {
                        if pending_scroll.item_ix == scroll_top.item_ix {
                            scroll_top.offset_in_item =
                                Pixels(pending_scroll.fraction * element_size.height.0);
                            self.logical_scroll_top = Some(scroll_top);
                        }
                    }
                }

                // EC-12: include_in_layout controls whether a rendered item enters
                // the prepaint+paint pipeline. Separate from should_render above:
                // overdraw items are always rendered (CPU layout) but only included
                // in the paint list when caching is active, so their textures are
                // ready before they scroll into view.
                #[cfg(feature = "texture-cache")]
                let include_in_layout = visible_height < available_height
                    || (self.caching_enabled && crate::is_overdraw_caching_enabled() && visible_height < available_height + self.overdraw);
                #[cfg(not(feature = "texture-cache"))]
                let include_in_layout = visible_height < available_height;

                if include_in_layout {
                    // S498 INV-X9: Overdraw items should only be in layout when caching is active.
                    #[cfg(feature = "texture-cache-debug")]
                    if visible_height >= available_height && !self.caching_enabled {
                        log::warn!(
                            "event=INVARIANT_VIOLATION rule=X9 ix={} reason=overdraw_without_caching visible_h={:.0} available_h={:.0}",
                            item_index, f32::from(visible_height), f32::from(available_height)
                        );
                    }
                    item_layouts.push_back(ItemLayout {
                        index: item_index,
                        element,
                        size: element_size,
                        #[cfg(feature = "texture-cache")]
                        origin: Point::default(),
                        #[cfg(feature = "texture-cache")]
                        is_overdraw: visible_height >= available_height,
                    });
                    if item.contains_focused(window, cx) {
                        rendered_focused_item = true;
                    }
                }
            } else {
                perf_cached_count += 1;
            }

            let size = size.unwrap();
            rendered_height += size.height;
            max_item_width = max_item_width.max(size.width);
            // During scrollbar drag, preserve old SumTree heights to keep
            // the pixel→item scroll mapping stable across frames.
            // Visual layout uses actual measured sizes (above); only the
            // SumTree entry is frozen to prevent mapping drift.
            let tree_size = if self.scrollbar_drag_start_height.is_some() {
                item.size_hint().unwrap_or(size)
            } else {
                size
            };
            measured_items.push_back(ListItem::Measured {
                size: tree_size,
                focus_handle: item.focus_handle(),
            });
        }
        rendered_height += padding.bottom;

        // Prepare to start walking upward from the item at the scroll top.
        cursor.seek(&Count(scroll_top.item_ix), Bias::Right);

        // If the rendered items do not fill the visible region, then adjust
        // the scroll top upward.
        if rendered_height - scroll_top.offset_in_item < available_height {
            while rendered_height < available_height {
                cursor.prev();
                if let Some(item) = cursor.item() {
                    let item_index = cursor.start().0;
                    let mut element = render_item(item_index, window, cx);
                    let element_size = element.layout_as_root(available_item_space, window, cx);
                    let focus_handle = item.focus_handle();
                    rendered_height += element_size.height;
                    let tree_size = if self.scrollbar_drag_start_height.is_some() {
                        item.size_hint().unwrap_or(element_size)
                    } else {
                        element_size
                    };
                    measured_items.push_front(ListItem::Measured {
                        size: tree_size,
                        focus_handle,
                    });
                    item_layouts.push_front(ItemLayout {
                        index: item_index,
                        element,
                        size: element_size,
                        #[cfg(feature = "texture-cache")]
                        origin: Point::default(),
                        #[cfg(feature = "texture-cache")]
                        is_overdraw: false,
                    });
                    if item.contains_focused(window, cx) {
                        rendered_focused_item = true;
                    }
                } else {
                    break;
                }
            }

            scroll_top = ListOffset {
                item_ix: cursor.start().0,
                offset_in_item: rendered_height - available_height,
            };

            match self.alignment {
                ListAlignment::Top => {
                    scroll_top.offset_in_item = scroll_top.offset_in_item.max(px(0.));
                    self.logical_scroll_top = Some(scroll_top);
                }
                ListAlignment::Bottom => {
                    scroll_top = ListOffset {
                        item_ix: cursor.start().0,
                        offset_in_item: rendered_height - available_height,
                    };
                    self.logical_scroll_top = None;
                }
            };
        }

        // Measure items in the leading overdraw
        let mut leading_overdraw = scroll_top.offset_in_item;
        while leading_overdraw < self.overdraw {
            cursor.prev();
            if let Some(item) = cursor.item() {
                let (size, tree_size) = if let ListItem::Measured { size, .. } = item {
                    (*size, *size)
                } else {
                    let mut element = render_item(cursor.start().0, window, cx);
                    let actual = element.layout_as_root(available_item_space, window, cx);
                    let tree = if self.scrollbar_drag_start_height.is_some() {
                        item.size_hint().unwrap_or(actual)
                    } else {
                        actual
                    };
                    (actual, tree)
                };

                leading_overdraw += size.height;
                measured_items.push_front(ListItem::Measured {
                    size: tree_size,
                    focus_handle: item.focus_handle(),
                });
            } else {
                break;
            }
        }

        let measured_range = cursor.start().0..(cursor.start().0 + measured_items.len());
        let mut cursor = old_items.cursor::<Count>(());
        let mut new_items = cursor.slice(&Count(measured_range.start), Bias::Right);
        new_items.extend(measured_items, ());
        cursor.seek(&Count(measured_range.end), Bias::Right);
        new_items.append(cursor.suffix(), ());
        self.items = new_items;

        // If none of the visible items are focused, check if an off-screen item is focused
        // and include it to be rendered after the visible items so keyboard interaction continues
        // to work for it.
        if !rendered_focused_item {
            let mut cursor = self
                .items
                .filter::<_, Count>((), |summary| summary.has_focus_handles);
            cursor.next();
            while let Some(item) = cursor.item() {
                if item.contains_focused(window, cx) {
                    let item_index = cursor.start().0;
                    let mut element = render_item(cursor.start().0, window, cx);
                    let size = element.layout_as_root(available_item_space, window, cx);
                    item_layouts.push_back(ItemLayout {
                        index: item_index,
                        element,
                        size,
                        #[cfg(feature = "texture-cache")]
                        origin: Point::default(),
                        #[cfg(feature = "texture-cache")]
                        is_overdraw: false,
                    });
                    break;
                }
                cursor.next();
            }
        }

        // Re-engagement check: if in Tail mode but suspended, check if the user
        // has scrolled back to the bottom. If so, re-engage auto-following.
        let should_reengage = if matches!(self.follow_state, FollowState::Tail { is_following: false }) {
            let total_height = self.items.summary().height;
            let scroll_max =
                (total_height + padding.top + padding.bottom - available_height).max(px(0.));
            let current_pos = self.scroll_top(&scroll_top);
            scroll_max == px(0.) || (scroll_max - current_pos) < px(1.0)
        } else {
            false
        };
        if should_reengage {
            self.follow_state = FollowState::Tail { is_following: true };
        }

        // Emit layout perf telemetry when frame is slow (> 8ms).
        let layout_elapsed = layout_start.elapsed().as_secs_f32();
        if layout_elapsed > 0.008 {
            emit_layout_perf(&LayoutPerfTelemetry {
                phase: "layout",
                total_secs: layout_elapsed,
                rendered_count: perf_rendered_count,
                cached_count: perf_cached_count,
                item_count: self.items.summary().count,
                scroll_top_ix: scroll_top.item_ix,
                slowest_item_secs: perf_slowest_secs,
                slowest_item_ix: perf_slowest_ix,
            });
        }

        // S498: Layout phase timing — end measurement with render vs GPUI breakdown.
        #[cfg(feature = "texture-cache-debug")]
        {
            let elapsed_ms = layout_phase_start.elapsed().as_secs_f32() * 1000.0;
            let render_ms = render_sum_secs * 1000.0;
            let gpui_ms = elapsed_ms - render_ms;
            if elapsed_ms > 1.0 {
                log::info!(
                    "event=layout_phase_end duration_ms={:.1} items_rendered={} items_cached={} render_sum_ms={:.1} gpui_layout_ms={:.1}",
                    elapsed_ms, perf_rendered_count, perf_cached_count, render_ms, gpui_ms
                );
            }
            // S513 perf: feed the consolidated paint_timing_breakdown unconditionally
            // (the >1 ms gate above suppresses the verbose log line, but the accumulator
            // still wants the real number — even sub-1ms layouts matter for FPS analysis).
            crate::frame_perf_record_layout(self.paint_frame_count, elapsed_ms);
        }

        LayoutItemsResponse {
            max_item_width,
            scroll_top,
            item_layouts,
        }
    }

    fn prepaint_items(
        &mut self,
        bounds: Bounds<Pixels>,
        padding: Edges<Pixels>,
        autoscroll: bool,
        render_item: &mut RenderItemFn,
        window: &mut Window,
        cx: &mut App,
    ) -> Result<LayoutItemsResponse, ListOffset> {
        window.transact(|window| {
            match self.measuring_behavior {
                ListMeasuringBehavior::Measure(has_measured) if !has_measured => {
                    self.layout_all_items(bounds.size.width, render_item, window, cx);
                }
                _ => {}
            }

            let mut layout_response = self.layout_items(
                Some(bounds.size.width),
                bounds.size.height,
                &padding,
                render_item,
                window,
                cx,
            );

            // Avoid honoring autoscroll requests from elements other than our children.
            window.take_autoscroll();

            // Only paint the visible items, if there is actually any space for them (taking padding into account)
            if bounds.size.height > padding.top + padding.bottom {
                let mut item_origin = bounds.origin + Point::new(px(0.), padding.top);
                item_origin.y -= layout_response.scroll_top.offset_in_item;
                let prepaint_phase_start = Instant::now();
                let mut prepaint_slowest_secs: f32 = 0.0;
                let mut prepaint_slowest_ix: usize = 0;
                for item in &mut layout_response.item_layouts {
                    let prepaint_item_start = Instant::now();
                    window.with_content_mask(Some(ContentMask { bounds }), |window| {
                        item.element.prepaint_at(item_origin, window, cx);
                    });
                    let prepaint_item_elapsed = prepaint_item_start.elapsed().as_secs_f32();
                    if prepaint_item_elapsed > prepaint_slowest_secs {
                        prepaint_slowest_secs = prepaint_item_elapsed;
                        prepaint_slowest_ix = item.index;
                    }

                    if let Some(autoscroll_bounds) = window.take_autoscroll()
                        && autoscroll
                    {
                        if autoscroll_bounds.top() < bounds.top() {
                            return Err(ListOffset {
                                item_ix: item.index,
                                offset_in_item: autoscroll_bounds.top() - item_origin.y,
                            });
                        } else if autoscroll_bounds.bottom() > bounds.bottom() {
                            let mut cursor = self.items.cursor::<Count>(());
                            cursor.seek(&Count(item.index), Bias::Right);
                            let mut height = bounds.size.height - padding.top - padding.bottom;

                            // Account for the height of the element down until the autoscroll bottom.
                            height -= autoscroll_bounds.bottom() - item_origin.y;

                            // Keep decreasing the scroll top until we fill all the available space.
                            while height > Pixels::ZERO {
                                cursor.prev();
                                let Some(item) = cursor.item() else { break };

                                let size = item.size().unwrap_or_else(|| {
                                    let mut item = render_item(cursor.start().0, window, cx);
                                    let item_available_size =
                                        size(bounds.size.width.into(), AvailableSpace::MinContent);
                                    item.layout_as_root(item_available_size, window, cx)
                                });
                                height -= size.height;
                            }

                            return Err(ListOffset {
                                item_ix: cursor.start().0,
                                offset_in_item: if height < Pixels::ZERO {
                                    -height
                                } else {
                                    Pixels::ZERO
                                },
                            });
                        }
                    }

                    #[cfg(feature = "texture-cache")]
                    {
                        item.origin = item_origin;
                    }
                    item_origin.y += item.size.height;
                }

                // EC-6: Update per-item visibility frame counters.
                // Increment for items in this frame's layout, remove items no longer visible.
                // TODO(perf): HashSet alloc per frame (~20 items at 60fps). Replace with
                // generation-stamp approach if profiling shows GC pressure during scroll.
                #[cfg(feature = "texture-cache")]
                if self.caching_enabled {
                    let current_indices: std::collections::HashSet<usize> =
                        layout_response.item_layouts.iter().map(|i| i.index).collect();
                    self.visible_frames.retain(|ix, _| current_indices.contains(ix));
                    for ix in &current_indices {
                        let count = self.visible_frames.entry(*ix).or_insert(0);
                        *count = count.saturating_add(1);
                    }
                }

                // Emit prepaint perf telemetry when slow (> 8ms).
                let prepaint_elapsed = prepaint_phase_start.elapsed().as_secs_f32();
                if prepaint_elapsed > 0.008 {
                    emit_layout_perf(&LayoutPerfTelemetry {
                        phase: "prepaint",
                        total_secs: prepaint_elapsed,
                        rendered_count: layout_response.item_layouts.len(),
                        cached_count: 0,
                        item_count: self.items.summary().count,
                        scroll_top_ix: layout_response.scroll_top.item_ix,
                        slowest_item_secs: prepaint_slowest_secs,
                        slowest_item_ix: prepaint_slowest_ix,
                    });
                }
            } else {
                layout_response.item_layouts.clear();
            }

            Ok(layout_response)
        })
    }

    // Scrollbar support

    fn set_offset_from_scrollbar(&mut self, point: Point<Pixels>) {
        let Some(bounds) = self.last_layout_bounds else {
            return;
        };
        let height = bounds.size.height;

        let padding = self.last_padding.unwrap_or_default();
        // CS-S449+S450 vendor patch: sign convention + drag_offset fix
        //
        // S449: Replaced .abs() with -point.y (scrollbar always sends negative offsets).
        // S450: Removed drag_offset from new_scroll_top calculation. During drag,
        // both point.y (from scrollbar) and scroll_max use frozen content_height
        // (scrollbar_drag_start_height). The drag_offset (LIVE - FROZEN) was
        // double-compensating: as content grew during drag, drag_offset would
        // exceed -point.y, making the subtraction go negative and clamping to 0
        // (scroll jumps to top = inversion). Since both sides of the mapping are
        // frozen, no correction is needed.
        //
        // Callers: scrollbar passes NEGATIVE point.y; our view.rs callers
        // (follow_output, scroll_to_bottom) also pass NEGATIVE (negated max.height).
        //
        // Ref: scroll-interaction-analysis-cog.md Bug 1, gpui-scroll-math-deep-dive.md
        // Track: P2.9 upstream PR candidate
        // Phase C: Grow frozen height during drag if content has grown.
        // Keeps frozen → live mapping consistent with max_offset_for_scrollbar().
        let live_height = self.items.summary().height;
        let content_height = match self.scrollbar_drag_start_height {
            Some(frozen) if live_height > frozen => {
                self.scrollbar_drag_start_height = Some(live_height);
                live_height
            }
            Some(frozen) => frozen,
            None => live_height,
        };
        let scroll_max = (content_height + padding.top + padding.bottom - height).max(px(0.));
        let new_scroll_top = (-point.y).max(px(0.)).min(scroll_max);

        if self.alignment == ListAlignment::Bottom && new_scroll_top == scroll_max {
            self.logical_scroll_top = None;
        } else {
            let (start, _, _) =
                self.items
                    .find::<ListItemSummary, _>((), &Height(new_scroll_top), Bias::Right);

            let item_ix = start.count;
            let offset_in_item = new_scroll_top - start.height;
            self.logical_scroll_top = Some(ListOffset {
                item_ix,
                offset_in_item,
            });
        }

        // Scrollbar drag exits tail mode entirely
        self.follow_state = FollowState::Normal;
        self.tail_scroll_velocity = 0.0;
    }
}

impl std::fmt::Debug for ListItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unmeasured { .. } => write!(f, "Unrendered"),
            Self::Measured { size, .. } => f.debug_struct("Rendered").field("size", size).finish(),
        }
    }
}

/// An offset into the list's items, in terms of the item index and the number
/// of pixels off the top left of the item.
#[derive(Debug, Clone, Copy, Default)]
pub struct ListOffset {
    /// The index of an item in the list
    pub item_ix: usize,
    /// The number of pixels to offset from the item index.
    pub offset_in_item: Pixels,
}

impl Element for List {
    type RequestLayoutState = ();
    type PrepaintState = ListPrepaintState;

    fn id(&self) -> Option<crate::ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (crate::LayoutId, Self::RequestLayoutState) {
        let layout_id = match self.sizing_behavior {
            ListSizingBehavior::Infer => {
                let mut style = Style::default();
                style.overflow.y = Overflow::Scroll;
                style.refine(&self.style);
                window.with_text_style(style.text_style().cloned(), |window| {
                    let state = &mut *self.state.0.borrow_mut();

                    let available_height = if let Some(last_bounds) = state.last_layout_bounds {
                        last_bounds.size.height
                    } else {
                        // If we don't have the last layout bounds (first render),
                        // we might just use the overdraw value as the available height to layout enough items.
                        state.overdraw
                    };
                    let padding = style.padding.to_pixels(
                        state.last_layout_bounds.unwrap_or_default().size.into(),
                        window.rem_size(),
                    );

                    let layout_response = state.layout_items(
                        None,
                        available_height,
                        &padding,
                        &mut self.render_item,
                        window,
                        cx,
                    );
                    let max_element_width = layout_response.max_item_width;

                    let summary = state.items.summary();
                    let total_height = summary.height;

                    window.request_measured_layout(
                        style,
                        move |known_dimensions, available_space, _window, _cx| {
                            let width =
                                known_dimensions
                                    .width
                                    .unwrap_or(match available_space.width {
                                        AvailableSpace::Definite(x) => x,
                                        AvailableSpace::MinContent | AvailableSpace::MaxContent => {
                                            max_element_width
                                        }
                                    });
                            let height = match available_space.height {
                                AvailableSpace::Definite(height) => total_height.min(height),
                                AvailableSpace::MinContent | AvailableSpace::MaxContent => {
                                    total_height
                                }
                            };
                            size(width, height)
                        },
                    )
                })
            }
            ListSizingBehavior::Auto => {
                let mut style = Style::default();
                style.refine(&self.style);
                window.with_text_style(style.text_style().cloned(), |window| {
                    window.request_layout(style, None, cx)
                })
            }
        };
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> ListPrepaintState {
        let state = &mut *self.state.0.borrow_mut();
        state.reset = false;

        let mut style = Style::default();
        style.refine(&self.style);

        let hitbox = window.insert_hitbox(bounds, HitboxBehavior::Normal);

        // If the width of the list has changed, invalidate all cached item heights
        if state
            .last_layout_bounds
            .is_none_or(|last_bounds| last_bounds.size.width != bounds.size.width)
        {
            // Save proportional scroll position before reset so layout_items()
            // can restore it after items are re-measured at the new width.
            // Mirrors the same logic in remeasure().
            if let Some(scroll_top) = state.logical_scroll_top {
                let mut cursor = state.items.cursor::<Count>(());
                cursor.seek(&Count(scroll_top.item_ix), Bias::Right);

                if let Some(item) = cursor.item() {
                    if let Some(size) = item.size() {
                        let fraction = if size.height.0 > 0.0 {
                            (scroll_top.offset_in_item.0 / size.height.0).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        state.pending_scroll = Some(PendingScrollFraction {
                            item_ix: scroll_top.item_ix,
                            fraction,
                        });
                    }
                }
            }

            let new_items = SumTree::from_iter(
                state.items.iter().map(|item| ListItem::Unmeasured {
                    focus_handle: item.focus_handle(),
                    // Preserve last known size as hint during width change — prevents
                    // SumTree total height from collapsing to 0.
                    size_hint: item.size_hint(),
                }),
                (),
            );

            state.items = new_items;
            state.measuring_behavior.reset();
        }

        let padding = style
            .padding
            .to_pixels(bounds.size.into(), window.rem_size());
        let layout =
            match state.prepaint_items(bounds, padding, true, &mut self.render_item, window, cx) {
                Ok(layout) => layout,
                Err(autoscroll_request) => {
                    state.logical_scroll_top = Some(autoscroll_request);
                    state
                        .prepaint_items(bounds, padding, false, &mut self.render_item, window, cx)
                        .unwrap()
                }
            };

        state.last_layout_bounds = Some(bounds);
        state.last_padding = Some(padding);
        ListPrepaintState { hitbox, layout }
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<crate::Pixels>,
        _: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        #[cfg(feature = "texture-cache")]
        let paint_start = std::time::Instant::now();
        let current_view = window.current_view();
        #[cfg(feature = "texture-cache")]
        let caching_enabled;
        #[cfg(feature = "texture-cache")]
        let cache_clear_color;
        #[cfg(feature = "texture-cache")]
        let visible_frames_snapshot;
        #[cfg(feature = "texture-cache")]
        let streaming_items_snapshot;
        #[cfg(feature = "texture-cache")]
        let trace_items_snapshot;
        #[cfg(feature = "texture-cache")]
        let prev_item_heights_snapshot;
        #[cfg(feature = "texture-cache")]
        let frame_count;
        #[cfg(feature = "texture-cache")]
        let cached_item_state_keys_snapshot;
        #[cfg(feature = "texture-cache")]
        let fade_alpha;
        #[cfg(feature = "texture-cache")]
        {
            let mut state = self.state.0.borrow_mut();
            caching_enabled = state.caching_enabled;
            cache_clear_color = state.cache_clear_color;
            visible_frames_snapshot = state.visible_frames.clone();
            streaming_items_snapshot = state.streaming_items.clone();
            trace_items_snapshot = state.trace_items.clone();
            prev_item_heights_snapshot = state.prev_item_heights.clone();
            cached_item_state_keys_snapshot = state.cached_item_state_keys.clone();
            fade_alpha = state.fade_alpha;
            state.paint_frame_count += 1;
            frame_count = state.paint_frame_count;
            crate::card_timeline::bump_frame();
        }
        // S502: publish the controller-driven fade alpha into the Scene so
        // the composite shader applies it via `globals.composite_fade_alpha`.
        // Reset to 1.0 at frame start by Scene::clear, so writing only when
        // < 1.0 would leave stale 1.0 frames untouched — but we write
        // unconditionally because overwriting with 1.0 is a no-op and keeps
        // this hunk single-line.
        #[cfg(feature = "texture-cache")]
        {
            window.next_frame.scene.composite_fade_alpha = fade_alpha;
        }
        #[cfg(feature = "texture-cache")]
        let mut current_frame_heights: HashMap<usize, Pixels> = HashMap::new();
        // S500: start from prior frame's captured keys so HIT sees keys from the
        // MISS that originally cached the item, even if this frame's MISS set is
        // empty. MISS/TRANSIENT overwrite their own entries below; entries for
        // items no longer painted this frame drop out naturally (see §3.3 of
        // S500-sage-element-state-gc-fix.md).
        //
        // Note: unlike `current_frame_heights` (empty at loop start), this
        // starts populated — pure-HIT items don't pass through the
        // MISS/TRANSIENT capture blocks, so they must inherit prior keys here
        // rather than re-capture them each frame.
        #[cfg(feature = "texture-cache")]
        let mut current_item_state_keys: HashMap<usize, Vec<(GlobalElementId, TypeId)>> =
            if caching_enabled {
                cached_item_state_keys_snapshot
            } else {
                HashMap::new()
            };
        // When caching is active, bypass content_mask entirely — textures have their
        // own bounds and the list-level mask was culling edge-item primitives during
        // texture capture (bounds ∩ content_mask = empty).
        // -- Visibility classification for texture cache eviction --
        // Collected here (before paint loop) so the renderer gets current-frame data.
        // Empty sets are valid — renderer will skip classify_entries that frame.
        #[cfg(feature = "texture-cache")]
        if caching_enabled {
            let mut visible_ids = HashSet::new();
            let mut buffer_ids = HashSet::new();
            let mut visible_min_ix = usize::MAX;
            let mut visible_max_ix = 0usize;
            for item in &prepaint.layout.item_layouts {
                let region_id = item.index as u64;
                if item.is_overdraw {
                    buffer_ids.insert(region_id);
                } else {
                    visible_ids.insert(region_id);
                    visible_min_ix = visible_min_ix.min(item.index);
                    visible_max_ix = visible_max_ix.max(item.index);
                }
            }
            // Viewport center index for distance-based eviction ordering.
            // Only non-overdraw items define "center of viewport".
            let has_visible_items = visible_min_ix <= visible_max_ix;
            if has_visible_items {
                let center = (visible_min_ix + visible_max_ix) / 2;
                crate::set_viewport_center_index(center);
            }
            crate::set_classification_ids(visible_ids, buffer_ids);
        }

        #[cfg(feature = "texture-cache")]
        let content_mask = if caching_enabled {
            None  // No clipping during texture capture — textures have their own bounds
        } else {
            Some(ContentMask { bounds })
        };
        #[cfg(not(feature = "texture-cache"))]
        let content_mask = Some(ContentMask { bounds });
        #[cfg(feature = "texture-cache")]
        let mut diag_hit: u32 = 0;
        #[cfg(feature = "texture-cache")]
        let mut diag_miss: u32 = 0;
        #[cfg(feature = "texture-cache")]
        let mut diag_streaming: u32 = 0;
        #[cfg(feature = "texture-cache")]
        let mut diag_transient: u32 = 0;
        #[cfg(feature = "texture-cache")]
        let mut diag_plain: u32 = 0;
        window.with_content_mask(content_mask, |window| {
            for item in &mut prepaint.layout.item_layouts {
                #[cfg(feature = "texture-cache")]
                if caching_enabled {
                    let is_traced = trace_items_snapshot.contains(&item.index);

                    let is_timeline_watched = crate::card_timeline::is_watched(item.index);

                    // FR-4: Streaming items render Fresh every frame — content is
                    // still changing, so any cached texture would be immediately stale.
                    if streaming_items_snapshot.contains(&item.index) {
                        if is_traced {
                            log::info!("event=trace_item ix={} path=streaming_skip bounds={:.0},{:.0},{:.0},{:.0}",
                                item.index, f32::from(item.origin.x), f32::from(item.origin.y),
                                f32::from(item.size.width), f32::from(item.size.height));
                        }
                        if is_timeline_watched {
                            crate::card_timeline::log_event(&format!(
                                "[list_paint] item={} status=STREAMING in_rendered_range=true caching_enabled=true size={:.0}x{:.0}",
                                item.index, f32::from(item.size.width), f32::from(item.size.height)));
                        }
                        // Height change detection: STREAMING path
                        if let Some(&prev_h) = prev_item_heights_snapshot.get(&item.index) {
                            if prev_h != item.size.height {
                                emit_height_change(&HeightChangeEvent {
                                    item_index: item.index,
                                    old_height: f32::from(prev_h),
                                    new_height: f32::from(item.size.height),
                                    cache_path: "STREAMING",
                                    frame_count,
                                });
                            }
                        }
                        current_frame_heights.insert(item.index, item.size.height);
                        diag_streaming += 1;
                        item.element.paint(window, cx);
                        continue;
                    }

                    let region_id = CacheRegionId(item.index as u64);
                    let item_bounds = Bounds {
                        origin: item.origin,
                        size: item.size,
                    };

                    // EC-6: Skip texture creation for transient items (visible < 2 frames).
                    // Items already cached still get composited (cache HIT path).
                    let frames_visible = visible_frames_snapshot
                        .get(&item.index)
                        .copied()
                        .unwrap_or(0);
                    if crate::is_transient_skip_enabled() && frames_visible < TRANSIENT_SKIP_FRAMES && !has_cached_region(region_id) {
                        if is_traced {
                            log::info!("event=trace_item ix={} path=transient_skip frames_visible={} bounds={:.0},{:.0},{:.0},{:.0}",
                                item.index, frames_visible, f32::from(item.origin.x), f32::from(item.origin.y),
                                f32::from(item.size.width), f32::from(item.size.height));
                        }
                        if is_timeline_watched {
                            crate::card_timeline::log_event(&format!(
                                "[list_paint] item={} status=TRANSIENT in_rendered_range=true caching_enabled=true frames_visible={} size={:.0}x{:.0}",
                                item.index, frames_visible, f32::from(item.size.width), f32::from(item.size.height)));
                        }
                        // Height change detection: TRANSIENT path
                        if let Some(&prev_h) = prev_item_heights_snapshot.get(&item.index) {
                            if prev_h != item.size.height {
                                emit_height_change(&HeightChangeEvent {
                                    item_index: item.index,
                                    old_height: f32::from(prev_h),
                                    new_height: f32::from(item.size.height),
                                    cache_path: "TRANSIENT",
                                    frame_count,
                                });
                            }
                        }
                        current_frame_heights.insert(item.index, item.size.height);
                        // Transient item — render Fresh without texture annotation.
                        diag_transient += 1;
                        // S500: capture element state keys added by this paint so
                        // subsequent HIT frames can re-extend them and prevent GC.
                        // Transient items can later flip to HIT once the
                        // frames_visible threshold is crossed and a cache region
                        // exists.
                        let state_keys_start =
                            window.next_frame.accessed_element_states.len();
                        item.element.paint(window, cx);
                        let state_keys_end =
                            window.next_frame.accessed_element_states.len();
                        // Guard: an empty-capture paint (degenerate or
                        // structurally-changed card) keeps the prior frame's
                        // capture rather than clearing it — a structural change
                        // that zeroes out element state also changes height, so
                        // invalidate_item_cache clears the entry on that path
                        // (S500 §3.1).
                        if state_keys_end > state_keys_start {
                            let keys: Vec<(GlobalElementId, TypeId)> = window
                                .next_frame
                                .accessed_element_states[state_keys_start..state_keys_end]
                                .to_vec();
                            current_item_state_keys.insert(item.index, keys);
                        }
                        continue;
                    }

                    // Determine cache path and detect height changes
                    let cache_path = if has_cached_region(region_id) { "HIT" } else { "MISS" };
                    if let Some(&prev_h) = prev_item_heights_snapshot.get(&item.index) {
                        if prev_h != item.size.height {
                            emit_height_change(&HeightChangeEvent {
                                item_index: item.index,
                                old_height: f32::from(prev_h),
                                new_height: f32::from(item.size.height),
                                cache_path,
                                frame_count,
                            });
                        }
                    }
                    current_frame_heights.insert(item.index, item.size.height);

                    if has_cached_region(region_id) {
                        if is_traced {
                            log::info!("event=trace_item ix={} path=cache_hit bounds={:.0},{:.0},{:.0},{:.0} clear_color=h{:.3},s{:.3},l{:.3},a{:.1}",
                                item.index, f32::from(item.origin.x), f32::from(item.origin.y),
                                f32::from(item.size.width), f32::from(item.size.height),
                                cache_clear_color.h, cache_clear_color.s, cache_clear_color.l, cache_clear_color.a);
                        }
                        if is_timeline_watched {
                            crate::card_timeline::log_event(&format!(
                                "[list_paint] item={} status=HIT in_rendered_range=true caching_enabled=true size={:.0}x{:.0}",
                                item.index, f32::from(item.size.width), f32::from(item.size.height)));
                        }
                        // Cache HIT — annotate empty region, skip paint.
                        // Renderer composites from cached texture.
                        diag_hit += 1;
                        // S500: re-extend accessed_element_states with keys
                        // captured at the MISS that originally cached this
                        // item, so Frame::finish's GC preserves its
                        // TextViewState (and other element state) across this
                        // HIT frame. Without this, an item spending >1 frame
                        // on HIT loses its state — the first DIRECT paint
                        // after transition re-creates it empty and shows a
                        // shell-only card until the async parse task lands.
                        if let Some(keys) = current_item_state_keys.get(&item.index) {
                            window.keep_element_states_alive(keys);
                        }
                        window.begin_cache_region(region_id, item_bounds, cache_clear_color, bounds);
                        window.end_cache_region(region_id);
                        continue;
                    }

                    // Cache MISS — paint normally, annotate for texture capture.
                    if is_traced {
                        log::info!("event=trace_item ix={} path=cache_miss bounds={:.0},{:.0},{:.0},{:.0} clear_color=h{:.3},s{:.3},l{:.3},a{:.1}",
                            item.index, f32::from(item.origin.x), f32::from(item.origin.y),
                            f32::from(item.size.width), f32::from(item.size.height),
                            cache_clear_color.h, cache_clear_color.s, cache_clear_color.l, cache_clear_color.a);
                    }
                    if is_timeline_watched {
                        crate::card_timeline::log_event(&format!(
                            "[list_paint] item={} status=MISS in_rendered_range=true caching_enabled=true size={:.0}x{:.0}",
                            item.index, f32::from(item.size.width), f32::from(item.size.height)));
                    }
                    // Fix: Replace content_mask_stack with a full-card mask during
                    // MISS capture. Ancestor masks (e.g. viewport clip) would cause
                    // paint_line() to skip glyphs outside the viewport — but the
                    // texture needs ALL card content, including off-screen portions.
                    // item_bounds is the full card area, not intersected with viewport.
                    // The renderer's viewport_clip parameter to begin_cache_region
                    // handles display-boundary culling separately.
                    // NOTE: If element.paint() panics, content_mask_stack is left
                    // corrupted — acceptable because GPUI panics are app-fatal.
                    let ancestor_masks = std::mem::take(&mut window.content_mask_stack);
                    window.content_mask_stack.push(ContentMask {
                        bounds: item_bounds,
                    });

                    window.begin_cache_region(region_id, item_bounds, cache_clear_color, bounds);
                    let ops_before = window.next_frame.scene.len();
                    if is_timeline_watched {
                        crate::card_timeline::begin_clip_drop_count();
                    }
                    // S500: capture element state keys added by this MISS paint
                    // so subsequent HIT frames can re-extend them and prevent
                    // GC of TextViewState etc.
                    let state_keys_start = window.next_frame.accessed_element_states.len();
                    item.element.paint(window, cx);
                    let state_keys_end = window.next_frame.accessed_element_states.len();
                    let ops_after = window.next_frame.scene.len();
                    // Guard: see TRANSIENT-path comment above — empty capture
                    // preserves the prior frame's keys. Structural changes that
                    // zero out element state also change height and go through
                    // invalidate_item_cache, which clears the entry.
                    if state_keys_end > state_keys_start {
                        let keys: Vec<(GlobalElementId, TypeId)> = window
                            .next_frame
                            .accessed_element_states[state_keys_start..state_keys_end]
                            .to_vec();
                        current_item_state_keys.insert(item.index, keys);
                    }
                    if is_timeline_watched {
                        let clipped = crate::card_timeline::end_clip_drop_count();
                        crate::card_timeline::log_event(&format!(
                            "[miss_paint] item={} ops_added={} ops_before={} ops_after={} clip_dropped={}",
                            item.index, ops_after - ops_before, ops_before, ops_after, clipped,
                        ));
                    }
                    window.end_cache_region(region_id);

                    // Restore after end_cache_region: scene finalization
                    // does not read content_mask_stack.
                    window.content_mask_stack = ancestor_masks;
                    diag_miss += 1;
                    continue;
                }

                // All caching_enabled=true branches above use `continue`.
                // This line is reached only when caching_enabled=false.

                // S502 FADE-OVERLAY: during the post-stop 100ms fade window,
                // caching is disabled but cached textures from the just-ended
                // scroll are still valid (renderer feedback set un-purged).
                // Paint DIRECT, then emit an empty cache-region bracket so the
                // composite shader draws the old texture on top at the current
                // fade alpha. As alpha decays to 0.0, the texture fades out,
                // revealing the fresh DIRECT paint underneath.
                //
                // Gated on `fade_alpha < 1.0` so non-fading DIRECT frames stay
                // on the single-line `element.paint` path below (no extra
                // bracket, no composite-shader work).
                #[cfg(feature = "texture-cache")]
                if fade_alpha < 1.0 {
                    let region_id = CacheRegionId(item.index as u64);
                    // If the cached texture was evicted mid-fade,
                    // has_cached_region returns false: skip the bracket and
                    // fall through to a plain DIRECT paint below. A
                    // DIRECT-only render is always correct; the overlay is
                    // additive improvement only.
                    if has_cached_region(region_id) {
                        let item_bounds = Bounds {
                            origin: item.origin,
                            size: item.size,
                        };
                        crate::elements::list_fade::emit_fade_overlay(
                            window,
                            cx,
                            &mut item.element,
                            region_id,
                            item_bounds,
                            bounds,
                            cache_clear_color,
                            fade_alpha,
                        );
                        continue;
                    }
                }

                #[cfg(feature = "texture-cache")]
                { diag_plain += 1; }
                item.element.paint(window, cx);
            }
        });
        #[cfg(feature = "texture-cache")]
        log::info!(
            "event=frame_cache_summary paint_frame={} caching={} hit={} miss={} streaming={} transient={} plain={}",
            frame_count, caching_enabled, diag_hit, diag_miss, diag_streaming, diag_transient, diag_plain
        );
        // S513 perf: feed the consolidated paint_timing_breakdown emitted at frame end.
        // When caching is OFF, hit/miss are 0 and plain reflects every painted item —
        // exactly the OFF baseline Klaus needs for the ON vs OFF FPS comparison.
        #[cfg(feature = "texture-cache")]
        crate::frame_perf_record_counts(frame_count, diag_hit, diag_miss, diag_plain);
        // Update prev_item_heights and cached_item_state_keys for next frame.
        // Single borrow_mut for both write-backs.
        #[cfg(feature = "texture-cache")]
        if caching_enabled {
            let mut inner = self.state.0.borrow_mut();
            inner.prev_item_heights = current_frame_heights;
            inner.cached_item_state_keys = current_item_state_keys;
        }

        let list_state = self.state.clone();
        let height = bounds.size.height;
        let scroll_top = prepaint.layout.scroll_top;
        let hitbox_id = prepaint.hitbox.id;
        let mut accumulated_scroll_delta = ScrollDelta::default();
        window.on_mouse_event(move |event: &ScrollWheelEvent, phase, window, cx| {
            if list_state.0.borrow().suppress_wheel_scroll {
                return;
            }
            if phase == DispatchPhase::Bubble && hitbox_id.should_handle_scroll(window) {
                accumulated_scroll_delta = accumulated_scroll_delta.coalesce(event.delta);
                let pixel_delta = accumulated_scroll_delta.pixel_delta(px(20.));
                list_state.0.borrow_mut().scroll(
                    &scroll_top,
                    height,
                    pixel_delta,
                    current_view,
                    window,
                    cx,
                )
            }
        });
        #[cfg(feature = "texture-cache")]
        {
            let paint_dur = paint_start.elapsed();
            if paint_dur.as_micros() > 4000 {
                log::info!(
                    "event=paint_timing paint_frame={} dur_us={} items={}",
                    frame_count,
                    paint_dur.as_micros(),
                    prepaint.layout.item_layouts.len()
                );
            }
        }
    }
}

impl IntoElement for List {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Styled for List {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

impl sum_tree::Item for ListItem {
    type Summary = ListItemSummary;

    fn summary(&self, _: ()) -> Self::Summary {
        match self {
            ListItem::Unmeasured { focus_handle, size_hint } => ListItemSummary {
                count: 1,
                rendered_count: 0,
                unrendered_count: 1,
                height: size_hint.map_or(px(0.), |s| s.height),
                has_focus_handles: focus_handle.is_some(),
            },
            ListItem::Measured {
                size, focus_handle, ..
            } => ListItemSummary {
                count: 1,
                rendered_count: 1,
                unrendered_count: 0,
                height: size.height,
                has_focus_handles: focus_handle.is_some(),
            },
        }
    }
}

impl sum_tree::ContextLessSummary for ListItemSummary {
    fn zero() -> Self {
        Default::default()
    }

    fn add_summary(&mut self, summary: &Self) {
        self.count += summary.count;
        self.rendered_count += summary.rendered_count;
        self.unrendered_count += summary.unrendered_count;
        self.height += summary.height;
        self.has_focus_handles |= summary.has_focus_handles;
    }
}

impl<'a> sum_tree::Dimension<'a, ListItemSummary> for Count {
    fn zero(_cx: ()) -> Self {
        Default::default()
    }

    fn add_summary(&mut self, summary: &'a ListItemSummary, _: ()) {
        self.0 += summary.count;
    }
}

impl<'a> sum_tree::Dimension<'a, ListItemSummary> for Height {
    fn zero(_cx: ()) -> Self {
        Default::default()
    }

    fn add_summary(&mut self, summary: &'a ListItemSummary, _: ()) {
        self.0 += summary.height;
    }
}

impl sum_tree::SeekTarget<'_, ListItemSummary, ListItemSummary> for Count {
    fn cmp(&self, other: &ListItemSummary, _: ()) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.count).unwrap()
    }
}

impl sum_tree::SeekTarget<'_, ListItemSummary, ListItemSummary> for Height {
    fn cmp(&self, other: &ListItemSummary, _: ()) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.height).unwrap()
    }
}

#[cfg(test)]
mod test {

    use gpui::{ScrollDelta, ScrollWheelEvent};
    use std::cell::Cell;
    use std::rc::Rc;

    use crate::{
        self as gpui, AppContext, Context, Element, IntoElement, ListState, Render, Styled,
        TestAppContext, Window, div, list, point, px, size,
    };

    #[gpui::test]
    fn test_reset_after_paint_before_scroll(cx: &mut TestAppContext) {
        let cx = cx.add_empty_window();

        let state = ListState::new(5, crate::ListAlignment::Top, px(10.));

        // Ensure that the list is scrolled to the top
        state.scroll_to(gpui::ListOffset {
            item_ix: 0,
            offset_in_item: px(0.0),
        });

        struct TestView(ListState);
        impl Render for TestView {
            fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
                list(self.0.clone(), |_, _, _| {
                    div().h(px(10.)).w_full().into_any()
                })
                .w_full()
                .h_full()
            }
        }

        // Paint
        cx.draw(point(px(0.), px(0.)), size(px(100.), px(20.)), |_, cx| {
            cx.new(|_| TestView(state.clone())).into_any_element()
        });

        // Reset
        state.reset(5);

        // And then receive a scroll event _before_ the next paint
        cx.simulate_event(ScrollWheelEvent {
            position: point(px(1.), px(1.)),
            delta: ScrollDelta::Pixels(point(px(0.), px(-500.))),
            ..Default::default()
        });

        // Scroll position should stay at the top of the list
        assert_eq!(state.logical_scroll_top().item_ix, 0);
        assert_eq!(state.logical_scroll_top().offset_in_item, px(0.));
    }

    #[gpui::test]
    fn test_scroll_by_positive_and_negative_distance(cx: &mut TestAppContext) {
        let cx = cx.add_empty_window();

        let state = ListState::new(5, crate::ListAlignment::Top, px(10.));

        struct TestView(ListState);
        impl Render for TestView {
            fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
                list(self.0.clone(), |_, _, _| {
                    div().h(px(20.)).w_full().into_any()
                })
                .w_full()
                .h_full()
            }
        }

        // Paint
        cx.draw(point(px(0.), px(0.)), size(px(100.), px(100.)), |_, cx| {
            cx.new(|_| TestView(state.clone())).into_any_element()
        });

        // Test positive distance: start at item 1, move down 30px
        state.scroll_by(px(30.));

        // Should move to item 2
        let offset = state.logical_scroll_top();
        assert_eq!(offset.item_ix, 1);
        assert_eq!(offset.offset_in_item, px(10.));

        // Test negative distance: start at item 2, move up 30px
        state.scroll_by(px(-30.));

        // Should move back to item 1
        let offset = state.logical_scroll_top();
        assert_eq!(offset.item_ix, 0);
        assert_eq!(offset.offset_in_item, px(0.));

        // Test zero distance
        state.scroll_by(px(0.));
        let offset = state.logical_scroll_top();
        assert_eq!(offset.item_ix, 0);
        assert_eq!(offset.offset_in_item, px(0.));
    }

    #[gpui::test]
    fn test_measure_all_after_width_change(cx: &mut TestAppContext) {
        let cx = cx.add_empty_window();

        let state = ListState::new(10, crate::ListAlignment::Top, px(0.)).measure_all();

        struct TestView(ListState);
        impl Render for TestView {
            fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
                list(self.0.clone(), |_, _, _| {
                    div().h(px(50.)).w_full().into_any()
                })
                .w_full()
                .h_full()
            }
        }

        let view = cx.update(|_, cx| cx.new(|_| TestView(state.clone())));

        // First draw at width 100: all 10 items measured (total 500px).
        // Viewport is 200px, so max scroll offset should be 300px.
        cx.draw(point(px(0.), px(0.)), size(px(100.), px(200.)), |_, _| {
            view.clone().into_any_element()
        });
        assert_eq!(state.max_offset_for_scrollbar().y, px(300.));

        // Second draw at a different width: items get invalidated.
        // Without the fix, max_offset would drop because unmeasured items
        // contribute 0 height.
        cx.draw(point(px(0.), px(0.)), size(px(200.), px(200.)), |_, _| {
            view.into_any_element()
        });
        assert_eq!(state.max_offset_for_scrollbar().y, px(300.));
    }

    #[gpui::test]
    fn test_remeasure(cx: &mut TestAppContext) {
        let cx = cx.add_empty_window();

        // Create a list with 10 items, each 100px tall. We'll keep a reference
        // to the item height so we can later change the height and assert how
        // `ListState` handles it.
        let item_height = Rc::new(Cell::new(100usize));
        let state = ListState::new(10, crate::ListAlignment::Top, px(10.));

        struct TestView {
            state: ListState,
            item_height: Rc<Cell<usize>>,
        }

        impl Render for TestView {
            fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
                let height = self.item_height.get();
                list(self.state.clone(), move |_, _, _| {
                    div().h(px(height as f32)).w_full().into_any()
                })
                .w_full()
                .h_full()
            }
        }

        let state_clone = state.clone();
        let item_height_clone = item_height.clone();
        let view = cx.update(|_, cx| {
            cx.new(|_| TestView {
                state: state_clone,
                item_height: item_height_clone,
            })
        });

        // Simulate scrolling 40px inside the element with index 2. Since the
        // original item height is 100px, this equates to 40% inside the item.
        state.scroll_to(gpui::ListOffset {
            item_ix: 2,
            offset_in_item: px(40.),
        });

        cx.draw(point(px(0.), px(0.)), size(px(100.), px(200.)), |_, _| {
            view.clone().into_any_element()
        });

        let offset = state.logical_scroll_top();
        assert_eq!(offset.item_ix, 2);
        assert_eq!(offset.offset_in_item, px(40.));

        // Update the `item_height` to be 50px instead of 100px so we can assert
        // that the scroll position is proportionally preserved, that is,
        // instead of 40px from the top of item 2, it should be 20px, since the
        // item's height has been halved.
        item_height.set(50);
        state.remeasure();

        cx.draw(point(px(0.), px(0.)), size(px(100.), px(200.)), |_, _| {
            view.into_any_element()
        });

        let offset = state.logical_scroll_top();
        assert_eq!(offset.item_ix, 2);
        assert_eq!(offset.offset_in_item, px(20.));
    }
}
