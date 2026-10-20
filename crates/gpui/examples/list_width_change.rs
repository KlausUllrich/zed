//! Shows what happens to the scroll position of a `List` when its width
//! changes while the list is scrolled into the middle of a tall item.
//!
//! Every item is a paragraph of wrapped text, so its height depends on the
//! list width. The example scrolls halfway into message 10, then shows and
//! hides a sidebar every two seconds. The header prints the scroll anchor:
//! with the anchor preserved, the same sentence stays at the top of the list.

#![cfg_attr(target_family = "wasm", no_main)]

#[path = "example_support/fonts.rs"]
mod example_support;

use std::time::Duration;

use gpui::{
    App, Bounds, Context, ListAlignment, ListOffset, ListState, Render, Window, WindowBounds,
    WindowOptions, div, list, prelude::*, px, rgb, size,
};
use gpui_platform::application;

const ITEM_COUNT: usize = 30;
const SENTENCES_PER_ITEM: usize = 24;
const SIDEBAR_WIDTH: f32 = 360.;
const ANCHOR_ITEM: usize = 10;
const ANCHOR_OFFSET: f32 = 240.;
const TOGGLES: usize = 4;

struct ListWidthChange {
    list_state: ListState,
    sidebar_open: bool,
}

impl ListWidthChange {
    fn new(cx: &mut Context<Self>) -> Self {
        let list_state = ListState::new(ITEM_COUNT, ListAlignment::Top, px(1000.));
        list_state.scroll_to(ListOffset {
            item_ix: ANCHOR_ITEM,
            offset_in_item: px(ANCHOR_OFFSET),
        });

        cx.spawn(async move |this, cx| {
            for _ in 0..TOGGLES {
                cx.background_executor().timer(Duration::from_secs(2)).await;
                let _ = this.update(cx, |this, cx| {
                    this.sidebar_open = !this.sidebar_open;
                    cx.notify();
                });
                // Redraw once more so the header shows the anchor after layout.
                cx.background_executor()
                    .timer(Duration::from_millis(100))
                    .await;
                let _ = this.update(cx, |_, cx| cx.notify());
            }
            cx.background_executor().timer(Duration::from_secs(2)).await;
            let _ = cx.update(|cx| cx.quit());
        })
        .detach();

        Self {
            list_state,
            sidebar_open: false,
        }
    }
}

fn sentence(item: usize, n: usize) -> String {
    format!("[{item}.{n}] The quick brown fox jumps over the lazy dog near the river bank. ")
}

impl Render for ListWidthChange {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let top = self.list_state.logical_scroll_top();
        let item_height = self
            .list_state
            .bounds_for_item(top.item_ix)
            .map(|bounds| bounds.size.height);
        let percent = item_height
            .filter(|height| *height > px(0.))
            .map(|height| {
                format!(
                    "{:.0}%",
                    100. * top.offset_in_item.as_f32() / height.as_f32()
                )
            })
            .unwrap_or_else(|| "?".into());
        let list_width = self.list_state.viewport_bounds().size.width;
        let label = std::env::var("LIST_DEMO_LABEL").unwrap_or_default();

        div()
            .size_full()
            .bg(rgb(0xFFFFFF))
            .flex()
            .flex_col()
            .child(
                div()
                    .p_2()
                    .bg(rgb(0x202020))
                    .text_color(rgb(0xFFFFFF))
                    .text_sm()
                    .child(format!(
                        "{label}  list width {:.0}px | scroll top: message {}, {:.0}px in ({percent} of {:.0}px)",
                        list_width.as_f32(),
                        top.item_ix,
                        top.offset_in_item.as_f32(),
                        item_height.map_or(0., |height| height.as_f32()),
                    )),
            )
            .child(
                div()
                    .flex_1()
                    .flex()
                    .flex_row()
                    .overflow_hidden()
                    .child(
                        list(self.list_state.clone(), |index, _window, _cx| {
                            let text: String =
                                (1..=SENTENCES_PER_ITEM).map(|n| sentence(index, n)).collect();
                            div()
                                .w_full()
                                .p_3()
                                .border_b_2()
                                .border_color(rgb(0x3366CC))
                                .bg(if index % 2 == 0 {
                                    rgb(0xF4F7FC)
                                } else {
                                    rgb(0xFFFFFF)
                                })
                                .child(div().font_weight(gpui::FontWeight::BOLD).child(format!(
                                    "Message {index}"
                                )))
                                .child(div().text_sm().child(text))
                                .into_any()
                        })
                        .flex_1()
                        .h_full(),
                    )
                    .when(self.sidebar_open, |this| {
                        this.child(
                            div()
                                .w(px(SIDEBAR_WIDTH))
                                .h_full()
                                .flex_shrink_0()
                                .bg(rgb(0xE8E8E8))
                                .border_l_1()
                                .border_color(rgb(0xBBBBBB))
                                .p_3()
                                .child("Sidebar"),
                        )
                    }),
            )
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        if !example_support::load_fonts(cx) {
            return;
        }
        let bounds = Bounds::centered(None, size(px(900.), px(600.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(ListWidthChange::new),
        )
        .unwrap();
        cx.activate(true);
    });
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    run_example();
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    gpui_platform::web_init();
    run_example();
}
