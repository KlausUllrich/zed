//! GPU texture caching for list item scroll compositing.
//!
//! Renders list items to offscreen GPU textures and composites them as quads
//! during scroll. Textures persist across frames and are reused until invalidated.

use super::*;
use gpui::Hsla;
use std::collections::HashMap;

/// A cached offscreen texture for a list item.
pub(crate) struct CachedItemTexture {
    #[allow(dead_code)]
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

/// Simple texture pool for Phase A. Stores per-item offscreen textures
/// keyed by CacheRegionId. Textures persist across frames until invalidated.
/// Phase B adds size-class bucketing and LRU eviction.
pub(crate) struct TexturePool {
    pub(crate) textures: HashMap<u64, CachedItemTexture>,
    /// Uniform buffer for per-item viewport globals (different size per item).
    item_globals_buffer: wgpu::Buffer,
    /// Stride between entries in item_globals_buffer (alignment-padded).
    globals_entry_stride: u64,
}

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

impl WgpuRenderer {
    fn ensure_texture_pool(&mut self) {
        if self.texture_pool.is_some() {
            return;
        }
        let resources = self.resources();
        let alignment = resources.device.limits().min_uniform_buffer_offset_alignment as u64;
        let globals_size = std::mem::size_of::<GlobalParams>() as u64;
        let entry_stride = globals_size.next_multiple_of(alignment);
        let max_items: u64 = 16;
        let item_globals_buffer = resources.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("item_texture_globals"),
            size: entry_stride * max_items,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.texture_pool = Some(TexturePool {
            textures: HashMap::new(),
            item_globals_buffer,
            globals_entry_stride: entry_stride,
        });
    }

    fn create_item_texture(&self, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// Pre-pass: render dirty cache regions to offscreen textures.
    /// Textures persist across frames — only re-rendered on cache miss or size change.
    pub(crate) fn process_cache_regions(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &Scene,
        instance_offset: &mut u64,
    ) -> bool {
        let regions: Vec<_> = scene.cache_regions().to_vec();
        if regions.is_empty() {
            return true;
        }

        self.ensure_texture_pool();

        let globals_size = std::mem::size_of::<GlobalParams>() as u64;
        let max_renders_per_frame: usize = 16;
        let mut render_idx: usize = 0;

        for region in &regions {
            let tex_width = (region.bounds.size.width.0.ceil() as u32).max(1);
            let tex_height = (region.bounds.size.height.0.ceil() as u32).max(1);

            // Skip items too large for a single texture
            if tex_width > self.max_texture_size || tex_height > self.max_texture_size {
                continue;
            }

            // Cache hit: texture exists with matching dimensions — reuse
            let pool = self.texture_pool.as_ref().unwrap();
            if let Some(cached) = pool.textures.get(&region.id.0) {
                if cached.width == tex_width && cached.height == tex_height {
                    continue;
                }
            }

            // Cap renders per frame to globals buffer capacity
            if render_idx >= max_renders_per_frame {
                continue; // Remaining items render Fresh (no texture)
            }

            // Cache miss or dimension change — render to new texture
            let mini_scene = scene.extract_region_as_mini_scene(region);
            let (texture, view) = self.create_item_texture(tex_width, tex_height);

            // Write per-item viewport globals at a unique offset
            let pool = self.texture_pool.as_ref().unwrap();
            let entry_offset = (render_idx as u64) * pool.globals_entry_stride;
            let item_globals = GlobalParams {
                viewport_size: [tex_width as f32, tex_height as f32],
                premultiplied_alpha: 0,
                pad: 0,
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

            // Render mini-scene to offscreen texture
            let clear_color = hsla_to_wgpu_color(region.clear_color);
            if !self.render_mini_scene_to_texture(
                encoder,
                &mini_scene,
                &view,
                clear_color,
                &item_bind_group,
                instance_offset,
            ) {
                return false;
            }

            // Store in pool (replaces any stale entry)
            self.texture_pool.as_mut().unwrap().textures.insert(
                region.id.0,
                CachedItemTexture {
                    texture,
                    view,
                    width: tex_width,
                    height: tex_height,
                },
            );
            render_idx += 1;
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
                PrimitiveBatch::Paths(_range) => {
                    // Phase A: skip paths in item textures (uncommon in card content).
                    true
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
                    let resources = self.resources();
                    let pipeline = resources
                        .pipelines
                        .subpixel_sprites
                        .as_ref()
                        .unwrap_or(&resources.pipelines.mono_sprites);
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
    pub(crate) fn draw_cached_regions(
        &self,
        scene: &Scene,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let pool = match &self.texture_pool {
            Some(p) => p,
            None => return true,
        };

        for region in scene.cache_regions() {
            let cached = match pool.textures.get(&region.id.0) {
                Some(c) => c,
                None => continue,
            };

            let sprite = PathSprite {
                bounds: region.bounds,
            };
            let sprite_data = unsafe { Self::instance_bytes(std::slice::from_ref(&sprite)) };
            if !self.draw_instances_with_texture(
                sprite_data,
                1,
                &cached.view,
                &self.resources().pipelines.paths,
                instance_offset,
                pass,
            ) {
                return false;
            }
        }

        true
    }

    /// Invalidate all cached textures (e.g., on DPI or width change).
    pub(crate) fn invalidate_texture_cache(&mut self) {
        if let Some(pool) = &mut self.texture_pool {
            pool.textures.clear();
        }
    }
}
