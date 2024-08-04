use consts::RTX_TITAN_MAX_BUFFER_SIZE;
use std::{borrow::Cow, sync::Arc};
use wgpu::util::DeviceExt;

pub mod consts;
pub mod data_utils;
pub mod debug_helpers;

use data_utils::gigs_of_zeroed_f32s;

pub async fn execute_gpu(numbers: &[f32]) -> Option<Vec<f32>> {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::default();

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
        .await
        .unwrap();

    execute_gpu_inner(&device, &queue, numbers).await
}

async fn execute_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    numbers: &[f32],
) -> Option<Vec<f32>> {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    // Gets the size in bytes of the buffer.
    let input_size = std::mem::size_of_val(numbers) as wgpu::BufferAddress;
    log::debug!("Size of input {}b", &input_size);

    // ------------------------- BUFFERS -------------------------
    // Instantiates buffer without data.
    // `usage` of buffer specifies how it can be used:
    //   `BufferUsages::MAP_READ` allows it to be read (outside the shader, by the CPU).
    //   `BufferUsages::COPY_DST` allows it to be the destination of a copy.
    let staging_buffers = create_staging_buffers(device, numbers);
    log::debug!("Created staging_buffer");

    let storage_buffers = create_storage_buffers(device, numbers, input_size);
    log::debug!("Created storage_buffer");

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // ------------------------- BIND THE BUFFERS -------------------------
    // A bind group defines how buffers are accessed by shaders.

    let bind_group_entries: Vec<wgpu::BindGroupEntry> = storage_buffers
        .iter()
        .enumerate()
        .map(|(bind_idx, buffer)| {
            log::debug!("{} buffer is {}b", bind_idx, buffer.size());

            wgpu::BindGroupEntry {
                binding: bind_idx as u32,             // Unique binding index for each buffer
                resource: buffer.as_entire_binding(), // Reference to the buffer's entire range
            }
        })
        .collect();

    let bind_group_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..storage_buffers.len())
        .map(|idx| wgpu::BindGroupLayoutEntry {
            binding: idx as u32,
            visibility: wgpu::ShaderStages::COMPUTE, // or whatever stages you need
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false }, // adjust based on your needs
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Custom Storage Bind Group Layout"),
        entries: &bind_group_layout_entries,
    });

    // Now create the bind group using this layout
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Combined Storage Bind Group"),
        layout: &bind_group_layout,
        entries: &bind_group_entries,
    });
    // Now `bind_group` contains a binding for each buffer in `storage_buffers`

    // ------------------------- ENCODE -------------------------
    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        log::debug!("set_pipeline complete");
        cpass.set_bind_group(0, &bind_group, &[]);
        log::debug!("set_bind_group complete");
        cpass.insert_debug_marker("bump a gigabyte of floats of 0.0 all by 1.0");
        cpass.dispatch_workgroups(numbers.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }
    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    storage_buffers
        .into_iter()
        .enumerate()
        .for_each(|(_e, sb)| {
            //FIXME: how do we set the offsets? e*size of sb?
            encoder.copy_buffer_to_buffer(&sb, 0, &sb, 0, input_size / RTX_TITAN_MAX_BUFFER_SIZE);
        });

    log::debug!("buffers created, sending to GPU");

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let mut buffer_slices = Vec::new();
    staging_buffers.iter().for_each(|sb| {
        buffer_slices.push(sb.slice(..));
    });

    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = flume::bounded(buffer_slices.len());
    let sender = Arc::new(sender);

    buffer_slices.iter().for_each(|bs| {
        let sender = sender.clone();
        bs.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        })
    });

    // Poll the device in a blocking manner so that our future resolves.
    device.poll(wgpu::Maintain::wait());

    // Await buffer futures to read
    if let Ok(Ok(())) = receiver.recv_async().await {
        log::debug!("Getting results...");
        // Retrieve and process buffer data
        let data: Vec<f32> = buffer_slices
            .iter()
            .map(|bs| {
                let data = bs.get_mapped_range();
                let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                drop(data); // Drop to free buffer before unmap
                result
            })
            .flatten()
            .collect();

        // Since buffer_slices was not moved, we can still access it here
        staging_buffers.iter().for_each(|sb| sb.unmap()); // Unmaps buffer from memory

        Some(data) // Return the collected data
    } else {
        log::error!("Failed to run compute on GPU!");
        None
    }
}

pub async fn run() {
    let numbers = gigs_of_zeroed_f32s(0.48);

    let steps = execute_gpu(&numbers).await.unwrap();

    dbg!(steps);
}

pub fn calculate_chunks(numbers: &[f32], max_buffer_size: u64) -> Vec<&[f32]> {
    let max_elements_per_chunk = max_buffer_size as usize / std::mem::size_of::<f32>(); // Calculate max f32 elements per buffer
    numbers.chunks(max_elements_per_chunk).collect()
}

fn create_storage_buffers(
    device: &wgpu::Device,
    numbers: &[f32],
    input_size: u64, // bytes..
) -> Vec<wgpu::Buffer> {
    if input_size > RTX_TITAN_MAX_BUFFER_SIZE {
        log::warn!("Supplied input is too large for a single storage buffer, splitting...");

        let chunks = calculate_chunks(numbers, RTX_TITAN_MAX_BUFFER_SIZE);

        chunks
            .iter()
            .enumerate()
            .map(|(e, seg)| {
                log::debug!("creating buffer {} of {}", e + 1, chunks.len());

                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Storage Buffer-{}", e)),
                    contents: bytemuck::cast_slice(seg),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                })
            })
            .collect()
    } else {
        vec![
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Storage Buffer-0"),
                contents: bytemuck::cast_slice(numbers),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            }),
        ]
    }
}

fn create_staging_buffers(device: &wgpu::Device, numbers: &[f32]) -> Vec<wgpu::Buffer> {
    log::warn!("Supplied input is too large for a single staging buffer, splitting...");

    let chunks = calculate_chunks(numbers, RTX_TITAN_MAX_BUFFER_SIZE);

    log::debug!("num_chunks: {}", chunks.len());
    (0..chunks.len())
        .into_iter()
        .map(|e| {
            log::debug!("creating buffer {} of {}", e + 1, chunks.len());
            let b = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("staging buffer-{}", e)),
                size: std::mem::size_of_val(chunks[e]) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            b
        })
        .collect()
}
