use consts::RTX_TITAN_MAX_BUFFER_SIZE;
use std::{borrow::Cow, sync::Arc};
use wgpu::util::DeviceExt;

pub mod consts;
pub mod data_utils;
pub mod debug_helpers;

use data_utils::gigs_of_zeroed_f32s;

const MAX_DISPATCH_SIZE: u32 = 65535;

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
    let (staging_buffers, storage_buffers, bind_group, compute_pipeline) = setup(device, numbers);

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
        cpass.dispatch_workgroups(MAX_DISPATCH_SIZE.min(numbers.len() as u32), 1, 1);
        // Number of cells to run, the (x,y,z) size of item being processed
    }

    // Assuming `storage_buffers` and `destination_buffers` are Vec<wgpu::Buffer> and have the same length
    storage_buffers
        .iter()
        .zip(staging_buffers.iter())
        .into_iter()
        .for_each(|(storage_buffer, staging_buffer)| {
            let sb_size = storage_buffer.size();
            let stg_size = staging_buffer.size();

            assert!(
                (sb_size % wgpu::COPY_BUFFER_ALIGNMENT == 0
                    && sb_size % wgpu::COPY_BUFFER_ALIGNMENT == 0)
            );
            assert_eq!(sb_size, stg_size);

            encoder.copy_buffer_to_buffer(
                storage_buffer, // Source buffer
                0,              // Source offset
                staging_buffer, // Destination buffer
                0,              // Destination offset
                stg_size,
            );
        });

    log::debug!("buffers created, submitting job to GPU");

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));
    log::debug!("Job submission complete.");

    let mut buffer_slices = Vec::new();
    staging_buffers.iter().for_each(|sb| {
        buffer_slices.push(sb.slice(..));
    });

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

fn setup(
    device: &wgpu::Device,
    numbers: &[f32],
) -> (
    Vec<wgpu::Buffer>,
    Vec<wgpu::Buffer>,
    wgpu::BindGroup,
    wgpu::ComputePipeline,
) {
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

    // ------------------------- BIND THE BUFFERS -------------------------
    // A bind group defines how buffers are accessed by shaders.
    let (bind_group_layout, bind_group) = setup_binds(&storage_buffers, device);

    let compute_pipeline = setup_pipeline(device, bind_group_layout, cs_module);
    (
        staging_buffers,
        storage_buffers,
        bind_group,
        compute_pipeline,
    )
}

fn setup_pipeline(
    device: &wgpu::Device,
    bind_group_layout: wgpu::BindGroupLayout,
    cs_module: wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });
    compute_pipeline
}

fn setup_binds(
    storage_buffers: &[wgpu::Buffer],
    device: &wgpu::Device,
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let bind_group_entries: Vec<wgpu::BindGroupEntry> = storage_buffers
        .iter()
        .enumerate()
        .map(|(bind_idx, buffer)| {
            log::debug!("{} buffer is {}b", bind_idx, buffer.size());
            wgpu::BindGroupEntry {
                binding: bind_idx as u32,
                resource: buffer.as_entire_binding(),
            }
        })
        .collect();

    let bind_group_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..storage_buffers.len())
        .map(|idx| wgpu::BindGroupLayoutEntry {
            binding: idx as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Combined Storage Bind Group"),
        layout: &bind_group_layout,
        entries: &bind_group_entries,
    });
    (bind_group_layout, bind_group)
}

pub async fn run() {
    let numbers = gigs_of_zeroed_f32s(0.48);

    assert!(numbers.iter().all(|n| *n == 0.0));
    log::debug!("numbers.len() = {}", numbers.len());

    let t1 = std::time::Instant::now();
    let results = execute_gpu(&numbers).await.unwrap();
    log::debug!(">RUNTIME: {}ms", t1.elapsed().as_millis());

    assert_eq!(numbers.len(), results.len());

    //FIXME:

    /*  NOTES:
    arrayLength(v_indicies) according to the gpu =  33,554,432
    according to us                              = 128,849,016

    Perhapes more conceriningly, we strike out with 0s long long before that 33.5 million:
        idx: 4,194,239= val:1
        idx: 4,194,240= val:0 (we should be 1.0 up until 33.5 million...)
        So this seems to be that the global_id.x never gets high enough

    which is exactly right for 1/4th of our bindings
    */
    results.iter().enumerate().for_each(|(e, v)| {
        if *v == 0.0 {
            println!("idx: {}, val:{:#?}", e, v);
            panic!()
        } else {
            // println!("idx: {}, val:{:#?}", e, v);
        }
    });

    assert!(results.iter().all(|n| *n == 1.0));
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
                log::debug!("creating storage buffer {} of {}", e + 1, chunks.len());

                let size = std::mem::size_of_val(seg) as u64;
                assert!(size % wgpu::COPY_BUFFER_ALIGNMENT == 0);

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
            let size = std::mem::size_of_val(chunks[e]) as u64;
            assert!(size % wgpu::COPY_BUFFER_ALIGNMENT == 0);
            log::debug!("creating staging buffer {} of {}", e + 1, chunks.len());
            let b = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("staging buffer-{}", e)),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            b
        })
        .collect()
}
