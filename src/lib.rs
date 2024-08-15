use consts::{MAX_DISPATCH_SIZE, RTX_TITAN_MAX_BUFFER_SIZE};
use std::{borrow::Cow, sync::Arc};
use wgpu::{util::DeviceExt, Features};

pub mod consts;
pub mod data_utils;
pub mod debug_helpers;

use data_utils::gigs_of_zeroed_f32s;

pub async fn execute_gpu(numbers: &[f32]) -> Option<Vec<f32>> {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                // What features can we add?
                required_features: Features::STORAGE_RESOURCE_BINDING_ARRAY
                    | Features::BUFFER_BINDING_ARRAY
                    | Features::MAPPABLE_PRIMARY_BUFFERS,
                // What limits can we uplevel to?
                // required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                ..Default::default()
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
            label: Some("compute pass descriptor"),
            timestamp_writes: None, // We can use this with ComputePassTimestampWrites, let's find out how
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
    //TODO: it's quite possible for the 'work' to NOT fit into a single 4 binds available to each one of our 0..n 'groups' so, this needs a refactor
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
            log::debug!("bind_idx:{} buffer is {}b", bind_idx, buffer.size());
            wgpu::BindGroupEntry {
                binding: bind_idx as u32,
                resource: buffer.as_entire_binding(),
            }
        })
        .collect();

    let bind_group_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..storage_buffers.len())
        .map(|bind_idx| wgpu::BindGroupLayoutEntry {
            binding: bind_idx as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();

    log::debug!(
        "created {} BindGroupEntries with {} corresponding BindGroupEntryLayouts.",
        bind_group_entries.len(),
        bind_group_layout_entries.len()
    );

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
    // let numbers = gigs_of_zeroed_f32s(0.99);

    assert!(numbers.iter().all(|n| *n == 0.0));
    log::debug!("numbers.len() = {}", numbers.len());

    let t1 = std::time::Instant::now();
    let results = execute_gpu(&numbers).await.unwrap();
    log::debug!(">RUNTIME: {}ms", t1.elapsed().as_millis());

    assert_eq!(numbers.len(), results.len());

    /*  NOTES:
    So our 48% of 1GB worth of floats makes an array of f32 of length according to :
        numbers.len() = 128,849,016           // CPU side, which is correct
        arrayLength(v_indicies) =  33,554,432 // GPU side, which is NOT correct.

    Perhaps more unusually, we strike out with 0s long long before that 33.5 million:
        idx: 4,194,239= val:1
        idx: 4,194,240= val:0 (we should be 1.0 up until 33.5 million...)
        So this seems to be that the global_id.x never gets high enough

    which is exactly right for 1/4th of our bindings.
    Our global_id.x maxes out at = 4,194,239, which is starting to look familiar

    If we add a const OFFSET of that 4_194_239 and try to do the add_one calls at evenly,
    strode segments of the array we get 1.0s up to:
        idx: 16,776,960, val:0.0
    which is ~4x what we had before so that makes sense, but we're still a long way from
    incrementing all the way up to 33.5 million, let alone 128.8 million.

    What happens if we just keep adding more at the stride * 'n' ?

    weirdly adding these you'd think may take us further, but it does not, we still tap out at=16_776_960:
        v_indices[global_id.x + OFFSET * 3u] = add_one(v_indices[global_id.x + OFFSET * 4u]);
        v_indices[global_id.x + OFFSET * 3u] = add_one(v_indices[global_id.x + OFFSET * 5u]);
        v_indices[global_id.x + OFFSET * 3u] = add_one(v_indices[global_id.x + OFFSET * 6u]);
        v_indices[global_id.x + OFFSET * 3u] = add_one(v_indices[global_id.x + OFFSET * 7u]);

    So what gives? is this ACTUALLY the size limit on our buffer?

    What happens if we go from the 0.48 to 0.99 in our gigs_of_zeroed_f32s, and keep the above, will there be more buffer for us to write to?
        ERROR wgpu::backend::wgpu_core > Handling wgpu errors as fatal by default
        thread 'main' panicked at /home/jer/.cargo/registry/src/index.crates.io-6f17d22bba15001f/wgpu-22.1.0/src/backend/wgpu_core.rs:3411:5:
        wgpu error: Validation Error

        Caused by:
          In Device::create_bind_group_layout, label = 'Custom Storage Bind Group Layout'
            Too many bindings of type StorageBuffers in Stage ShaderStages(COMPUTE), limit is 4, count was 8. Check the limit `max_storage_buffers_per_shader_stage` passed to `Adapter::request_device`

    Ok so < 1/2 a GB is the max, which is weird because our wgpu::Limits report:
         {...,  "max_storage_buffer_binding_size": 2147483648, ... } // 2.147GB


    Well regardless, we'll have to refactor the binding part again to accomodate a max of 4 bindings per group, so if our data is bigger than those 4 bindings can hold, we'll need additional groups too :(


    Maybe it's time to reach out to chris? Reach out: chris@fleetwood.dev



    */
    results.iter().enumerate().for_each(|(e, v)| {
        if *v == 1.0 {
            // Add this check to avoid underflow when e is 0

            println!(
                "Pre Panic @ idx-2: {}, val: {}",
                format_large_number((e - 2) as u32),
                format_large_number(results[e - 2] as u32)
            );

            println!(
                "Pre Panic @ idx-1: {}, val: {}",
                format_large_number((e - 1) as u32),
                format_large_number(results[e - 1] as u32)
            );
            println!(
                "Panic     @ idx  : {}, val: {}",
                format_large_number(e as u32),
                format_large_number(*v as u32)
            );
            panic!()
        }
    });
    assert!(results.iter().all(|n| *n == 1.0));
}

fn format_large_number(num: u32) -> String {
    let num_str = num.to_string();
    let num_digits = num_str.len();

    if num_digits <= 3 {
        return num_str;
    }

    // Calculate the number of underscores to be added
    let num_underscores = (num_digits - 1) / 3;

    // Precalculate the capacity
    let capacity = num_digits + num_underscores;

    let mut result = String::with_capacity(capacity);

    for (i, c) in num_str.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push('_');
        }
        result.push(c);
    }

    result.chars().rev().collect()
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
                log::debug!("creating Storage Buffer {} of {}", e + 1, chunks.len());

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
