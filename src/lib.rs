use consts::RTX_TITAN_MAX_BUFFER_SIZE;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

use data_utils::gigs_of_zeroed_f32s;

pub mod consts;
pub mod debug_helpers;

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

    // Instantiates buffer without data.
    // `usage` of buffer specifies how it can be used:
    //   `BufferUsages::MAP_READ` allows it to be read (outside the shader, by the CPU).
    //   `BufferUsages::COPY_DST` allows it to be the destination of a copy.
    let staging_buffer = create_staging_buffers(device, input_size);
    log::debug!("Created staging_buffer");

    let storage_buffers = create_storage_buffers(device, numbers, input_size); // Instantiates buffer with data (`numbers`).
                                                                               // Usage allowing the buffer to be:
                                                                               //   A storage buffer (can be bound within a bind group and thus available to a shader).
                                                                               //   The destination of a copy.
                                                                               //   The source of a copy.
    log::debug!("Created storage_buffer");

    // A bind group defines how buffers are accessed by shaders.
    // It is to WebGPU what a descriptor set is to Vulkan.
    // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

    // A pipeline specifies the operation of a shader

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);

    let bind_groups: Vec<wgpu::BindGroup> = storage_buffers
        .iter()
        .enumerate()
        .map(|(index, buffer)| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Bind Group {}", index)),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: index as u32, // Note: typically binding starts at 0 for each group
                    resource: buffer.as_entire_binding(),
                }],
            })
        })
        .collect();

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
        bind_groups
            .into_iter()
            .enumerate()
            .for_each(|(e, bg)| cpass.set_bind_group(e as u32, &bg, &[]));
        cpass.insert_debug_marker("compute collatz iterations");
        cpass.dispatch_workgroups(numbers.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }
    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    storage_buffers.into_iter().enumerate().for_each(|(e, sb)| {
        //FIXME: how do we set the offsets? e*size of sb?
        encoder.copy_buffer_to_buffer(&sb, 0, &staging_buffer, 0, input_size);
    });

    log::debug!("buffers created, sending to GPU");

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    // Awaits until `buffer_future` can be read from
    if let Ok(Ok(())) = receiver.recv_async().await {
        log::debug!("Getting results...");
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result = bytemuck::cast_slice(&data).to_vec();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
                                // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                //   delete myPointer;
                                //   myPointer = NULL;
                                // It effectively frees the memory

        // Returns data from buffer
        Some(result)
    } else {
        log::error!("failed to run compute on gpu!");
        None
    }
}

pub async fn run() {
    let numbers = gigs_of_zeroed_f32s(1);

    let steps = execute_gpu(&numbers).await.unwrap();

    dbg!(steps);
}

pub mod data_utils {

    pub fn gigs_of_zeroed_f32s(n: usize) -> Vec<f32> {
        let bytes_per_gb = 1024 * 1024 * 1024; // 1 GB in bytes
        let bytes_per_f32 = std::mem::size_of::<f32>(); // Size of f32 in bytes
        let elements = n * bytes_per_gb / bytes_per_f32;

        vec![0.0; elements]
    }
}

fn create_storage_buffers(
    device: &wgpu::Device,
    numbers: &[f32],
    input_size: u64,
) -> Vec<wgpu::Buffer> {
    if input_size > RTX_TITAN_MAX_BUFFER_SIZE {
        log::warn!("Supplied input is too large for a single buffer, splitting...");

        numbers
            .chunks((input_size / RTX_TITAN_MAX_BUFFER_SIZE) as usize)
            .enumerate()
            .map(|(e, seg)| {
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

fn create_staging_buffers(device: &wgpu::Device, input_size: u64) -> Vec<wgpu::Buffer> {
    if input_size > RTX_TITAN_MAX_BUFFER_SIZE {
        log::warn!("Supplied input is too large for a single buffer, splitting...");
        let num_chunks = (input_size / RTX_TITAN_MAX_BUFFER_SIZE) as usize;
        (0..num_chunks)
            .into_iter()
            .map(|e| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("staging buffer-{}", e)),
                    size: input_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect()
    } else {
        vec![device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: input_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })]
    }
}
