# GP-GPU with wgpu.

## Context:
Sometimes I run `llms` at home, most recently for [this project](www.github.com/alphastrata/titler), which uses `NuExtract` to title pdfs from my many, many years of collecting papers from avrix.

Seeing the GPU go _brrrrrrrr_, made me think about getting large data (i.e model weights) onto the gpu, and thinking about that made me curious so.... how does one get a gigabyte's worth of `f32`s onto a gpu, add one to all of them, then bring them back?

_You may be surprised to see how (in wgpu at least) non trivial this is._

This post is not for beginners, go and check out and internalise these two examples' worth of content to get up to speed:
1. [hello_compute](https://github.com/gfx-rs/wgpu/tree/trunk/examples/src/hello_compute)
2. [repeated_compute](https://github.com/gfx-rs/wgpu/tree/trunk/examples/src/repeated_compute)

You've read those? -- great, let's get started!

# Buffers 
We need a few buffers, namely `staging_buffers, storage_buffers`, we'll have to take the length of the array holding our gigabyte of `f32`s, and split it into an equal number of these `staging_buffers, storage_buffers`, because each storage buffer needs a copy-back-to-cpu-land buffer of equal size and so on.

Naively if we try to slap the entirety of our `f32`s into a single buffer we get an error like this:
TODO:

# Detour to Limits 
Unfortunately all GPUs have different sizes of buffers, and different numbers of buffers per `group`(more on that later) that they support, so to get these:
```rust
use wgpu;

pub async fn debug_gpu_info() {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();

    println!("features : {:?}", adapter.features());
    println!("get_info : {:?}", adapter.get_info());
    println!("limits   : {:?}", adapter.limits());
}
```
I'd reccomend dumping the output of these to a file for quick reference, because simply taking the output of, for example the `println!("get_info : {:?}", adapter.get_info());` for example won't tell you much that is actually useful from the perspective of wgpu interfacing with the driver your GPU is using, i.e my card has `24GB` of VRAM, but I cannot just 'access' it in the same way you'd ask for memory when CPU programming (unfortunately). (This should seem obvious from the section on buffers we just started talking about no?)

For my RTX Titan the `wgpu::Limits` indicate:
```json
{  
  "max_bind_groups": 8, // THIS IS IMPORTANT
  "max_bindings_per_bind_group": 1000,
  "max_dynamic_storage_buffers_per_pipeline_layout": 16,
  "max_samplers_per_shader_stage": 1048576,
  "max_storage_buffers_per_shader_stage": 1048576,
  "max_uniform_buffers_per_shader_stage": 1048576,
  "max_uniform_buffer_binding_size": 65536,
  "max_storage_buffer_binding_size": 2147483648,
  "max_buffer_size": 18446744073709551615, // This seems wrong, 180GB?
  "min_uniform_buffer_offset_alignment": 64,
  "min_storage_buffer_offset_alignment": 32,
  "max_inter_stage_shader_components": 128,
  "max_compute_workgroup_storage_size": 49152,
  "max_compute_invocations_per_workgroup": 1024,
  "max_compute_workgroup_size_x": 1024,
  "max_compute_workgroup_size_y": 1024,
  "max_compute_workgroup_size_z": 64,
  "max_compute_workgroups_per_dimension": 65535, // THIS IS IMPORTANT
  "min_subgroup_size": 32,
  "max_subgroup_size": 32,
  "max_push_constant_size": 256,
  "max_non_sampler_bindings": 4294967295 
}
```

I'll setup some consts, based on that:
```rust
pub const RTX_TITAN_MAX_BUFFER_SIZE: u64 = 134_217_728; //134MB
pub const RTX_TITAN_MAX_BIND_GROUPS: u64 = 8;
pub const RTX_TITAN_MAX_BINDS_PER_GROUP: u64 = 1_000;
pub const MAX_DISPATCH_SIZE: u32 = 65_535;
```

# Buffers for realz this time
So this `RTX_TITAN_MAX_BUFFER_SIZE` tells us that we'll need to be taking our big-ass array of f32s, and at subdividing it into buffers that are `<=134`MB. 
```rust
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

                /*
                We have to ensure our buffers, when pumped full of data meet the alignment requirements:
                    https://docs.rs/wgpu/latest/wgpu/constant.COPY_BUFFER_ALIGNMENT.html,
                    https://sotrh.github.io/learn-wgpu/showcase/alignment/
                */
                let size = std::mem::size_of_val(seg) as u64;
                assert!(size % wgpu::COPY_BUFFER_ALIGNMENT == 0);

                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Storage Buffer-{}", e)),
                    // Initialise the buffer with the contents of our segment, which is one of our chunks.
                    contents: bytemuck::cast_slice(seg), 
                    // This is a storage buffer
                    usage: wgpu::BufferUsages::STORAGE 
                        // This buffer can be the TARGET of a copy (i.e into this buffer)
                        | wgpu::BufferUsages::COPY_DST 
                        // This buffer can be the SOURCE of a copy (i.e out of this buffer)
                        | wgpu::BufferUsages::COPY_SRC, 
                })
            })
            .collect()
    } else {
    // handle the n of 1.
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
```
For a simple definition of `calculate_chunks`, doing the maximum size a buffer supports over the size of an `f32` (`4` bytes in our case), then leveraging the magic of [`chunks`](https://doc.rust-lang.org/std/slice/struct.Chunks.html).
```rust
pub fn calculate_chunks(numbers: &[f32], max_buffer_size: u64) -> Vec<&[f32]> {
    let max_elements_per_chunk = max_buffer_size as usize / std::mem::size_of::<f32>();
    numbers.chunks(max_elements_per_chunk).collect()
}
```
Now we setup corresponding `staging` buffers, for which code looks very similar:
```rust
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
                usage: 
                // This buffer can be read by both the CPU and the GPU, 
                wgpu::BufferUsages::MAP_READ | 
                // This buffer can be the TARGET of a copy
                wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            b
        })
        .collect()
}

```

# The 'hell' of `Bindings`
Next up we need to make sure the bindings (which is how you make things available to your shader code), and the bind-group-layout (which is a higher level up of bindings) are all setup.If you were unable to gleam any insights form the aforementioned examples from wgpu actual's tutorials/documentation, know that: we essentially just allocated, some gpu memory, but currently our GPU has no idea what to do with it, and our shader code has no idea what that memory contains, nor how to access it. Which is where `bind_groups, bindings` and `layouts` come in.

A `bind_group`, contains a `layout` which is really a human-labelled collection of `entries`. To make things _extra_ nauseating wgpu prefaces all these with `BindGroup`, so you have to deal with this eye cancer:
````
wgpu::BindGroup
wgpu::BindGroupEntry
wgpu::BindGroupLayout
wgpu::BindGroupDescriptor
wgpu::BindGroupLayoutEntry
wgpu::BindGroupLayoutDescriptor
````

When in doubt I find the following illustration useful:
```
               ┌────────────────────┐
               │ BindGroup          │
               │        ▲        ▲  │
               │┌───────┼┐┌──────┼┐ │
               ││Entries│││BindGroupLayout
               ││        ││       │ │
┌────────────┐ ││ ┌────┐ ││┌────┐ │ │
│ Storage    │ ││ │Entry |││BindGroupLayoutEntry
│            │◄┼┼─┼    │◄┼┼┼    │ │ │
└────────────┘ ││ └────┘ ││└────┘ │ │
┌────────────┐ ││ ┌────┐ ││┌────┐ │ │
│ Storage    │ ││ │    │ │││    │ │ │
│            ◄─┼┼─┼    │◄┼┼┼    │ │ │
└────────────┘ ││ └────┘ ││└────┘ │ │
┌────────────┐ ││ ┌────┐ ││┌────┐ │ │
│ Storage    │ ││ │    │ │││    │ │ │
│            │◄┼┼─┼    │◄┼┼┼    │ │ │
└────────────┘ ││ └────┘ ││└────┘ │ │
               ││        ││       │ │
               ││        ││       │ │
               │└────────┘└───────┘ │
               │                    │
               └────────────────────┘

```
Note also that all of these things are usually being added to your wgpu stuff is being wrapped in smart pointers etc which is why the `&[T]`s you're passing to all these `wgpu::Device::create_bind_group_layout` etc are living outside the scope of our helper functions that create them.

So for _every_ `storage_buffer` we create, it will have a unique entry that points at said `wgpu::Buffer`, which will be described by its own layout, and you'll end up with two collections of these, in 1:1 pairings, these pairings will be encapsulated in a single `BindGroupLayout`, which will be encapsulated in a single `BindGroup`.

TODO: that's easy meme or math confused kid meme?

yikes, what does this code look like you may ask?
```rust
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
    // NOTE: that because there are no `&`s in the layout entries, the ONLY thing that binds them is the bind_idx, 0:0, 1:1, 2:2,...n:n

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
```
I'll mention this again because it's important, the **only** way that our `bind_group_entries` know which of the `bind_group_layout_entries`, applies to them is that they share the same `bind_idx`, so it is **absolutely crucial** that you don't mess this up.

Ok now we have data in our `Buffer`s, and the memory layout is sorted, we need to put all this information into a `pipeline`.

# Pipelines 
Aptly named as you put stuff in one side, and if all goes well it'll fall out the other :wink.

We're doing compute so we'll make a `wgpu:::ComputePipeline`, and as the name suggests it's a pipeline so we'll set it up, put our stuff in one side then go to the other side for what we hope is our gigabyte's worth of floats all incremented by 1.

The `ComputePipeline` follows a similar diagram to the earlier one describing the relation between binds and buffers, but the pipeline requires our shader modules, because it's _in_ the pipeline (i.e on the GPU) that our shader(s) are gonna do their thing:
```rust 
//Setup a shader module:
let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: None,
    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
});

```
Our actual wgsl code is trivial, but we'll talk about it here to underscore the work we _just_ did in `Bind`ings:
```rust
@group(0)
@binding(0)
var<storage, read_write> v_indices: array<f32>;

fn add_one(n: f32) -> f32 {
    return n + 1.0;
}

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    v_indices[global_id.x] = add_one(v_indices[global_id.x]); 
}
```
The keen eyed amongst you will be asking, "we declared 8 buffers, so the `bind_idx` would count up to `7`, why do you only have `@binding(0)`?" and by Joe you'd be right! (but we'll get to that because you're really jumping the pain gun I've been dealing with here, which is of course in no small part the motivation for writing all this up). 
TODO: clever girl gif 

Anyway, we have a shader, let's give it to the pipeline, using a helper:
```rust 
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
```

# Actually 'doing' work:
Now after 100s of lines of boilerplate and setup we can finally `Encode` (i.e issue gpu commands) with our `wgpu::Device`, we're doing 'compute' so we'll be doing a `compute_pass`, which in most of the code out there you'll see referred to as `cpass` as I've done here:
```rust
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute pass descriptor"),
            timestamp_writes: None, 
        });
        cpass.set_pipeline(&compute_pipeline); // But wait, we didn't make this...
        log::debug!("set_pipeline complete");
        cpass.set_bind_group(0, &bind_group, &[]);
        log::debug!("set_bind_group complete");
        cpass.insert_debug_marker("bump a gigabyte of floats of 0.0 all by 1.0");
        cpass.dispatch_workgroups(MAX_DISPATCH_SIZE.min(numbers.len() as u32), 1, 1);
    }
```

Then we use that encoder to say, I'd like to do some buffer -> buffer copying please:
```rust
/* Snip ... */
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
                staging_buffer, // Target buffer
                0,              // Destination offset
                stg_size,
            );
        });
```


# Home stretch!
Here we're still not done, we've only enqued our work, we need to then ask the `wgpu::Queue` to submit our work, then we'll go _wait_ for it (position ourselves at the other side of the pipe where stuff spews out...)
`queue.submit(Some(encoder.finish())); `

But.... now we're done!

# Results
... except for checking our results:

```rust
// use the wgpu::slice to create an easy way to try and 'read'(later) from our buffers
let mut buffer_slices = Vec::new();
staging_buffers.iter().for_each(|sb| {
    buffer_slices.push(sb.slice(..));
});

// Setup some tx,rx
let (sender, receiver) = flume::bounded(buffer_slices.len());
let sender = Arc::new(sender);

buffer_slices.iter().for_each(|bs| {
    let sender = sender.clone();
    bs.map_async(wgpu::MapMode::Read, move |v| {
        sender.send(v).unwrap();
    })
});

// Poll the device in a blocking manner so that our futures(every iteration of that above bs.map_async) resolves.
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
/* Snip... */
    
```
We can call `slice()` on our `wgpu:Buffer`s, then use `Read` to asynchronously move their data across a bounded channel (bounded because we want to stay listening until we've received _out_ of our pipe the same number of things we put _in_).
---

# Naive Results pt1
As you'd of noticed we peppered our app with a bunch of logging, so running this on our input of a GB of f32s will give us something like this:
```sh
 DEBUG wgpu_prac_v2                > numbers.len() = 265751104
 DEBUG wgpu_prac_v2                > Size of input 1063004416b
 WARN  wgpu_prac_v2                > Supplied input is too large for a single staging buffer, splitting...
 DEBUG wgpu_prac_v2                > num_chunks: 8
/*
Snipping
*/
 DEBUG wgpu_prac_v2                > creating staging buffer 7 of 8
 DEBUG wgpu_prac_v2                > creating staging buffer 8 of 8
 DEBUG wgpu_prac_v2                > Created staging_buffer
 WARN  wgpu_prac_v2                > Supplied input is too large for a single storage buffer, splitting...
 DEBUG wgpu_prac_v2                > creating storage buffer 1 of 8
 DEBUG wgpu_prac_v2                > creating storage buffer 2 of 8
 DEBUG wgpu_prac_v2                > creating storage buffer 3 of 8
/*
Snipping
*/
 DEBUG wgpu_prac_v2                > 6 buffer is 134217728b
 DEBUG wgpu_prac_v2                > 7 buffer is 123480320b
```
Great, however...
```sh
 ERROR wgpu::backend::wgpu_core    > Handling wgpu errors as fatal by default
thread 'main' panicked at /home/jer/.cargo/registry/src/index.crates.io-6f17d22bba15001f/wgpu-22.1.0/src/backend/wgpu_core.rs:3411:5:
wgpu error: Validation Error

Caused by:
  In Device::create_bind_group_layout, label = 'Custom Storage Bind Group Layout'
    Too many bindings of type StorageBuffers in Stage ShaderStages(COMPUTE), limit is 4, count was 8. Check the limit `max_storage_buffers_per_shader_stage` passed to `Adapter::request_device`
```

This is why I mentioned (TODO link) earlier that the, `max_storage_buffers_per_shader_stage` is important, having said that the value we get back (on my card) is: `"max_storage_buffers_per_shader_stage": 1048576`, which ... is, I'm pretty sure **not** 8.

## Trying to push forward
Ok, so let's drop our input size, 1/2 a gig of floats is still quite a large number of them, and a reasonable 'bbrrrrr' is sure to be had right?

Whip up a quick `run` function to test this more speedier and:
```rust
pub async fn run() {
    let numbers = gigs_of_zeroed_f32s(0.48); // Yes I know it's slightly less than half.

    assert!(numbers.iter().all(|n| *n == 0.0));
    log::debug!("numbers.len() = {}", numbers.len());

    let t1 = std::time::Instant::now();
    let results = execute_gpu(&numbers).await.unwrap();
    log::debug!("GPU RUN: {}ms", t1.elapsed().as_millis());

    assert_eq!(numbers.len(), results.len());

    assert!(results.iter().all(|n| *n == 1.0));
}
```

Ok so with ~1/2 the input size...
_drumroll please_
```sh

 DEBUG wgpu_prac_v2                > numbers.len() = 128849016
 DEBUG wgpu_prac_v2                > Size of input 515396064b
 WARN  wgpu_prac_v2                > Supplied input is too large for a single staging buffer, splitting...
 DEBUG wgpu_prac_v2                > num_chunks: 4
 DEBUG wgpu_prac_v2                > creating staging buffer 1 of 4
/*
Snip
*/ 
 DEBUG wgpu_prac_v2                > set_pipeline complete
 DEBUG wgpu_prac_v2                > set_bind_group complete
 DEBUG wgpu_prac_v2                > buffers created, submitting job to GPU
 DEBUG wgpu_prac_v2                > Job submission complete.
 DEBUG wgpu_prac_v2                > Getting results...
 DEBUG wgpu_prac_v2                > >RUNTIME: 1816ms

thread 'main' panicked at /home/jer/Documents/rust/wgpu-prac-v2/src/lib.rs:313:5:
assertion failed: results.iter().all(|n| *n == 1.0)
```
Our 0s are not all turned 1s...

Well, the obvious thing to do is to see how many of our 0s, got the 1 treatment.
To achieve this we'll `println!` debug, like all the bestesterest devs do:
```rust
    results.iter().enumerate().for_each(|(e, v)| {
        if *v == 0.0 {
            println!("idx: {}, val:{:#?}", e, v);
            panic!() // you'll see in a moment why I didn't want to print all the 0s...
        }
    });
```
>Output:
```sh
idx: 4194240, val:0.0
thread 'main' panicked at /home/jer/Documents/rust/wgpu-prac-v2/src/lib.rs:309:13:
```
What...!  mere 4.19 Million of our 128 Million floats has been one-a-tised?
Debugging this I started with what (probably?) even multiple of our 128 Million is the what we got?

Answer: `30.7204680705`, and that ain't even _even_!

hmmmm, how many 1s are we expecting _per_ buffer?
Answer: 128 Million over the number of buffers which is 4, so ~33.5 Million.
Well... it looks (TODO: fallout thumb) like we've got about ~8% of a Buffer's worth, maybe the problem is in our shader?

# Could it be the bindings?
Ahh, our bindings are created in a loop, that runs once for each of our buffers so the problem must be that we've only, in wgsl declared one, of our _four_ bindings in the BindGroup:
```rust
@group(0)
@binding(0)
var<storage, read_write> v_indices: array<f32>;
```
Well, this is a problem. if we add the others:
```rust
@group(0)
@binding(0)
var<storage, read_write> v0_indices: array<f32>;
@group(0)
@binding(1)
var<storage, read_write> v1_indices: array<f32>;
@group(0)
@binding(2)
var<storage, read_write> v2_indices: array<f32>;
@group(0)
@binding(3)
var<storage, read_write> v3_indices: array<f32>;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
// We'll also modify our add ones to just tick each of these new arrays:
    v0_indices[global_id.x] = add_one(v0_indices[global_id.x]);
    v1_indices[global_id.x] = add_one(v1_indices[global_id.x]);
    v2_indices[global_id.x] = add_one(v2_indices[global_id.x]);
    v3_indices[global_id.x] = add_one(v3_indices[global_id.x]);

/* Snip */
```
This is, well.. to put it _lightly_ disgusting, in the strictest definition of the _hard_ part of hard coding something this is _hard_ stuck like this forever, our shader is now bespoke coupled to our input -- but, for now, let's allow this and see if shit works:
```sh
 DEBUG wgpu_prac_v2                > Getting results...
 DEBUG wgpu_prac_v2                > >RUNTIME: 1747ms
idx: 4194240, val:0.0
thread 'main' panicked at /home/jer/Documents/rust/wgpu-prac-v2/src/lib.rs:315:13:
```
well darn...

# Perhaps our array isn't all there?
Maybe it's a counting issue, how _high_ does our global_id.x go to? i.e how much of the array that we're expecting do we have GPU side?
```rust
    //Modify our debug output a bit:
    results.iter().enumerate().for_each(|(e, v)| {
        if *v == 0.0 {
            println!("Pre Panic @ idx-1: {}, val: {}", e - 1, format!("{:_}", results[e - 1] as u32));
            println!("Panic     @ idx  : {}, val: {}", e, format!("{:_}", v as u32));
            panic!()
        }
    });
```
And modify our shader to, instead of `add_one`ing everything we'll just write the length of the array as the gpu sees it:
```rust
/* Snip */
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let len = f32(arrayLength(&v_indices));
    v_indices[global_id.x] = len; 
/* Snip */

```
but alas (Note I'm formatting these numbers to make them easier to read from now on)
```sh
Pre Panic @ idx-1: 4_194_239, val:35_54_432
Panic     @ idx  : 4_194_240, val:0.0
thread 'main' panicked at /home/jer/Documents/rust/wgpu-prac-v2/src/lib.rs:316:13:    
```
Ok so our `v_indices` from our array is 1/4 the length of our input, which is expected.
Our `global_id.x`, is far short of that, does mean there's a hard limit on the global invocations of a thread? let's investigate that.

# The Workgroups, it's gotta be the workgroups!
You call for threads in `Workgroups` in `wgpu`, and we did that here:
```rust
//NOTE: MAX_DISPATCH_SIZE = 65_535, according to my GPU's wgpu::Limits' output.
cpass.dispatch_workgroups(MAX_DISPATCH_SIZE.min(numbers.len() as u32), 1, 1);
```
but the minimum of the length of our input, 128Million and that MAX_DISPATCH_SIZE is **not** 4.1 Million, so what gives?

# wax-on-wax-off
Well, (and I'm not ashamed to say that this is what I did, because learning GPU programming is hard), I exhaustively checked every magic number in my code and in the output of `wgpu::Limits`until I found that:
4_194_240 is *exactly* divisible by our `@workgroup_size(64)`, and if we bump that number up to 256 (we'd expect a number four times as high right?!)
```sh
Pre Panic @ idx-1: 16_776_959, val:33_554_432.0
Panic     @ idx  : 16_776_960, val:0.0
thread 'main' panicked at /home/jer/Documents/rust/wgpu-prac-v2/src/lib.rs:316:13:
```
HAHA! FUCK YOU WGPU we'll wade through the molasses that is your documentation and the general murk that GPGPU programming tutorials are made of, even if we have to push every single line of wgpu(rust obviously not that JS shit) and WGSL that exists on github through chatGPT we're gonna do this!!!

We need only turn the volume up to 11 and crank that `workgroup_size(512)`, the most splendid of powers of 2, the amount of RAM in my third ever PC build as a child. And we'll touch that that sweet sweet nectar of 33.5 Million onesised 0.0s, _WITNESS ME_.
TODO: witness me gif

```sh
 ERROR wgpu::backend::wgpu_core    > Handling wgpu errors as fatal by default
thread 'main' panicked at /home/jer/.cargo/registry/src/index.crates.io-6f17d22bba15001f/wgpu-22.1.0/src/backend/wgpu_core.rs:3411:5:
wgpu error: Validation Error

Caused by:
  In Device::create_compute_pipeline, label = 'Compute Pipeline'
    Error matching shader requirements against the pipeline
      Shader entry point's workgroup size [512, 1, 1] (512 total invocations) must be less or equal to the per-dimension limit [256, 256, 64] and the total invocation limit 256
```
Well, when the errors are good, they _are_ good.

When one sees this sorta thing `[256, 256, 64]`, it's probably indicative that we can restructure our data in such a way that it's more palatable for the GPU, which is probably a good thing to realise.

_We have maybe been thinking about this the wrong way, the innovations of the GPU were in part largely driven by the need to have better 3d performance, and we've been giving this thing a 1d array.

let's restructure our data.

# his thinking is rather 1 dimensional cap'n
TODO: Spok to kirk meme when fighting khan on the 'z' axis 

what if we pretend that `global_id.x + OFFSET` is like an index into the non existant `y` of our array?
>shader
```rust

const OFFSET:u32 = 16776959;
@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    v_indices[global_id.x] = add_one(v_indices[global_id.x]); 
    v_indices[global_id.x + OFFSET] = add_one(v_indices[global_id.x + OFFSET]); 
```
gives us;
```sh
 DEBUG wgpu_prac_v2                > Getting results...
 DEBUG wgpu_prac_v2                > >RUNTIME: 1676ms
Pre Panic @ idx-1: 33_553_918, val: 33_554_432
```
Hoorah, a single buffer has been one-a-tised.

But this feels dirty, I don't like this const offset nonsense to dictate a special stride that we only know as a result of a bunch of debug prints, can we do better by somehow deriving the relationship between whatever x,y and z values are passed to a workgroup and have that dictate what our output needs to be, because at the moment (even though we can't yet touch that 4th buffer) it is:
```
 DEBUG wgpu_prac_v2                > bind_idx:0 buffer is 134217728b
 DEBUG wgpu_prac_v2                > bind_idx:1 buffer is 134217728b
 DEBUG wgpu_prac_v2                > bind_idx:2 buffer is 134217728b
 DEBUG wgpu_prac_v2                > bind_idx:3 buffer is 112742880b
```
strictly smaller than the others, meaning that that above implementaion would try to write out of bounds, which we can only assume is as bad in GPGPU land as it is in CPU land.

So we know that the TOTAL number of threads (and therefore thread.xyz) indexes we'll be able to extract for our GPU is given by the: `(workgroup dimensions * MAX_DISPATCH_SIZE)`, and those workgroup dimensions canot exceed `256` (in my case ) which in the `wgpu::Limits` I get from my titan only corresponds to the `max_push_constant_size`, and those workgroup dimsenions have individual hard limits on their sizes that we saw earlier from the Error `...  must be less or equal to the per-dimension limit [256, 256, 64] and the total invocation limit 256`. 
So we could supply any of these:

| **X** | **Y** | **Z** | **Total Invocations** |
|-------|-------|-------|-----------------------|
| 256   | 1     | 1     | 256                   |
| 128   | 2     | 1     | 256                   |
| 64    | 4     | 1     | 256                   |
| 32    | 8     | 1     | 256                   |
| 16    | 16    | 1     | 256                   |
| 16    | 8     | 2     | 256                   |
| 8     | 8     | 4     | 256                   |
| 4     | 4     | 16    | 256                   |
| 8     | 4     | 8     | 256                   |
| 4     | 8     | 8     | 256                   |

Which I don't think helps us terribly much for _this_ problem, as our goal of one-and-done-ing a bunch of 0s, is kinda (albeit silly) not particularly favouring any dimensional slicey-dicey way of looking at things.

What is probably more interesting is, _within_ our shader code can we know what the XYZ values are, i.e withing any invocation do we know information about the workgroup size?o

[According to the docs](https://www.w3.org/TR/WGSL/#example-4ad7a4a0), we have the following:
```rust
 @compute @workgroup_size(64)
 fn cs_main(
   @builtin(local_invocation_id) local_id: vec3<u32>,
   @builtin(local_invocation_index) local_index: u32,
   @builtin(global_invocation_id) global_id: vec3<u32>,
) {}
```
So not useful.


## Those other pesky buffers:
Because this is proving difficult, let's change gears for a moment and confirm that some of those _other_ buffers, i.e the one's potentially not at `@binding(0)` have 0s in them, we can even try to one-a-tise them.

```rust
@group(0)
@binding(1) // we'll just keep bumping this
var<storage, read_write> v_indices: array<f32>;
```

Recall that with the `@binding(0)` set to `0`, we've only been able to make ones of that first buffer from the 0th element to 33.5 Million (close to `2^25`), i.e:
```sh
DEBUG wgpu_prac_v2                > >RUNTIME: 1745ms
Pre Panic @ idx-1: 33_553_918, val: 1
Panic     @ idx  : 33_553_919, val: 0
```
Which now we look at it, is a strange place for our 1-ing to be conking out, the buffer size according to our debug logs is: `134217728b`, and each of our `f32`s is `4 bytes`, and when we take that `134217728 / 4` it `!= 33,553-918`, it's `33,554,432`. Where's our missing `514` values from the `0th` buffer?!

# returning to 0
before moving on we _must_ get this first buffer one-and-done-ified.

Let's take a fundamental look at some of our numbers again, because it looks as though we're getting a little muddled with what our constraints actually are.

Our max_dispatch is `2^16-1 = 65,535`.
Our buffer is `134217728` bytes, it's full of `f32`s which are `4` bytes each, which means we should be able to store `2^25 = 33,554,432` 0s (or 1s) per buffer.
Our attempts to one-atise our `0`th buffer were conking out at `16,776,959`, to get past this we added an offset of that same ammount, and did two `add_one`s on two distinct indicies with an 'offset' of = `16,776,959`(the number we were making it up to). not a power of `2`, it's `1` away from `2^24`, infact we'd be expecting to get to `2^24 = 16,777,216`, so we're short at the ends, we know the reason we're counting up short, `256` elements.

So this actually demonstates a fundamental difficutly with the approach we backed ourselves into earlier, in that if we're going to do the indicie + an offset approach we need to ensure that we move in a stride such that we do not miss elements in the end.
TODO: img to show what's happening.

We've been thinking that every thread in our dispatch should do very little work (call `add_one`, maybe a handful of times) intsead of thinking how can we chunk the work we have amongst the threads we have.

So what is the number of floats in the _entire_ buffer over the MAX_DISPATCH size? 
Answer: 2^9 = 512.

A power of two must mean good things.

Let's work out a way for each of our threads to do itself, and somehow an additional 511 indicies worth of `add_one`.

The results first:
```
DEBUG wgpu_prac_v2                > >RUNTIME: 1768ms
    Pre Panic @ idx-2: 33_554_430, val: 1
    Pre Panic @ idx-1: 33_554_431, val: 1
    Panic     @ idx  : 33_554_432, val: 0
```
_finally_.

Also this has the added benefit of that we now have two numbers we can work with, the `workgroup_size` and this new `n` offset.
> our shader
```rust

const OFFSET:u32 = 256;

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_index = global_id.x * OFFSET;

    // Loop over the OFFSET indices that this thread is responsible for
    for (var i = 0u; i < OFFSET; i++) {
        let index = base_index + i;
        
        if (index < arrayLength(&v_indices)) {
            v_indices[index] = add_one(v_indices[index]);
        }
    }
}
```

Cool as the `workgroup_size.x` and the `OFFSET` are the same we can now change these to scale the relationship of how much work each axis of the workgroup is doing in a nice manner proportional to our work.

Now, let's check out that `1st` buffer, and finally get off this blasted `0th` one!

# mother is the first other
once again changing our `@binding(0)` to 1, can we see 1s in the array's indicies > 2^25?

We'll tweak our panicing printer to just bail at the first `1.0` (which we expect to be at our result's 2^25th index) and 0s, prior:
```rust
    results.iter().enumerate().for_each(|(e, v)| {
        if *v == 1.0 {
        /* Snipping prints */
        }
```
and, the results:
```
DEBUG wgpu_prac_v2                > >RUNTIME: 1796ms
    Pre Panic @ idx-2: 33_554_430, val: 0
    Pre Panic @ idx-1: 33_554_431, val: 0
    Panic     @ idx  : 33_554_432, val: 1
```
_works first time, 100% of the time, 1% of the time_.
TODO: sunglasses meme.

# sitrep;

It seems we're in the home stretch now, we just need to
- [] get ALL the buffers accesseded and have our shader `add_one` to them.
- [] have a guard to make sure the _final_ buffer doesn't go out of bounds with its writes (remember that buffer is smaller for us):
```sh
 DEBUG wgpu_prac_v2                > Created storage_buffer
 DEBUG wgpu_prac_v2                > bind_idx:0 buffer is 134217728b
 DEBUG wgpu_prac_v2                > bind_idx:1 buffer is 134217728b
 DEBUG wgpu_prac_v2                > bind_idx:2 buffer is 134217728b
 DEBUG wgpu_prac_v2                > bind_idx:3 buffer is 112742880b // THIS ONE IS SMALLER
```
- [] we need to ramp up to our _actual_ input size, the goal was 1GB of floats, strictly not less than that. 
- [] we need to address how portable this is, there's a few magic numbers at the moment we're not sure will work across different hardware configurations so we need to sort out a way to handle that. (what if the GPU's max dispatch is wildly different etc...)

# Brute forcing our way to success.
Ok, so at this point, we can up our data back to the target of 1GB's worth of 0.0f32s and, we can just do two _absolutely_ disgusting things like this:
1. Go ham on bindings
```rust
@group(0)
@binding(0)
var<storage, read_write> flat_buffer0: array<f32>;
@group(0)
@binding(1)
var<storage, read_write> flat_buffer1: array<f32>;
@group(0)
@binding(2)
var<storage, read_write> flat_buffer2: array<f32>;
@group(0)
@binding(3)
var<storage, read_write> flat_buffer3: array<f32>;

@group(0)
@binding(4)
var<storage, read_write> flat_buffer4: array<f32>;
@group(0)
@binding(5)
var<storage, read_write> flat_buffer5: array<f32>;
@group(0)
@binding(6)
var<storage, read_write> flat_buffer6: array<f32>;
@group(0)
@binding(7)
var<storage, read_write> flat_buffer7: array<f32>;

```
and, 2. Smush our existing index+offset loop like this
```rust
/* Snip */ 
for (var i = 0u; i < OFFSET; i++) {
        let index = base_index + i;
        
        if (index < arrayLength(&flat_buffer0)) {
            flat_buffer0[index] = add_one(flat_buffer0[index]);
            flat_buffer1[index] = add_one(flat_buffer1[index]);
            flat_buffer2[index] = add_one(flat_buffer2[index]);
            flat_buffer3[index] = add_one(flat_buffer3[index]);

            flat_buffer4[index] = add_one(flat_buffer4[index]);
            flat_buffer5[index] = add_one(flat_buffer5[index]);
            flat_buffer6[index] = add_one(flat_buffer6[index]);
            flat_buffer7[index] = add_one(flat_buffer7[index]);

        }
    }
/* Snip */ 
```

Then for the first time (everytime?):
```sh
 DEBUG wgpu_prac_v2 > buffers created, submitting job to GPU
 DEBUG wgpu_prac_v2 > Job submission complete.
 DEBUG wgpu_prac_v2 > Getting results...
 DEBUG wgpu_prac_v2 > TOTAL RUNTIME: 3507ms
```

... and if that solution wasn't so absolutely disgusting, we'd be done.
But I am not sure I can sleep at night with this many hard-coded things, that'll defnitely not work on someone else's machine, and are in general this hyper-tailored.

Surely it's possible to do _better_?

# Addressing all the buffers, in a nicer fashion

Allegedly (and [I found this](https://wgpu.rs/doc/wgpu/struct.Features.html#associatedconstant.STORAGE_RESOURCE_BINDING_ARRAY) by reading the wgpu sourcecode, not docs or GH issues, or tutorials, that is really the reason I sat down and begun cataloging all this...)

So let's try to use that and refactor our shader to support.

Refactoring the way we ask for the `wgpu::Device` to request the `wgpu:Feature` we want to use:
> From our `execute_gpu` function:
```rust
/*Snip*/
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: Features::STORAGE_RESOURCE_BINDING_ARRAY
                    | Features::BUFFER_BINDING_ARRAY,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
/*Snip*/
```

Then, according to the docs we can just declare our binding is an array of `T`, but it _looks like_ we need to declare said `T`, we'll not make a custom `T`, yet and just call the spade the spade it is.
```rust
@group(0)
@binding(0)
var<storage, read_write> buffers: array<array<f32>, 4>; // adding more magic here which i dislike :(
```
One issue with the above is that it is again giving us more magic numbers, we already know that we're running on half the intended input, so that `4` is soon to be an `8`. :(, but it can live in the fuck-it-bucket for now.

Luckily wgpu is starigt back to form with the errors:
with the above:
```sh
 ERROR wgpu::backend::wgpu_core    > Handling wgpu errors as fatal by default
thread 'main' panicked at /home/jer/.cargo/registry/src/index.crates.io-6f17d22bba15001f/wgpu-22.1.0/src/backend/wgpu_core.rs:3411:5:
wgpu error: Validation Error

Caused by:
  In Device::create_shader_module

Shader validation error:


      Type [2] '' is invalid
        Base type [1] for the array is invalid


note: run with RUST_BACKTRACE=1 environment variable to display a backtrace 
    
```
and if we change the `T` to be a custom type like this:
```rust
@group(0)
@binding(0)
var<storage, read_write> buffers: array<Buffer, 4>;

struct Buffer{
    inner: array<f32>
}
```
we get again, the same:
```sh
 ERROR wgpu_core::device::global   > Device::create_shader_module error:
Shader validation error:


 ERROR wgpu::backend::wgpu_core    > Handling wgpu errors as fatal by default
thread 'main' panicked at /home/jer/.cargo/registry/src/index.crates.io-6f17d22bba15001f/wgpu-22.1.0/src/backend/wgpu_core.rs:3411:5:
wgpu error: Validation Error

Caused by:
  In Device::create_shader_module

Shader validation error:


      Type [3] '' is invalid
        Base type [2] for the array is invalid


note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

```
Well the docs say it's native only on dx12 and vulkan, so let's check that we're on vulkan:
```sh
 DEBUG wgpu_prac_v2::debug_helpers > get_info : AdapterInfo { name: "NVIDIA TITAN RTX", vendor: 4318, device: 7682, device_type: DiscreteGpu, driver: "NVIDIA", driver_info: "555.58.02", backend: Vulkan }
```
We're good there...
And it **is** in our list of supported features...

# STORAGE_RESOURCE_BINDING_ARRAY
Let's look at the trace output from `wgpu` with 
```sh
RUST_LOG=wgpu=trace cargo run -r
```
which is long and verbose, but hopefully it can tell us something:
```sh
    Finished `release` profile [optimized] target(s) in 0.06s
     Running `target/release/wgpu-prac-v2`
 DEBUG wgpu_hal::vulkan::instance > Instance version: 0x40311b
 INFO  wgpu_hal::vulkan::instance > Debug utils not enabled: debug_utils_user_data not passed to Instance::from_raw
 DEBUG wgpu_hal::vulkan::instance > Enabling device properties2
 DEBUG wgpu_core::instance        > Instance::new: created Vulkan backend
 DEBUG wgpu_core::instance        > Instance::new: failed to create Gl backend: InstanceError { message: "unable to open libEGL", source: Some(Library(DlOpen { desc: "libEGL.so: cannot open shared object file: No such file or directory" })) }
 TRACE wgpu_core::instance        > Instance::request_adapter
 INFO  wgpu_core::instance        > Adapter Vulkan AdapterInfo { name: "NVIDIA TITAN RTX", vendor: 4318, device: 7682, device_type: DiscreteGpu, driver: "NVIDIA", driver_info: "555.58.02", backend: Vulkan }
 TRACE wgpu_core::storage         > User is inserting AdapterId(0,1,vk)
  TRACE wgpu_core::instance         > Adapter::drop Id(0,1,vk)
 TRACE wgpu_core::storage          > User is removing AdapterId(0,1,vk)
 TRACE wgpu_core::global           > Global::drop
 DEBUG wgpu_prac_v2                > numbers.len() = 128849016
 DEBUG wgpu_hal::vulkan::instance  > Instance version: 0x40311b
 INFO  wgpu_hal::vulkan::instance  > Debug utils not enabled: debug_utils_user_data not passed to Instance::from_raw
 DEBUG wgpu_hal::vulkan::instance  > Enabling device properties2
 DEBUG wgpu_core::instance         > Instance::new: created Vulkan backend
 DEBUG wgpu_core::instance         > Instance::new: failed to create Gl backend: InstanceError { message: "unable to open libEGL", source: Some(Library(DlOpen { desc: "libEGL.so: cannot open shared object file: No such file or directory" })) }
 TRACE wgpu_core::instance         > Instance::request_adapter
 INFO  wgpu_core::instance         > Adapter Vulkan AdapterInfo { name: "NVIDIA TITAN RTX", vendor: 4318, device: 7682, device_type: DiscreteGpu, driver: "NVIDIA", driver_info: "555.58.02", backend: Vulkan }
 TRACE wgpu_core::storage          > User is inserting AdapterId(0,1,vk)
 TRACE wgpu_core::instance         > Adapter::request_device
 DEBUG wgpu_hal::vulkan::adapter   > Supported extensions: ["VK_KHR_swapchain", "VK_KHR_swapchain_mutable_format", "VK_EXT_robustness2"]
 TRACE wgpu_core::instance         > Adapter::create_device
 TRACE wgpu_core::storage          > User is inserting DeviceId(0,1,vk)
 TRACE wgpu_core::instance         > Created Device Id(0,1,vk)
 TRACE wgpu_core::storage          > User is inserting QueueId(0,1,vk)
 TRACE wgpu_core::instance         > Created Queue Id(0,1,vk)
 ERROR wgpu_core::device::global   > Device::create_shader_module error:
Shader validation error:


 TRACE wgpu_core::storage          > User is inserting as error ShaderModuleId(0,1,vk)
 ERROR wgpu::backend::wgpu_core    > Handling wgpu errors as fatal by default
thread 'main' panicked at /home/jer/.cargo/registry/src/index.crates.io-6f17d22bba15001f/wgpu-22.1.0/src/backend/wgpu_core.rs:3411:5:
wgpu error: Validation Error

Caused by:
  In Device::create_shader_module

Shader validation error:


      Type [3] '' is invalid
        Base type [2] for the array is invalid


note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
 TRACE wgpu_core::global           > Global::drop
 TRACE wgpu_core::device::queue    > Drop Queue with '' label
 TRACE wgpu_core::device::resource > Drop Device with '' label
 TRACE wgpu_core::command::allocator > CommandAllocator::dispose encoders 0
```
Well that didn't tell us much, other than `wgpu` would have trouble using libGL, but as we want vulkan anyway that's useless.











