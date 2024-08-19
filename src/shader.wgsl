// @group(0)
// @binding(0)
// var<storage, read_write> flat_buffer0: array<f32>;
// @group(0)
// @binding(1)
// var<storage, read_write> flat_buffer1: array<f32>;
// @group(0)
// @binding(2)
// var<storage, read_write> flat_buffer2: array<f32>;
// @group(0)
// @binding(3)
// var<storage, read_write> flat_buffer3: array<f32>;

// @group(0)
// @binding(4)
// var<storage, read_write> flat_buffer4: array<f32>;
// @group(0)
// @binding(5)
// var<storage, read_write> flat_buffer5: array<f32>;
// @group(0)
// @binding(6)
// var<storage, read_write> flat_buffer6: array<f32>;
// @group(0)
// @binding(7)
// var<storage, read_write> flat_buffer7: array<f32>;

struct OurBuffer {
    inner: array<f32, 33554432>,
}

@group(0) @binding(0)
var<storage, read_write> all_buffers: array<OurBuffer, 8>;

const OFFSET: u32 = 256u;
const BUFF_LENGTH: u32 = 33554432u;
const NUM_BUFFERS: u32 = 8u;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_index = global_id.x * OFFSET;

    // Loop over the OFFSET indices that this thread is responsible for
    for (var j = 0u; j < NUM_BUFFERS; j++) {
        for (var i = 0u; i < OFFSET; i++) {
            let index = base_index + i;

            if (index < BUFF_LENGTH) {
                all_buffers[j].inner[index] = add_one(all_buffers[j].inner[index]);
            }
        }
    }
}


// Function to add one to a given value
fn add_one(n: f32) -> f32 {
    return n + 1.0;
}
