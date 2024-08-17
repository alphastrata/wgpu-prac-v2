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

// Function to add one to a given value
fn add_one(n: f32) -> f32 {
    return n + 1.0;
}

const OFFSET:u32 = 256u;

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_index = global_id.x * OFFSET;
    let len = arrayLength(&flat_buffer0);


// working on one-buffer
    // Loop over the OFFSET indices that this thread is responsible for
    for (var i = 0u; i < OFFSET; i++) {
        let index = base_index + i;
        
        if (index < arrayLength(&flat_buffer0)) {
            // flat_buffer[index] = add_one(flat_buffer[index]);
            flat_buffer0[index] = add_one(flat_buffer0[index]);
            flat_buffer1[index] = add_one(flat_buffer1[index]);
            flat_buffer2[index] = add_one(flat_buffer2[index]);
            flat_buffer3[index] = add_one(flat_buffer3[index]); // we know this is gonna write out of bounds...

        }
    }
}
