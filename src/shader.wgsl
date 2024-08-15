// @group(0)
// @binding(0)
// var<storage, read_write> v_indices: array<f32>;

@group(0)
@binding(0)
var<storage, read_write> buffers: array<Buffer, 4>;

struct Buffer{
    inner: array<f32>
}

fn add_one(n: f32) -> f32 {
    return n + 1.0;
}

const OFFSET: u32 = 256;

// fn get_element(index: u32) -> f32 {
//     let buffer_size = arrayLength(&buffers[0]);
//     let buffer_index = index / buffer_size;  // Determine which buffer to use
//     let local_index = index % buffer_size;   // Determine the index within that buffer
//     return buffers[buffer_index][local_index];
// }

// fn set_element(index: u32, value: f32) {
//     let buffer_size = arrayLength(&buffers[0]);
//     let buffer_index = index / buffer_size;
//     let local_index = index % buffer_size;
//     buffers[buffer_index][local_index] = value;
// }

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_index = global_id.x * OFFSET;

    // Loop over the OFFSET indices that this thread is responsible for
    for (var i = 0u; i < OFFSET; i++) {
        let index = base_index + i;
        
        // Access and modify the element across the contiguous buffer
        // let value = get_element(index);
        // set_element(index, add_one(value));
    }
}










// const OFFSET:u32 = 256;

// @compute
// @workgroup_size(256, 1, 1)
// fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
//     let base_index = global_id.x * OFFSET;

//     // Loop over the OFFSET indices that this thread is responsible for
//     for (var i = 0u; i < OFFSET; i++) {
//         let index = base_index + i;
        
//         if (index < arrayLength(&v_indices)) {
//             v_indices[index] = add_one(v_indices[index]);
//         }
//     }
// }


// const OFFSET:u32 = 4194240;
// const OFFSET:u32 = 16776959;
// const OFFSET:u32 = 16777216;

// @compute
// // @workgroup_size(64)
// @workgroup_size(256, 1,1)
// fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
//     // let len = f32(arrayLength(&v_indices));
//     // v_indices[global_id.x] = len; //33.5 Million...
//     // v_indices[global_id.x + OFFSET] = len; //33.5 Million...

//     v_indices[global_id.x] = add_one(v_indices[global_id.x]); 
//     v_indices[global_id.x + OFFSET] = add_one(v_indices[global_id.x + OFFSET]); 


//     // v_indices[global_id.x] = add_one(v_indices[global_id.x]); 0s until 4.1 million
    
//     // v_indices[global_id.x] = f32(global_id.x); // highest our threads go = 4,194,239, makes sense given the above

//     // Base index using global_id.x
//     // v0_indices[global_id.x] = add_one(v0_indices[global_id.x]);
//     // v1_indices[global_id.x] = add_one(v1_indices[global_id.x]);
//     // v2_indices[global_id.x] = add_one(v2_indices[global_id.x]);
//     // v3_indices[global_id.x] = add_one(v3_indices[global_id.x]);

//     // v_indices[global_id.x + OFFSET] = add_one(v_indices[global_id.x + OFFSET]);
//     // v_indices[global_id.x + OFFSET * 1u] = add_one(v_indices[global_id.x + OFFSET * 1u]);
//     // v_indices[global_id.x + OFFSET * 2u] = add_one(v_indices[global_id.x + OFFSET * 2u]);
//     // v_indices[global_id.x + OFFSET * 3u] = add_one(v_indices[global_id.x + OFFSET * 3u]);
//     // up to here takes us up to 16.7 million

    
//     // adding these you'd think may take us further, but it does not, we still tap out at 
//     // 16_776_960
//     // v_indices[global_id.x + OFFSET * 3u] = add_one(v_indices[global_id.x + OFFSET * 4u]);
//     // v_indices[global_id.x + OFFSET * 3u] = add_one(v_indices[global_id.x + OFFSET * 5u]);
//     // v_indices[global_id.x + OFFSET * 3u] = add_one(v_indices[global_id.x + OFFSET * 6u]);
//     // v_indices[global_id.x + OFFSET * 3u] = add_one(v_indices[global_id.x + OFFSET * 7u]);
// }
