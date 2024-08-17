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







// struct MyBuffer{
//     inner: array<f32>,
// }

// var<storage, read_write> buffer_array: array<MyBuffer, 10>;


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
