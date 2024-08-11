@group(0)
@binding(0)
var<storage, read_write> v_indices: array<f32>;

fn add_one(n: f32) -> f32 {
    return n + 1.0;
}


const OFFSET:u32 = 4194240;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // let len = f32(arrayLength(&v_indices));
    // v_indices[global_id.x] = len; 33.5 Million...

    // v_indices[global_id.x] = add_one(v_indices[global_id.x]); 0s until 4.1 million

    // v_indices[global_id.x] = f32(global_id.x); // highest our threads go

    // Base index using global_id.x
    v_indices[global_id.x] = add_one(v_indices[global_id.x]);
    v_indices[global_id.x + OFFSET] = add_one(v_indices[global_id.x + OFFSET]);
    v_indices[global_id.x + OFFSET * 2u] = add_one(v_indices[global_id.x + OFFSET * 2u]);
    v_indices[global_id.x + OFFSET * 3u] = add_one(v_indices[global_id.x + OFFSET * 3u]);
    // up to here takes us up to 16.7 million

    
    // adding these you'd think may take us further, but it does not, we still tap out at 
    // 16_776_960
    // v_indices[global_id.x + OFFSET * 3u] = add_one(v_indices[global_id.x + OFFSET * 4u]);
    // v_indices[global_id.x + OFFSET * 3u] = add_one(v_indices[global_id.x + OFFSET * 5u]);
    // v_indices[global_id.x + OFFSET * 3u] = add_one(v_indices[global_id.x + OFFSET * 6u]);
    // v_indices[global_id.x + OFFSET * 3u] = add_one(v_indices[global_id.x + OFFSET * 7u]);
}
