@group(0)
@binding(0)
var<storage, read_write> v_indices: array<f32>;

fn add_one(n: f32) -> f32 {
    return n + 1.0;
}

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let len = f32(arrayLength(&v_indices));
    // v_indices[global_id.x] = add_one(v_indices[global_id.x]);
    v_indices[global_id.x] = len;
    // v_indicies[global_id.x] =len;

}
