pub fn gigs_of_zeroed_f32s(n: f32) -> Vec<f32> {
    let bytes_per_gb = 1024 * 1024 * 1024; // 1 GB in bytes
    let bytes_per_f32 = std::mem::size_of::<f32>(); // Size of f32 in bytes
    let total_bytes = (n * bytes_per_gb as f32) as usize; // Total bytes for n gigabytes
    let elements = total_bytes / bytes_per_f32; // Number of f32 elements that fit in the specified bytes

    vec![0.0; elements] // Creates a vector of zeroed f32s of the calculated size
}
