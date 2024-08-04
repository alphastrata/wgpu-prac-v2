pub fn gigs_of_zeroed_f32s(n: usize) -> Vec<f32> {
    let bytes_per_gb = 1024 * 1024 * 1024; // 1 GB in bytes
    let bytes_per_f32 = std::mem::size_of::<f32>(); // Size of f32 in bytes
    let elements = n * bytes_per_gb / bytes_per_f32;

    vec![0.0; elements]
}
