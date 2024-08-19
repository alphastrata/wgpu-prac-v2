use wgpu;

pub async fn debug_gpu_info() {
    let instance = wgpu::Instance::default();

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();

    log::debug!("features : {:?}", adapter.features());
    log::debug!("get_info : {:?}", adapter.get_info());
    log::debug!("limits   : {:?}", adapter.limits());
}

#[inline(always)]
fn add_one_to(v: &[f32]) -> Vec<f32> {
    v.iter().map(|n| n + 1.0).collect()
}

pub fn run_cpu_sanity_check(numbers: &[f32]) -> Vec<f32> {
    let t1 = std::time::Instant::now();
    let results = add_one_to(numbers);
    log::debug!("CPU RUN: {}ms", t1.elapsed().as_millis());

    results
}
