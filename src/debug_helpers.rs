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
