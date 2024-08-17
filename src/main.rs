use wgpu_prac_v2::run;

pub fn main() {
    pretty_env_logger::init();

    // pollster::block_on(wgpu_prac_v2::debug_helpers::debug_gpu_info());

    pollster::block_on(run());
}
