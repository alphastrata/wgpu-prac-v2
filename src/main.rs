use wgpu_prac_v2::{data_utils::gigs_of_zeroed_f32s, run};

pub fn main() {
    pretty_env_logger::init();

    // let floats = gigs_of_zeroed_f32s(1.0);
    // log::debug!("floats len  : {}", floats.len());
    // log::debug!(
    //     "floats bytes: {}",
    //     std::mem::size_of::<f32>() * floats.len()
    // );

    // pollster::block_on(debug_gpu_info());

    pollster::block_on(run());
}
