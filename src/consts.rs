/// Max size of a buffer MY gpu can take...
/*
Caused by:
  In Device::create_buffer
    Buffer size 1073741824 is greater than the maximum buffer size (268435456)
*/

// Caused by:
//   In Device::create_bind_group, label = 'Bind Group 0'
//     Buffer binding 0 range 268435456 exceeds `max_*_buffer_binding_size` limit 134217728
pub const RTX_TITAN_MAX_BUFFER_SIZE: u64 = 134_217_728; // 1 << 27
pub const RTX_TITAN_MAX_BIND_GROUPS: u64 = 8;
pub const RTX_TITAN_MAX_BINDS_PER_GROUP: u64 = 1_000;
pub const MAX_DISPATCH_SIZE: u32 = 65_535; // (1 << 16) -1
