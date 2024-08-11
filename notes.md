Hello,

(There is a simple project here, which hopefully explains what's going on here).

I am experimenting with moving some CUDA to using wgpu, but when working on something I got stuck on how to ship large ammounts of data over to the GPU, 

Here's a simple project that I hope illustrates where I'm stuck:
- create `1/2` a gig of `f32`s, all initialised to `0.0`
- split them across multiple storage buffers
- setup staging buffers to get the values back
- a shader that should increment them all by `1.0`
- some checks to make sure what comes back is indeed incremented properly

Issues:
- at index idx: `4194240`, the value is `0.0`, when we're expecting all `1.0`s.
- this number is not a multiple of our buffer sizes/lengths etc so I'm confused about how we write `1.0`s up until that point.

I recall (maybe?) seeing something about a compilation option or flag one can set to ensure that any _n_ number of bindings declared CPU side will always appear as a single contigious bind on the GPU side, so you'd always have:
	```rust
	@group(0)
	@binding(0)
	var<storage, read_write> v_indices: array<f32>; 
	```
kinda thing.

What am I missing?
