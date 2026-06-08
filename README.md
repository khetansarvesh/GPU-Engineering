# GPU-Engineering


### Single GPU Coding
- A piece of code running on a core of the GPU is called a kernel.
- It can be written at a high level in CUDA (for Nvidia GPUs) or Triton (For Nvidia / AMD / Intel GPUs)
- This is then compiled to Parallel Thread Execution (PTX), the low-level assembly used by NVIDIA GPUs.




### Multi GPU Coding
- How to [communicate](https://khetansarvesh.medium.com/communication-primitives-between-distributed-gpus-475592742a3d?postPublishedType=initial) between multiple GPUs
