# GPU-Engineering

So different companies have differnt GPUs and differnt programming languages to give instructions to these GPUs.

| Feature               | NVIDIA             | AMD              | Intel             |
|-----------------------|--------------------|------------------|-------------------|
| Kernel Language       | CUDA C++           | HIP C++          | SYCL / DPC++      |
| Compiler              | `nvcc`             | `hipcc`          | `icpx`            |
| Runtime               | CUDA               | ROCm             | oneAPI            |
| Math Libraries        | cuBLAS, cuDNN, NCCL| rocBLAS, MIOpen, RCCL | oneMKL, oneDNN, oneCCL |
| Inference Optimizer   | TensorRT-LLM       | (none)           | OpenVINO          |
| Compiles To           | PTX → SASS         | GCN / CDNA       | SPIR-V            |
| Hardware              | H100, B200, A100   | MI300X, MI350 (soon) | Gaudi 3, Arc, Xeon |

Hence there was a need of one hardware agnostic language. OpenAI developed Triton !

# $\color{cyan}{Single\ GPU\  Coding }$

- A piece of code running on a core of the GPU is called a kernel.
- It can be written at a high level in CUDA (for Nvidia GPUs) or Triton (For Nvidia / AMD / Intel GPUs)
- This is then compiled to Parallel Thread Execution (PTX), the low-level assembly used by NVIDIA GPUs.

This blog [here](https://www.aleksagordic.com/blog/matmul) gives a very good understanding of how to write a kernel that can do matmul in a H100 GPU

Important things to learn : 
- GPU memory hierarchy — Learn HBM → L2 → shared memory → registers. Understand why FlashAttention is fast (it tiles into SRAM to avoid HBM round-trips).
- CUDA basics — Read a few kernel implementations. Understand thread blocks, warps, coalesced access, Tensor Cores. You don't need to write FlashAttention, but you need to read kernel code.

# $\color{cyan}{Multi\ GPU\  Coding }$
- How to [communicate](https://khetansarvesh.medium.com/communication-primitives-between-distributed-gpus-475592742a3d?postPublishedType=initial) between multiple GPUs
