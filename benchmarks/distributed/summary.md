# Distributed training

No multi-GPU measurements exist. Every recorded run used world_size = 1. DDP, FSDP, DeepSpeed ZeRO-2, tensor and pipeline parallelism are validated on CPU (configuration, topology, 1-process gloo execution) and have CUDA/multi-GPU tests that skip without hardware.
