# Inference

Recorded single-stream generation speeds: 49 tok/s (`small`, Apple M-series CPU, KV cache) and ~53 tok/s (`large`, A40, KV cache); see training/char_small_cpu and training/fineweb_edu_large_a40. No continuous-batching throughput measurements exist; `forgeline.evaluation.performance.measure_generation` produces them on the current device.
