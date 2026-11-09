# 5–10 Minute Capstone Presentation Outline

## 0:00–0:45 — Problem and motivation
- Introduce audio feature extraction and why frame-level processing is parallel.
- Explain the goal: implement a phase-derived audio feature pipeline on an NVIDIA GPU.

## 0:45–1:45 — Signal-processing method
- 16 kHz waveform
- 400-sample Hann window, 160-sample hop, 512-point FFT
- Temporal phase residual:
  `wrap(angle(S_t * conj(S_(t-1))) - 2*pi*k*H/N)`
- 20 static coefficients + 20 deltas + 20 delta-deltas = 60 features

## 1:45–3:30 — CUDA architecture
- Show frame preparation kernel.
- Explain batched cuFFT.
- Show phase cepstral extraction and temporal-difference kernels.
- Explain host/device transfers and output CSV.

## 3:30–5:00 — Demonstration
- Generate a synthetic WAV.
- Run the GPU backend.
- Run CPU/GPU comparison.
- Open the CSV and feature heatmap.
- Show `nvidia-smi` to identify the GPU.

## 5:00–7:00 — Results
- Present actual hardware and input durations.
- Compare CPU and GPU runtimes over multiple input sizes.
- Report maximum absolute feature difference.
- Explain any cases where the GPU is slower due to launch/setup overhead.

## 7:00–8:30 — Challenges and lessons
- Phase wrapping and phase advance correction
- cuFFT integration and memory management
- Parallel indexing
- Numerical differences and performance measurement

## 8:30–9:30 — Future work and conclusion
- Multi-file batching, better VAD, kernel tuning
- Integration with a GMM-UBM speaker-verification system
- Summarize what was learned about GPU acceleration
