# GPU-Accelerated Phase-Based Audio Feature Extraction (CUDA)

## 1. Project overview

This project implements a GPU-accelerated audio feature-extraction pipeline in CUDA C++.
It transforms 16 kHz PCM WAV audio into a sequence of 60-dimensional, phase-derived
features. The goal is to demonstrate how an audio signal-processing workload can be
decomposed into parallel GPU operations and benchmarked against a single-threaded CPU
reference implementation.

This is an audio signal-processing capstone, not a complete speaker-verification system.
The output features can be used by a downstream speaker-recognition or audio-classification
model, but no speaker labels or evaluation labels are used here.

## 2. Why phase-derived features?

Magnitude spectra are commonly used in audio processing, but this project focuses on
temporal phase change. For adjacent STFT frames, the phase residual for frequency bin `k`
and frame `t` is:

`delta_phi(k,t) = wrap(angle(S(k,t) * conj(S(k,t-1))) - 2*pi*k*H/N)`

where `H=160` is the hop length and `N=512` is the FFT size. The expected phase advance
is removed, leaving a residual phase change. The implementation derives sine and cosine
representations of this residual and applies a cosine basis transform.

Magnitude/energy is used only for a simple voice-activity/stability gate. It is not used
as the feature representation.

## 3. Feature layout

The STFT uses a 400-sample Hann window, 160-sample hop, and 512-point FFT.

Each frame produces:
- 10 sine phase-cepstral coefficients
- 10 cosine phase-cepstral coefficients
- 20 first-order temporal delta coefficients
- 20 second-order delta-delta coefficients

Total: **60 features per frame**.

The first frame has no previous frame and is set to zero. Low-energy frames are zeroed by
the simple RMS gate (`0.005`).

## 4. GPU implementation

The GPU pipeline uses:
1. CUDA kernel `prepareFrames`: creates windowed, zero-padded frames.
2. NVIDIA cuFFT: performs batched 512-point complex FFTs.
3. CUDA kernel `extractPhaseCepstra`: computes phase residuals and the 20 base features.
4. CUDA kernel `buildDeltas`: computes first-order temporal differences.
5. CUDA kernel `buildDeltaDeltas`: computes second-order temporal differences.

The frame and feature dimensions are parallelized across CUDA threads. cuFFT performs the
FFT stage on the GPU. The CPU reference uses a single-threaded radix-2 FFT and the same
feature definitions to provide a numerical comparison.

## 5. Requirements

- NVIDIA GPU supported by the installed CUDA Toolkit
- CUDA Toolkit with `nvcc` and cuFFT (`cufft.h`, `libcufft`)
- GNU Make
- Python 3 for generating the demonstration WAV
- Optional Python packages for plots: `numpy`, `matplotlib`

The Makefile defaults to compute capability 7.5. If your GPU has a different compute
capability, override `CUDA_ARCH`. For example, for compute capability 8.6:

```bash
make clean
make CUDA_ARCH="-gencode arch=compute_86,code=sm_86"
```

For a newer GPU, use the architecture supported by your installed CUDA Toolkit.
Check your GPU with `nvidia-smi` and CUDA compiler with `nvcc --version`.

## 6. Build and run

```bash
make
python3 scripts/generate_demo_wav.py --output data/demo.wav
mkdir -p output
./build/phase_audio_cuda --input data/demo.wav --output output/demo_gpu.csv --backend gpu
```

Run CPU and GPU and compare features:

```bash
./build/phase_audio_cuda --input data/demo.wav --output output/demo_both.csv --backend both
```

Run CPU only:

```bash
./build/phase_audio_cuda --input data/demo.wav --output output/demo_cpu.csv --backend cpu
```

CLI arguments:
- `--input`: required PCM16 WAV input
- `--output`: output CSV path (default `features.csv`)
- `--backend`: `gpu`, `cpu`, or `both`
- `--help`: print usage

Input constraints: mono or multichannel PCM 16-bit WAV at 16 kHz. Multichannel input is
converted to mono by averaging channels. The code deliberately rejects other sample rates
rather than silently resampling.

## 7. Output format

The CSV has one row per analysis frame:
- `frame`
- `feature_0` through `feature_59`

The feature matrix is frame-major. The first 20 columns are static phase cepstra, the next
20 are deltas, and the final 20 are delta-deltas.

Validate and visualize:

```bash
python3 -m pip install numpy matplotlib
python3 scripts/inspect_features.py output/demo_gpu.csv --plot output/features.png
```

## 8. Experiments to include in the final report

Run at least these experiments on the actual CUDA lab GPU:
1. 1-second synthetic signal
2. 4-second synthetic signal
3. 15-second synthetic signal
4. Several real 16 kHz PCM16 WAV recordings, if permitted by the course/data license

For each input, record number of samples, number of frames, GPU event time, CPU time, maximum
absolute CPU/GPU feature difference, and CPU/GPU time ratio. Run each case at least five
times and report the median. Separate first-run setup overhead from steady-state timing if
you add that measurement.

Do not invent speedup values. GPU launch overhead can make the GPU slower for very short
signals. Discuss this honestly and identify the input size at which GPU processing becomes
beneficial on the hardware used.

## 9. Proof of execution checklist

Save the following under `proof/` before submission:
- terminal output for a successful GPU run
- terminal output for `--backend both`
- generated feature CSV files for multiple input durations
- feature heatmap PNG
- `nvidia-smi` output showing the GPU used
- a small results table with hardware, input length, frames, CPU ms, GPU ms, max error

The included repository does not claim execution evidence before you run it on your CUDA
environment. Capture genuine output from your own lab run.

## 10. Limitations and next steps

- The VAD gate is a simple RMS threshold, not a production-grade speech detector.
- The WAV reader supports PCM16 only and requires 16 kHz input.
- The CPU/GPU comparison is intended as a correctness/performance experiment, not a
  comprehensive benchmark suite.
- The current program extracts features; it does not train a GMM-UBM or perform speaker
  verification.

Possible extensions: implement a batched multi-file mode, tune block sizes, use pinned host
memory, add a proper VAD, and feed the 60-D features into a GMM-UBM speaker-verification
pipeline.

## 11. Repository structure

```text
cuda_phase_audio_capstone/
├── Makefile
├── README.md
├── src/main.cu
├── scripts/generate_demo_wav.py
├── scripts/inspect_features.py
├── docs/project_description.md
├── docs/presentation_outline.md
├── docs/experiment_log_template.csv
├── data/                 # generated demo WAV goes here
└── proof/                # add genuine execution evidence here
```
