# Project Description

## Title
GPU-Accelerated Phase-Based Audio Feature Extraction Using CUDA and cuFFT

## Problem and motivation
Audio analysis commonly converts a waveform into compact frame-level descriptors before
classification or recognition. Repeating windowing, FFT, phase processing, and temporal
feature calculations over many frames is a computationally parallel workload. This project
implements those stages on an NVIDIA GPU using CUDA C++ and cuFFT.

The project is motivated by digital signal processing and speaker/audio analysis. It focuses
on phase-derived information rather than treating the magnitude spectrum as the feature
representation.

## Technical approach
A 16 kHz waveform is split into 400-sample Hann-windowed frames with 160-sample overlap.
Each frame is zero-padded to 512 samples and transformed using batched cuFFT. For each
frequency bin, the temporal phase difference between adjacent frames is corrected for the
expected phase advance. Sine and cosine functions of the residual phase are projected onto
a cosine basis to produce 20 static coefficients. First- and second-order temporal
differences expand this representation to 60 dimensions.

A low-energy RMS gate is used only to suppress unstable/silent frames. It is not used to
construct the features.

## GPU work
The CUDA implementation includes separate kernels for frame preparation, phase-derived
feature extraction, and temporal delta calculations. The FFT is performed by cuFFT. A
single-threaded CPU implementation provides a reference for numerical comparison.

## Evaluation
The project evaluates:
- correctness through CPU/GPU feature comparison
- output validity (finite values and expected 60-dimensional layout)
- runtime over different audio durations
- throughput and the effect of workload size on GPU utilization

The final report must include measurements collected on the actual GPU used. No performance
numbers are assumed in advance.

## Lessons and challenges to discuss after running
- Mapping frame/bin/feature indices to CUDA threads
- Managing host/device memory and cuFFT plans
- Handling phase wrapping and expected phase advance
- Choosing an energy threshold without using magnitude as the feature itself
- Understanding GPU launch overhead for small workloads
- Comparing numerical results across CPU and GPU floating-point implementations

## Scope
This is a feature-extraction capstone, not a complete speaker verification model. A natural
next step is integrating the output with a GMM-UBM / MAP / log-likelihood-ratio pipeline.
