#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kSampleRate = 16000;
constexpr int kWindow = 400;
constexpr int kHop = 160;
constexpr int kFftSize = 512;
constexpr int kBins = kFftSize / 2 + 1;
constexpr int kCepstra = 10;
constexpr int kBaseDims = 2 * kCepstra;
constexpr int kFeatureDims = 3 * kBaseDims;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kVadRms = 0.005f;

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) throw std::runtime_error( \
      std::string("CUDA error: ") + cudaGetErrorString(e)); \
} while (0)

#define CUFFT_CHECK(call) do { \
  cufftResult e = (call); \
  if (e != CUFFT_SUCCESS) throw std::runtime_error( \
      "cuFFT call failed with code " + std::to_string(static_cast<int>(e))); \
} while (0)

struct WavData {
  int sample_rate = 0;
  std::vector<float> samples;
};

uint16_t readU16(std::istream& in) {
  uint8_t b[2]; in.read(reinterpret_cast<char*>(b), 2);
  if (!in) throw std::runtime_error("Unexpected end of WAV file");
  return static_cast<uint16_t>(b[0] | (b[1] << 8));
}
uint32_t readU32(std::istream& in) {
  uint8_t b[4]; in.read(reinterpret_cast<char*>(b), 4);
  if (!in) throw std::runtime_error("Unexpected end of WAV file");
  return static_cast<uint32_t>(b[0] | (b[1] << 8) | (b[2] << 16) |
                               (static_cast<uint32_t>(b[3]) << 24));
}

WavData loadWav(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open input WAV: " + path);
  char riff[4], wave[4];
  in.read(riff, 4); (void)readU32(in); in.read(wave, 4);
  if (!in || std::string(riff, 4) != "RIFF" ||
      std::string(wave, 4) != "WAVE")
    throw std::runtime_error("Input is not a RIFF/WAVE file");

  uint16_t format = 0, channels = 0, bits = 0;
  uint32_t sampleRate = 0;
  std::vector<uint8_t> pcm;
  while (in && !in.eof()) {
    char id[4]; in.read(id, 4);
    if (!in) break;
    uint32_t size = readU32(in);
    std::string chunk(id, 4);
    if (chunk == "fmt ") {
      format = readU16(in); channels = readU16(in);
      sampleRate = readU32(in);
      (void)readU32(in); (void)readU16(in);
      bits = readU16(in);
      if (size > 16) in.seekg(size - 16, std::ios::cur);
    } else if (chunk == "data") {
      pcm.resize(size);
      in.read(reinterpret_cast<char*>(pcm.data()), size);
    } else {
      in.seekg(size, std::ios::cur);
    }
    if (size & 1U) in.seekg(1, std::ios::cur);
  }
  if (format != 1 || (bits != 16) || channels == 0)
    throw std::runtime_error("Only PCM 16-bit WAV files are supported");
  if (sampleRate != kSampleRate)
    throw std::runtime_error("WAV must be 16 kHz; input is " +
                             std::to_string(sampleRate) + " Hz");
  const size_t bytesPerSample = 2;
  const size_t frameBytes = bytesPerSample * channels;
  if (pcm.size() < frameBytes)
    throw std::runtime_error("WAV contains no audio samples");

  WavData wav; wav.sample_rate = static_cast<int>(sampleRate);
  const size_t count = pcm.size() / frameBytes;
  wav.samples.resize(count);
  for (size_t i = 0; i < count; ++i) {
    float sum = 0.0f;
    for (uint16_t c = 0; c < channels; ++c) {
      const size_t p = i * frameBytes + c * 2;
      int16_t value = static_cast<int16_t>(
          static_cast<uint16_t>(pcm[p] | (pcm[p + 1] << 8)));
      sum += static_cast<float>(value) / 32768.0f;
    }
    wav.samples[i] = sum / channels;
  }
  return wav;
}

int frameCount(size_t n) {
  if (n <= kWindow) return 1;
  return 1 + static_cast<int>((n - kWindow + kHop - 1) / kHop);
}

__device__ float wrapPhase(float x) {
  return atan2f(sinf(x), cosf(x));
}

__global__ void prepareFrames(const float* signal, int n, int frames,
                              cufftComplex* spectrumInput) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = frames * kFftSize;
  if (idx >= total) return;
  int frame = idx / kFftSize;
  int bin = idx % kFftSize;
  int sampleIndex = frame * kHop + bin;
  float value = 0.0f;
  if (bin < kWindow && sampleIndex < n) {
    float hann = 0.5f - 0.5f * cosf(2.0f * kPi * bin / (kWindow - 1));
    value = signal[sampleIndex] * hann;
  }
  spectrumInput[idx].x = value;
  spectrumInput[idx].y = 0.0f;
}

__global__ void extractPhaseCepstra(const float* signal, int n, int frames,
                                    const cufftComplex* spectrum,
                                    float* baseFeatures) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = frames * kBaseDims;
  if (idx >= total) return;
  int frame = idx / kBaseDims;
  int dim = idx % kBaseDims;
  int cep = dim % kCepstra;
  bool useSine = dim < kCepstra;

  // Magnitude/time-domain energy is used only as a voice-activity gate.
  float energy = 0.0f;
  int start = frame * kHop;
  for (int i = 0; i < kWindow && start + i < n; ++i) {
    float x = signal[start + i];
    energy += x * x;
  }
  float rms = sqrtf(energy / kWindow);
  if (rms < kVadRms || frame == 0) {
    baseFeatures[idx] = 0.0f;
    return;
  }

  float sum = 0.0f;
  const cufftComplex* current = spectrum + frame * kFftSize;
  const cufftComplex* previous = spectrum + (frame - 1) * kFftSize;
  for (int k = 0; k < kBins; ++k) {
    float re = current[k].x * previous[k].x +
               current[k].y * previous[k].y;
    float im = current[k].y * previous[k].x -
               current[k].x * previous[k].y;
    float phase = atan2f(im, re);
    float expected = 2.0f * kPi * k * kHop / kFftSize;
    float residual = wrapPhase(phase - expected);
    float basis = cosf(kPi * cep * (k + 0.5f) / kBins);
    sum += (useSine ? sinf(residual) : cosf(residual)) * basis;
  }
  baseFeatures[idx] = sum / kBins;
}

__global__ void buildDeltas(const float* base, int frames, float* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = frames * kBaseDims;
  if (idx >= total) return;
  int frame = idx / kBaseDims;
  int dim = idx % kBaseDims;
  int left = max(frame - 1, 0);
  int right = min(frame + 1, frames - 1);
  float delta = 0.5f * (base[right * kBaseDims + dim] -
                         base[left * kBaseDims + dim]);
  output[frame * kFeatureDims + dim] = base[idx];
  output[frame * kFeatureDims + kBaseDims + dim] = delta;
}

__global__ void buildDeltaDeltas(const float* features, int frames,
                                 float* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = frames * kBaseDims;
  if (idx >= total) return;
  int frame = idx / kBaseDims;
  int dim = idx % kBaseDims;
  int left = max(frame - 1, 0);
  int right = min(frame + 1, frames - 1);
  float deltaDelta = 0.5f *
      (features[right * kFeatureDims + kBaseDims + dim] -
       features[left * kFeatureDims + kBaseDims + dim]);
  output[frame * kFeatureDims + 2 * kBaseDims + dim] = deltaDelta;
}

std::vector<float> runGpu(const std::vector<float>& samples, int frames,
                          double& elapsedMs) {
  const int n = static_cast<int>(samples.size());
  float* dSignal = nullptr;
  cufftComplex* dSpectrum = nullptr;
  float* dBase = nullptr;
  float* dFeatures = nullptr;
  cudaEvent_t start{}, stop{};
  try {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSignal), n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSpectrum), frames * kFftSize * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dBase), frames * kBaseDims * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dFeatures), frames * kFeatureDims * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dSignal, samples.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, kFftSize, CUFFT_C2C, frames));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int threads = 256;
    prepareFrames<<<(frames * kFftSize + threads - 1) / threads, threads>>>(
        dSignal, n, frames, dSpectrum);
    CUDA_CHECK(cudaGetLastError());
    CUFFT_CHECK(cufftExecC2C(plan, dSpectrum, dSpectrum, CUFFT_FORWARD));
    extractPhaseCepstra<<<(frames * kBaseDims + threads - 1) / threads,
                          threads>>>(dSignal, n, frames, dSpectrum, dBase);
    CUDA_CHECK(cudaGetLastError());
    buildDeltas<<<(frames * kBaseDims + threads - 1) / threads, threads>>>(
        dBase, frames, dFeatures);
    CUDA_CHECK(cudaGetLastError());
    buildDeltaDeltas<<<(frames * kBaseDims + threads - 1) / threads, threads>>>(
        dFeatures, frames, dFeatures);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    elapsedMs = ms;

    std::vector<float> result(frames * kFeatureDims);
    CUDA_CHECK(cudaMemcpy(result.data(), dFeatures,
                          result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(dSignal); cudaFree(dSpectrum); cudaFree(dBase); cudaFree(dFeatures);
    return result;
  } catch (...) {
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    if (dSignal) cudaFree(dSignal);
    if (dSpectrum) cudaFree(dSpectrum);
    if (dBase) cudaFree(dBase);
    if (dFeatures) cudaFree(dFeatures);
    throw;
  }
}

// CPU radix-2 FFT reference. This is a single-threaded reference, not a
// multithreaded substitute for the CUDA implementation.
void cpuFft(std::vector<std::complex<float>>& a) {
  const int n = static_cast<int>(a.size());
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (int len = 2; len <= n; len <<= 1) {
    float angle = -2.0f * kPi / len;
    std::complex<float> root(cosf(angle), sinf(angle));
    for (int i = 0; i < n; i += len) {
      std::complex<float> w(1.0f, 0.0f);
      for (int j = 0; j < len / 2; ++j) {
        auto u = a[i + j];
        auto v = a[i + j + len / 2] * w;
        a[i + j] = u + v;
        a[i + j + len / 2] = u - v;
        w *= root;
      }
    }
  }
}

std::vector<float> runCpu(const std::vector<float>& samples, int frames,
                          double& elapsedMs) {
  const int n = static_cast<int>(samples.size());
  std::vector<std::complex<float>> spectra(frames * kFftSize);
  std::vector<float> base(frames * kBaseDims, 0.0f);
  std::vector<float> features(frames * kFeatureDims, 0.0f);
  auto begin = std::chrono::steady_clock::now();

  for (int f = 0; f < frames; ++f) {
    std::vector<std::complex<float>> frame(kFftSize, {0.0f, 0.0f});
    for (int i = 0; i < kWindow && f * kHop + i < n; ++i) {
      float hann = 0.5f - 0.5f * cosf(2.0f * kPi * i / (kWindow - 1));
      frame[i] = {samples[f * kHop + i] * hann, 0.0f};
    }
    cpuFft(frame);
    std::copy(frame.begin(), frame.end(), spectra.begin() + f * kFftSize);
  }

  for (int f = 1; f < frames; ++f) {
    float energy = 0.0f;
    for (int i = 0; i < kWindow && f * kHop + i < n; ++i) {
      float x = samples[f * kHop + i]; energy += x * x;
    }
    if (sqrtf(energy / kWindow) < kVadRms) continue;
    for (int d = 0; d < kBaseDims; ++d) {
      int c = d % kCepstra;
      bool sine = d < kCepstra;
      float sum = 0.0f;
      for (int k = 0; k < kBins; ++k) {
        auto cur = spectra[f * kFftSize + k];
        auto prev = spectra[(f - 1) * kFftSize + k];
        float phase = std::arg(cur * std::conj(prev));
        float residual = atan2f(sinf(phase - 2.0f * kPi * k * kHop /
                                    kFftSize),
                                cosf(phase - 2.0f * kPi * k * kHop /
                                    kFftSize));
        float basis = cosf(kPi * c * (k + 0.5f) / kBins);
        sum += (sine ? sinf(residual) : cosf(residual)) * basis;
      }
      base[f * kBaseDims + d] = sum / kBins;
    }
  }
  for (int f = 0; f < frames; ++f) {
    int l = std::max(f - 1, 0), r = std::min(f + 1, frames - 1);
    for (int d = 0; d < kBaseDims; ++d) {
      float delta = 0.5f * (base[r * kBaseDims + d] -
                            base[l * kBaseDims + d]);
      features[f * kFeatureDims + d] = base[f * kBaseDims + d];
      features[f * kFeatureDims + kBaseDims + d] = delta;
    }
  }
  for (int f = 0; f < frames; ++f) {
    int l = std::max(f - 1, 0), r = std::min(f + 1, frames - 1);
    for (int d = 0; d < kBaseDims; ++d)
      features[f * kFeatureDims + 2 * kBaseDims + d] =
          0.5f * (features[r * kFeatureDims + kBaseDims + d] -
                  features[l * kFeatureDims + kBaseDims + d]);
  }
  auto end = std::chrono::steady_clock::now();
  elapsedMs = std::chrono::duration<double, std::milli>(end - begin).count();
  return features;
}

void writeCsv(const std::string& path, const std::vector<float>& features,
              int frames) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Cannot write output: " + path);
  out << "frame";
  for (int d = 0; d < kFeatureDims; ++d) out << ",feature_" << d;
  out << '\n' << std::setprecision(9);
  for (int f = 0; f < frames; ++f) {
    out << f;
    for (int d = 0; d < kFeatureDims; ++d)
      out << ',' << features[f * kFeatureDims + d];
    out << '\n';
  }
}

struct Options {
  std::string input, output = "features.csv", backend = "gpu";
};
Options parseArgs(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--input" || arg == "--output" || arg == "--backend") &&
        i + 1 < argc) {
      std::string value = argv[++i];
      if (arg == "--input") o.input = value;
      else if (arg == "--output") o.output = value;
      else o.backend = value;
    } else if (arg == "--help") {
      std::cout << "Usage: phase_audio_cuda --input FILE.wav "
                   "[--output FILE.csv] [--backend gpu|cpu|both]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown or incomplete argument: " + arg);
    }
  }
  if (o.input.empty()) throw std::runtime_error("Missing --input");
  if (o.backend != "gpu" && o.backend != "cpu" && o.backend != "both")
    throw std::runtime_error("--backend must be gpu, cpu, or both");
  return o;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parseArgs(argc, argv);
    WavData wav = loadWav(options.input);
    int frames = frameCount(wav.samples.size());
    std::cout << "Input: " << options.input << "\n"
              << "Sample rate: " << wav.sample_rate << " Hz\n"
              << "Samples: " << wav.samples.size() << "\n"
              << "Frames: " << frames << "\n"
              << "Feature dimensions: " << kFeatureDims << "\n";

    std::vector<float> features;
    double gpuMs = 0.0, cpuMs = 0.0;
    if (options.backend == "cpu") {
      features = runCpu(wav.samples, frames, cpuMs);
      std::cout << "CPU processing: " << cpuMs << " ms\n";
    } else {
      features = runGpu(wav.samples, frames, gpuMs);
      std::cout << "GPU kernel + cuFFT time: " << gpuMs << " ms\n";
      if (options.backend == "both") {
        auto cpuFeatures = runCpu(wav.samples, frames, cpuMs);
        double maxError = 0.0;
        for (size_t i = 0; i < features.size(); ++i)
          maxError = std::max(maxError,
                              static_cast<double>(std::fabs(features[i] -
                                                            cpuFeatures[i])));
        std::cout << "CPU processing: " << cpuMs << " ms\n"
                  << "Maximum absolute CPU/GPU feature difference: "
                  << maxError << "\n";
        if (gpuMs > 0.0)
          std::cout << "Measured CPU/GPU time ratio: "
                    << cpuMs / gpuMs << "x\n";
      }
    }
    writeCsv(options.output, features, frames);
    std::cout << "Wrote " << frames << " rows x " << kFeatureDims
              << " features to " << options.output << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
