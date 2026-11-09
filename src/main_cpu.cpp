#include <algorithm>
#include <cmath>
#include <complex>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const int kSampleRate = 16000;
const int kWindow = 400;
const int kHop = 160;
const int kFftSize = 512;
const int kBins = kFftSize / 2 + 1;
const int kCepstra = 10;
const int kBaseDims = 2 * kCepstra;
const int kFeatureDims = 3 * kBaseDims;
const float kPi = 3.14159265358979323846f;
const float kVadRms = 0.005f;

struct WavData {
    int sample_rate;
    std::vector<float> samples;
};

uint16_t readU16(std::istream& in) {
    unsigned char b[2];
    in.read(reinterpret_cast<char*>(b), 2);
    if (!in) throw std::runtime_error("Unexpected end of WAV file");
    return static_cast<uint16_t>(b[0] | (static_cast<unsigned int>(b[1]) << 8));
}

uint32_t readU32(std::istream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) throw std::runtime_error("Unexpected end of WAV file");
    return static_cast<uint32_t>(
        static_cast<uint32_t>(b[0]) |
        (static_cast<uint32_t>(b[1]) << 8) |
        (static_cast<uint32_t>(b[2]) << 16) |
        (static_cast<uint32_t>(b[3]) << 24));
}

WavData loadWav(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open input WAV: " + path);

    char riff[4], wave[4];
    in.read(riff, 4);
    (void)readU32(in);
    in.read(wave, 4);
    if (!in || std::string(riff, 4) != "RIFF" ||
        std::string(wave, 4) != "WAVE") {
        throw std::runtime_error("Input is not a RIFF/WAVE file");
    }

    uint16_t format = 0, channels = 0, bits = 0;
    uint32_t sampleRate = 0;
    std::vector<unsigned char> pcm;

    while (in) {
        char id[4];
        in.read(id, 4);
        if (!in) break;
        uint32_t size = readU32(in);
        std::string chunk(id, 4);

        if (chunk == "fmt ") {
            format = readU16(in);
            channels = readU16(in);
            sampleRate = readU32(in);
            (void)readU32(in);
            (void)readU16(in);
            bits = readU16(in);
            if (size > 16) in.seekg(size - 16, std::ios::cur);
        } else if (chunk == "data") {
            pcm.resize(size);
            in.read(reinterpret_cast<char*>(pcm.data()), size);
            if (!in) throw std::runtime_error("Incomplete WAV data chunk");
        } else {
            in.seekg(size, std::ios::cur);
        }
        if (size & 1U) in.seekg(1, std::ios::cur);
    }

    if (format != 1 || bits != 16 || channels == 0) {
        throw std::runtime_error("Only PCM 16-bit WAV files are supported");
    }
    if (sampleRate != static_cast<uint32_t>(kSampleRate)) {
        throw std::runtime_error("WAV must be 16 kHz; input is " +
                                 std::to_string(sampleRate) + " Hz");
    }

    const size_t bytesPerSample = 2;
    const size_t frameBytes = bytesPerSample * channels;
    if (pcm.size() < frameBytes || pcm.size() % frameBytes != 0) {
        throw std::runtime_error("WAV contains no complete audio samples");
    }

    WavData wav;
    wav.sample_rate = static_cast<int>(sampleRate);
    const size_t count = pcm.size() / frameBytes;
    wav.samples.resize(count);

    for (size_t i = 0; i < count; ++i) {
        float sum = 0.0f;
        for (uint16_t c = 0; c < channels; ++c) {
            const size_t p = i * frameBytes + c * 2;
            uint16_t raw = static_cast<uint16_t>(
                pcm[p] | (static_cast<unsigned int>(pcm[p + 1]) << 8));
            int16_t value;
            std::memcpy(&value, &raw, sizeof(value));
            sum += static_cast<float>(value) / 32768.0f;
        }
        wav.samples[i] = sum / channels;
    }
    return wav;
}

int frameCount(size_t n) {
    if (n <= static_cast<size_t>(kWindow)) return 1;
    return 1 + static_cast<int>(
        (n - kWindow + kHop - 1) / kHop);
}

float wrapPhase(float x) {
    return std::atan2(std::sin(x), std::cos(x));
}

// Single-threaded radix-2 FFT reference implementation.
void cpuFft(std::vector<std::complex<float> >& a) {
    const int n = static_cast<int>(a.size());

    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        const float angle = -2.0f * kPi / len;
        const std::complex<float> root(std::cos(angle), std::sin(angle));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; ++j) {
                const std::complex<float> u = a[i + j];
                const std::complex<float> v = a[i + j + len / 2] * w;
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
    std::vector<std::complex<float> > spectra(frames * kFftSize);
    std::vector<float> base(frames * kBaseDims, 0.0f);
    std::vector<float> features(frames * kFeatureDims, 0.0f);

    const std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    // Frame, apply Hann window, and calculate the FFT.
    for (int f = 0; f < frames; ++f) {
        std::vector<std::complex<float> > frame(
            kFftSize, std::complex<float>(0.0f, 0.0f));
        for (int i = 0; i < kWindow && f * kHop + i < n; ++i) {
            const float hann = 0.5f - 0.5f *
                std::cos(2.0f * kPi * i / (kWindow - 1));
            frame[i] = std::complex<float>(samples[f * kHop + i] * hann, 0.0f);
        }
        cpuFft(frame);
        std::copy(frame.begin(), frame.end(), spectra.begin() + f * kFftSize);
    }

    // Calculate 10 sine and 10 cosine phase-cepstral coefficients per frame.
    for (int f = 1; f < frames; ++f) {
        float energy = 0.0f;
        for (int i = 0; i < kWindow && f * kHop + i < n; ++i) {
            const float x = samples[f * kHop + i];
            energy += x * x;
        }
        if (std::sqrt(energy / kWindow) < kVadRms) continue;

        for (int d = 0; d < kBaseDims; ++d) {
            const int c = d % kCepstra;
            const bool useSine = d < kCepstra;
            float sum = 0.0f;

            for (int k = 0; k < kBins; ++k) {
                const std::complex<float> current =
                    spectra[f * kFftSize + k];
                const std::complex<float> previous =
                    spectra[(f - 1) * kFftSize + k];

                const float phase = std::arg(current * std::conj(previous));
                const float expected =
                    2.0f * kPi * k * kHop / kFftSize;
                const float residual = wrapPhase(phase - expected);
                const float basis =
                    std::cos(kPi * c * (k + 0.5f) / kBins);

                sum += (useSine ? std::sin(residual) :
                                   std::cos(residual)) * basis;
            }
            base[f * kBaseDims + d] = sum / kBins;
        }
    }

    // First-order temporal differences (deltas).
    for (int f = 0; f < frames; ++f) {
        const int left = std::max(f - 1, 0);
        const int right = std::min(f + 1, frames - 1);
        for (int d = 0; d < kBaseDims; ++d) {
            const float delta = 0.5f *
                (base[right * kBaseDims + d] -
                 base[left * kBaseDims + d]);
            features[f * kFeatureDims + d] = base[f * kBaseDims + d];
            features[f * kFeatureDims + kBaseDims + d] = delta;
        }
    }

    // Second-order temporal differences (delta-deltas).
    for (int f = 0; f < frames; ++f) {
        const int left = std::max(f - 1, 0);
        const int right = std::min(f + 1, frames - 1);
        for (int d = 0; d < kBaseDims; ++d) {
            features[f * kFeatureDims + 2 * kBaseDims + d] = 0.5f *
                (features[right * kFeatureDims + kBaseDims + d] -
                 features[left * kFeatureDims + kBaseDims + d]);
        }
    }

    const std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    elapsedMs = std::chrono::duration<double, std::milli>(end - begin).count();
    return features;
}

void writeCsv(const std::string& path, const std::vector<float>& features,
              int frames) {
    std::ofstream out(path.c_str());
    if (!out) throw std::runtime_error("Cannot write output: " + path);

    out << "frame";
    for (int d = 0; d < kFeatureDims; ++d) {
        out << ",feature_" << d;
    }
    out << '\n' << std::setprecision(9);

    for (int f = 0; f < frames; ++f) {
        out << f;
        for (int d = 0; d < kFeatureDims; ++d) {
            out << ',' << features[f * kFeatureDims + d];
        }
        out << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string input;
        std::string output = "features_cpu.csv";

        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--input" && i + 1 < argc) {
                input = argv[++i];
            } else if (arg == "--output" && i + 1 < argc) {
                output = argv[++i];
            } else if (arg == "--backend" && i + 1 < argc) {
                const std::string backend(argv[++i]);
                if (backend != "cpu") {
                    throw std::runtime_error(
                        "This Windows build supports only --backend cpu");
                }
            } else if (arg == "--help") {
                std::cout << "Usage: phase_audio_cpu --input FILE.wav "
                             "[--output FILE.csv] [--backend cpu]\n";
                return 0;
            } else {
                throw std::runtime_error("Unknown or incomplete argument: " + arg);
            }
        }

        if (input.empty()) {
            throw std::runtime_error("Missing --input FILE.wav");
        }

        const WavData wav = loadWav(input);
        const int frames = frameCount(wav.samples.size());

        std::cout << "Input: " << input << "\n"
                  << "Sample rate: " << wav.sample_rate << " Hz\n"
                  << "Samples: " << wav.samples.size() << "\n"
                  << "Frames: " << frames << "\n"
                  << "Feature dimensions: " << kFeatureDims << "\n";

        double cpuMs = 0.0;
        const std::vector<float> features =
            runCpu(wav.samples, frames, cpuMs);

        std::cout << "CPU processing: " << cpuMs << " ms\n";
        writeCsv(output, features, frames);
        std::cout << "Wrote " << frames << " rows x " << kFeatureDims
                  << " features to " << output << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
