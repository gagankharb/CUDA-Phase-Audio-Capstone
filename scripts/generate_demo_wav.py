#!/usr/bin/env python3
"""Generate a deterministic 16 kHz PCM16 test signal (silence + tones + chirp)."""
import argparse
import math
import wave
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/demo.wav")
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()
    rate = 16000
    count = int(rate * args.seconds)
    samples = []
    for i in range(count):
        t = i / rate
        if t < 0.35 or 3.65 < t:
            x = 0.0
        else:
            freq = 180.0 + 420.0 * (t / args.seconds)
            x = 0.35 * math.sin(2 * math.pi * 220 * t)
            x += 0.18 * math.sin(2 * math.pi * freq * t)
            # Deterministic low-level modulation makes the signal nonstationary.
            x *= 0.75 + 0.25 * math.sin(2 * math.pi * 3.0 * t)
        samples.append(max(-1.0, min(1.0, x)))
    pcm = bytearray()
    for x in samples:
        value = int(round(x * 32767))
        pcm.extend(int(value).to_bytes(2, byteorder="little", signed=True))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(args.output, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    print(f"Generated {args.output}: {count} samples at {rate} Hz")

if __name__ == "__main__":
    main()
