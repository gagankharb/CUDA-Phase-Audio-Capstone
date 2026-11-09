#!/usr/bin/env python3
"""Validate and plot a feature CSV produced by phase_audio_cuda."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--plot", default="output/features.png")
    args = parser.parse_args()
    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    names = data.dtype.names
    feature_names = [n for n in names if n.startswith("feature_")]
    matrix = np.column_stack([data[n] for n in feature_names])
    if matrix.shape[1] != 60:
        raise SystemExit(f"Expected 60 feature columns, got {matrix.shape[1]}")
    if not np.isfinite(matrix).all():
        raise SystemExit("CSV contains NaN or infinity")
    print(f"Validated: {matrix.shape[0]} frames x {matrix.shape[1]} features")
    print(f"Feature range: [{matrix.min():.6f}, {matrix.max():.6f}]")
    plt.figure(figsize=(12, 5))
    plt.imshow(matrix.T, aspect="auto", origin="lower", interpolation="nearest")
    plt.xlabel("Frame")
    plt.ylabel("Feature index")
    plt.title("CUDA phase-derived audio features (60 dimensions)")
    plt.colorbar(label="Feature value")
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.plot, dpi=160)
    print(f"Saved plot: {args.plot}")

if __name__ == "__main__":
    main()
