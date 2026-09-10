from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "samples1_3d.npy"
PLOT_PATH = SCRIPT_DIR / "samples_distribution.png"


def f(i, k=0.00230259):
    return i / 8500


def generate_group(n_samples, n_dimensions, resize_samples):
    sampler = qmc.LatinHypercube(d=n_dimensions)
    samples = sampler.random(n=n_samples)
    samples = samples * f(resize_samples)
    for i in range(n_samples):
        while samples[i].sum() > 1:
            samples[i] = sampler.random(n=1)
            samples[i] = samples[i] * f(resize_samples)
    total_sum = np.sum(samples, axis=1)
    ave = np.sum(total_sum) / len(total_sum)
    return ave, samples


def main() -> None:
    total_groups = 10000
    samples_per_group = 47
    dimensions_per_sample = 3
    ave1 = np.zeros(total_groups)
    samples1 = np.zeros((total_groups, samples_per_group, dimensions_per_sample))
    for i in tqdm(range(total_groups), desc="Processing Groups"):
        if i <= total_groups:
            ave1[i], samples1[i, :, :] = generate_group(samples_per_group, dimensions_per_sample, i)
    np.save(OUTPUT_PATH, samples1)
    samples2 = np.sum(samples1, axis=2)
    greater_than_1_positions = np.where(samples2 > 1)
    greater_than_1_count = np.sum(samples2 > 1)
    print("positions greater than 1:", greater_than_1_positions)
    print("count greater than 1:", greater_than_1_count)
    samples3 = np.mean(samples2, axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(samples3, bins=50, range=(0, 1), alpha=0.75, color="blue", edgecolor="black")
    plt.title("Distribution of samples3 (Average values in the 0-1 range)")
    plt.xlabel("Average value1")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {OUTPUT_PATH}")
    print(f"saved {PLOT_PATH}")


if __name__ == "__main__":
    main()
