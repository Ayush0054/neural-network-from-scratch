"""
view_weights
~~~~~~~~~~~~

Show what the trained network learned, from ``weights/network.pkl``.

Usage (from any folder)::

    python neural-network/view_weights.py                 # show on screen
    python neural-network/view_weights.py --save          # write PNGs to weights/
    python neural-network/view_weights.py --file other.pkl

Figures:
1. Hidden neurons: the 784 input weights of each hidden neuron as a 28x28
   image. Red = this pixel pushes the neuron UP, blue = pushes it DOWN.
2. Output layer: the (10 x hidden) weight matrix as a heatmap. Each row is one
   digit; each column is one hidden neuron.
3. Biases and weight distribution: bias bars for each layer and a histogram of
   all weights.
"""

import argparse
import os
import pickle

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILE = os.path.join(BASE_DIR, "weights", "network.pkl")


def load(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["sizes"], data["weights"], data["biases"]


def print_summary(sizes, weights, biases):
    print(f"Network sizes: {sizes}")
    total = 0
    for i, (w, b) in enumerate(zip(weights, biases), start=1):
        n = w.size + b.size
        total += n
        print(f"  Layer {i}: weights {w.shape}, biases {b.shape}  "
              f"w mean {w.mean():+.3f}, std {w.std():.3f}, "
              f"min {w.min():+.2f}, max {w.max():+.2f}")
    print(f"  Total parameters: {total:,}")


def plot_hidden_neurons(plt, weights):
    w = weights[0]                       # shape (n_hidden, 784)
    n = w.shape[0]
    if w.shape[1] != 784:
        print("First layer is not 784 inputs, skipping the 28x28 view.")
        return None
    cols = min(10, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.7 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= n:
            continue
        img = w[i].reshape(28, 28)
        lim = np.abs(img).max()          # symmetric scale: 0 is always white
        ax.imshow(img, cmap="bwr", vmin=-lim, vmax=lim)
        ax.set_title(f"h{i}", fontsize=8)
    fig.suptitle("Hidden neurons: input weights (red +, blue -)")
    fig.tight_layout()
    return fig


def plot_output_layer(plt, weights):
    w = weights[-1]                      # shape (10, n_hidden)
    lim = np.abs(w).max()
    fig, ax = plt.subplots(figsize=(max(6, 0.3 * w.shape[1]), 4))
    im = ax.imshow(w, cmap="bwr", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xlabel("hidden neuron")
    ax.set_ylabel("output digit")
    ax.set_yticks(range(w.shape[0]))
    ax.set_xticks(range(w.shape[1]))
    ax.tick_params(axis="x", labelsize=7)
    fig.colorbar(im, ax=ax, label="weight")
    ax.set_title("Output layer weights: which hidden neurons each digit uses")
    fig.tight_layout()
    return fig


def plot_biases_and_histogram(plt, weights, biases):
    n_layers = len(biases)
    fig, axes = plt.subplots(1, n_layers + 1, figsize=(4.5 * (n_layers + 1), 3.5))
    for i, b in enumerate(biases):
        ax = axes[i]
        vals = b.ravel()
        ax.bar(range(len(vals)), vals, color=["#d6604d" if v > 0 else "#4393c3" for v in vals])
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"Layer {i + 1} biases ({len(vals)})")
        ax.set_xlabel("neuron")
    ax = axes[-1]
    for i, w in enumerate(weights):
        ax.hist(w.ravel(), bins=80, alpha=0.6, label=f"layer {i + 1}", density=True)
    ax.set_title("Weight distribution")
    ax.set_xlabel("weight value")
    ax.legend()
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="View trained network weights.")
    parser.add_argument("--file", default=DEFAULT_FILE, help="path to network.pkl")
    parser.add_argument("--save", action="store_true", help="save PNGs next to the .pkl file")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise SystemExit(f"No weights at {args.file}. Run run_network.py first.")

    sizes, weights, biases = load(args.file)
    print_summary(sizes, weights, biases)

    try:
        import matplotlib
        if args.save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is not installed. Run `.venv/bin/pip install matplotlib`.")

    figs = {
        "hidden_neurons": plot_hidden_neurons(plt, weights),
        "output_layer": plot_output_layer(plt, weights),
        "biases_and_histogram": plot_biases_and_histogram(plt, weights, biases),
    }

    if args.save:
        out_dir = os.path.dirname(os.path.abspath(args.file))
        for name, fig in figs.items():
            if fig is None:
                continue
            path = os.path.join(out_dir, f"{name}.png")
            fig.savefig(path, dpi=120)
            print(f"Saved {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
