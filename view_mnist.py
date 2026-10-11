"""
view_mnist
~~~~~~~~~~

View MNIST digits from ``data/mnist.pkl`` (or ``data/mnist.pkl.gz``).

Usage examples (run from the repo root)::

    python src/view_mnist.py                      # first 32 training images
    python src/view_mnist.py --split test -n 16   # first 16 test images
    python src/view_mnist.py --start 100 -n 8     # training images 100..107
    python src/view_mnist.py --digit 7            # only images labelled 7
    python src/view_mnist.py --ascii -n 3         # print in the terminal
    python src/view_mnist.py --save digits.png    # write to a PNG file
"""

#### Libraries
# Standard library
import argparse
import gzip
import os
import pickle

# Third-party libraries
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SPLITS = {"train": 0, "val": 1, "test": 2}


def load_data():
    """Return ``(training_data, validation_data, test_data)``, each an
    ``(images, labels)`` tuple. Prefers the uncompressed ``mnist.pkl``
    and falls back to ``mnist.pkl.gz``."""
    pkl = os.path.join(DATA_DIR, "mnist.pkl")
    if os.path.exists(pkl):
        f = open(pkl, "rb")
    else:
        f = gzip.open(os.path.join(DATA_DIR, "mnist.pkl.gz"), "rb")
    with f:
        # The file was pickled under Python 2, hence ``latin1``.
        return pickle.load(f, encoding="latin1")


def select(images, labels, start, count, digit=None):
    """Return the indices of up to ``count`` images beginning at
    ``start``, optionally keeping only those labelled ``digit``."""
    indices = np.arange(start, len(labels))
    if digit is not None:
        indices = indices[labels[indices] == digit]
    return indices[:count]


def show_ascii(images, labels, indices):
    """Print each image as text, darker pixels drawn with denser characters."""
    for i in indices:
        print("index %d  label %d" % (i, labels[i]))
        for row in images[i].reshape(28, 28):
            print("".join("#" if v > 0.5 else "+" if v > 0.1 else "." for v in row))
        print()


def show_grid(images, labels, indices, title, save=None):
    """Draw the images in a grid with matplotlib, either on screen or to
    the file ``save``."""
    try:
        import matplotlib
        if save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is not installed. Run `.venv/bin/pip install "
                         "matplotlib`, or use --ascii to print in the terminal.")

    cols = min(8, len(indices))
    rows = int(np.ceil(len(indices) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(1.4 * cols, 1.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, indices):
        ax.imshow(images[i].reshape(28, 28), cmap="gray_r")
        ax.set_title("#%d: %d" % (i, labels[i]), fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        print("Saved %s" % save)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="View MNIST digits.")
    parser.add_argument("--split", choices=SPLITS, default="train",
                        help="which dataset split to view (default: train)")
    parser.add_argument("--start", type=int, default=0,
                        help="index of the first image (default: 0)")
    parser.add_argument("-n", "--count", type=int, default=32,
                        help="number of images to show (default: 32)")
    parser.add_argument("--digit", type=int, choices=range(10),
                        help="only show images with this label")
    parser.add_argument("--ascii", action="store_true",
                        help="print images in the terminal instead of a window")
    parser.add_argument("--save", metavar="FILE",
                        help="save the grid to an image file instead of opening a window")
    args = parser.parse_args()

    data = load_data()
    images, labels = data[SPLITS[args.split]]
    print("%s: %d images, shape %s, pixel range %.2f-%.2f"
          % (args.split, len(labels), images.shape, images.min(), images.max()))

    indices = select(images, labels, args.start, args.count, args.digit)
    if len(indices) == 0:
        raise SystemExit("No images match those options.")

    if args.ascii:
        show_ascii(images, labels, indices)
    else:
        title = "MNIST %s" % args.split
        if args.digit is not None:
            title += " (digit %d)" % args.digit
        show_grid(images, labels, indices, title, args.save)


if __name__ == "__main__":
    main()
