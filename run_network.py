import os
import pickle
import sys

import numpy as np

import mnist_loader
import neural_network

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
MODEL_FILE = os.path.join(WEIGHTS_DIR, "network.pkl")


def save(net, path=MODEL_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"sizes": net.sizes, "weights": net.weights, "biases": net.biases}, f)
    print(f"Saved weights to {path}")


def load(path=MODEL_FILE):
    with open(path, "rb") as f:
        data = pickle.load(f)
    net = neural_network.Network(data["sizes"])
    net.weights = data["weights"]
    net.biases = data["biases"]
    print(f"Loaded weights from {path}")
    return net


def test(net, test_data, show=10):
    # Accuracy on the full test set
    correct = net.evaluate(test_data)
    total = len(test_data)
    print(f"\nTest accuracy: {correct} / {total} = {correct / total:.2%}")

    # Some single predictions
    print(f"\nFirst {show} test images:")
    for x, y in test_data[:show]:
        output = net.feedforward(x)
        pred = int(np.argmax(output))
        mark = "OK " if pred == y else "BAD"
        print(f"  {mark} predicted {pred}, correct {y}, confidence {output[pred][0]:.2f}")

    # Correct predictions for each digit
    print("\nAccuracy for each digit:")
    for d in range(10):
        items = [(x, y) for x, y in test_data if y == d]
        ok = sum(int(np.argmax(net.feedforward(x)) == y) for x, y in items)
        print(f"  {d}: {ok}/{len(items)} = {ok / len(items):.2%}")


if __name__ == "__main__":
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)

    retrain = "--retrain" in sys.argv

    if os.path.exists(MODEL_FILE) and not retrain:
        net = load()
    else:
        net = neural_network.Network([784, 30, 10])
        net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
        save(net)

    test(net, test_data)