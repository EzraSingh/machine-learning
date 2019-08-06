#!/usr/bin/env python3
import os, settings
from src.ai import NeuralNetwork
from matplotlib import pyplot as plt

epochs = int(os.getenv("TRAINING_EPOCHS"))

def run_classifier():
    from scripts.classification import train, test
    # Neural network that maps input point (x, y) and maps to ouptut point (x', y')
    simple_classifier = NeuralNetwork(
        topology=(2, 10, 2), 
        learning_rate=0.1,
        bias=True
    )

    score = test(simple_classifier, "Initial Results")
    print("Initial Score: {0:.2f}%".format(100 * score))

    train(simple_classifier, epochs)

    score = test(simple_classifier, "Post Training Results")
    print("Post Score: {0:.2f}%".format(100 * score))
    plt.show()


if __name__ == "__main__":
    run_classifier()