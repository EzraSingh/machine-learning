#!/usr/bin/env python3
import os, numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

TOLERANCE = float(os.getenv("GRADING_TOLERANCE"))

# data classes, these are the nerual network output space targets
CLASS_A = [ 0, 1 ]
CLASS_B = [ 1, 0 ]

group_A = [(3, 4), (4.2, 5.3), (4, 3), (6, 5), (4, 6), (3.7, 5.8), (3.2, 4.6), (5.2, 5.9), (5, 4), (7, 4), (3, 7), (4.3, 4.3) ] 
group_B = [(-3, -4), (-2, -3.5), (-1, -6), (-3, -4.3), (-4, -5.6), (-3.2, -4.8), (-2.3, -4.3), (-2.7, -2.6), (-1.5, -3.6), (-3.6, -5.6), (-4.5, -4.6), (-3.7, -5.8) ]

labeled_data = []

for data in group_A:
    labeled_data.append([data, CLASS_A])

for data in group_B:
    labeled_data.append([data, CLASS_B])

# load data and label associations
data, labels = zip(*labeled_data)
labels = np.array(labels)
data = np.array(data)

log_template = "{{\nlabel: {}\nresults: {}\nerror: {}\n}}\n"

plt.xlabel('Epoch')
plt.ylabel('Error')
plt.yscale('log')

window_id = 221

def test(network, header, verbose=os.getenv("VERBOSE", False)):
    global window_id
    score = 0
    epoch_set = np.arange(0, len(data))
    error_set = []
    for i in range(len(data)):
        results = network.run(data[i])
        error = labels[i] - results
        error = np.abs(np.average(error))
        if verbose:
            print(log_template.format(labels[i], results, np.round(error, 2)))
        if error < TOLERANCE:
            score += 1
        error_set.append(error)
    plt.subplot(window_id)
    plt.axis([0, len(data), 0, 0.1])
    plt.title(header)
    plt.plot(epoch_set, error_set)
    window_id += 1
    return score / len(data)

def train(network, epochs=20):
    for _ in range(epochs):
        for i in range(len(data)):
            network.train(data[i], labels[i])

