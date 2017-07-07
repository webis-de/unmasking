# General-purpose unmasking framework
# Copyright (C) 2017 Janek Bevendorff, Webis Group
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.


import json
import numpy as np
from sklearn.svm import LinearSVC


def load_unmasking_results(*file_names: str):
    """
    Load unmasking result curves from the given input files.
    Feature vectors will contain raw curve data and their gradients.

    :param file_names: files to load curves from
    :return: numpy matrix of curve data and array of labels (None if not all data have labels)
    """
    X = None
    y = None

    none_labels = False
    for file_name in file_names:
        with open(file_name) as f:
            json_data = json.load(f)

        if X is None:
            X = [0.0] * len(json_data["curves"])
            y = [0] * len(json_data["curves"])

        for i, c in enumerate(json_data["curves"]):
            if not json_data["curves"][c]["curve"]:
                continue

            data = np.array(json_data["curves"][c]["curve"])
            X[i] = np.add(X[i], np.concatenate((data, np.gradient(data))))

            if none_labels or "cls" not in json_data["curves"][c]:
                none_labels = True
                continue

            y[i] = int(json_data["curves"][c]["cls"] == "SAME_AUTHOR")

    for i, x in enumerate(X):
        X[i] = np.divide(x, len(file_names))

    return np.array(X), (np.array(y) if not none_labels else None)


def train(input_file: str, threshold: float):
    """
    Train a model from a given input file.
    Samples with an uncertainty value of less than `threshold` will be discarded.

    :param input_file: input file name
    :param threshold: uncertainty threshold
    :return: trained classifiers for determining which samples to classify (1) and for
             actually classifying the filtered curves (2)
    """
    X, y = load_unmasking_results(input_file)

    if y is None:
        raise ValueError("Training data must have labels")

    clf1 = LinearSVC()
    clf1.fit(X, y)

    dist = clf1.decision_function(X)

    # eliminate samples below threshold
    X = np.fromiter((x for i, x in enumerate(X) if abs(dist[i]) >= threshold), float)
    y = np.fromiter((y for i, y in enumerate(y) if abs(dist[i]) >= threshold), float)

    # retrain classifier
    clf2 = LinearSVC()
    clf2.fit(X, y)

    return clf1, clf2
