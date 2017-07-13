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

from output.interfaces import Output
from conf.interfaces import Configurable

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json
import msgpack
import numpy as np
import os


# noinspection PyPep8Naming
class MetaClassificationModel(Configurable, Output, ABC):
    """
    Base class for meta classification models.
    """

    def __init__(self):
        self._clf = []
        self._str_labels = tuple()
        self._version = 1

    @abstractmethod
    def _get_estimator(self, index: int) -> BaseEstimator:
        """
        Create a new estimator instance.

        :param index: estimator index number (relevant if model has multiple estimators of different types)
        :return: estimator instance
        """
        pass

    @abstractmethod
    def fit(self, X: Iterable[Iterable[float]], y: Iterable[int]):
        """
        Fit model to data in X with labels from y.

        :param X: training samples
        :param y: labels
        """
        pass

    @abstractmethod
    def predict(self, X: Iterable[Iterable[float]]) -> Iterable[int]:
        """
        Predict classes for samples in X.

        :param X: samples
        :return: predicted labels
        """
        pass

    @abstractmethod
    def decision_function(self, X: Iterable[Iterable[float]]) -> Iterable[float]:
        """
        Classification decision function / probabilities for samples in X.

        :param X: samples
        :return: decision function values
        """
        pass

    def load_unmasking_results(self, file_name: str) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[str, ...]]:
        """
        Load unmasking result curves from the given input files and convert them to Numpy arrays
        which can be used with a classifier.

        :param file_name: file to load curves from
        :return: (1) numpy matrix of curve data, (2) array of int labels (None if not all data have labels)
                 and (3) tuple mapping int labels to the original string labels (by position)
        """
        X = None
        y = None

        none_labels = False
        with open(file_name) as f:
            json_data = json.load(f)

        if "meta" not in json_data:
            raise ValueError("No meta section")

        if "curves" not in json_data:
            raise ValueError("No curves section")

        meta = json_data["meta"]
        if "classes" not in meta:
            classes = set()
            for c in json_data["curves"]:
                if "cls" in c:
                    classes.add(c["cls"])
        else:
            classes = meta["classes"]

        classes = tuple([c for c in sorted(classes)])
        classes_inv = {k: i for i, k in enumerate(classes)}

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

            y[i] = classes_inv.get(json_data["curves"][c]["cls"])

        return np.array(X), (np.array(y) if not none_labels else None), classes

    def fit_from_unmasking_results(self, file_name: str):
        """
        Fit a model from unmasking results in the given file.

        :param file_name: unmasking result file name
        """
        X, y, classes = self.load_unmasking_results(file_name)
        self._str_labels = classes

        if y is None:
            raise ValueError("Training labels must be set")

        self.fit(X, y)

    def predict_from_unmasking_results(self, file_name: str) -> Tuple[Iterable[int], Tuple[str, ...]]:
        """
        Predict samples from unmasking results file.
        Returns a Numpy array of predicted int labels and a tuple mapping these int
        labels to the original string labels (if given in the input file).
        Individual labels may be nan if decision probability is below threshold

        :return: tuple predicted classes and original string classes
        """
        X, _, classes = self.load_unmasking_results(file_name)
        return self.predict(X), classes if classes else self._str_labels

    def load(self, file_name: str):
        """
        Load model from given file.

        :param file_name: input file name
        """
        with open(file_name, "rb") as f:
            in_data = msgpack.unpack(f, use_list=False)

        if "version" not in in_data:
            raise IOError("Invalid model format")

        if in_data["version"] > self._version or in_data["version"] < 1:
            raise ValueError("Unsupported model version: " + in_data["version"])

        if in_data["version"] == 1:
            for i, clf_dict in enumerate(in_data["clf"]):
                clf = self._get_estimator(i)
                for k in clf_dict[0]:
                    if k[0] == ord("a"):
                        clf.__dict__[k[1:].decode("utf-8")] = np.array(clf_dict[0][k])
                    if k[0] == ord("s"):
                        clf.__dict__[k[1:].decode("utf-8")] = clf_dict[0][k].decode("utf-8")
                    else:
                        clf.__dict__[k[1:].decode("utf-8")] = clf_dict[0][k]
                self._clf.append(clf)

                labels = []
                for l in clf_dict[1]:
                    labels.append(l.decode("utf-8"))
                self._str_labels = tuple(labels)

    def save(self, output_dir: str, file_name: Optional[str] = None):
        out_dict = {
            "version": self._version,
            "clf": []
        }
        for clf in self._clf:
            clf_dict = clf.__dict__.copy()
            for k in clf_dict:
                if isinstance(clf_dict[k], np.ndarray):
                    clf_dict["a" + k] = tuple(clf_dict[k])
                if isinstance(clf_dict[k], str):
                    clf_dict["s" + k] = clf_dict[k]
                else:
                    clf_dict["t" + k] = clf_dict[k]
            out_dict["clf"].append((clf_dict, self._str_labels))

        if file_name is None:
            file_name = self._get_output_filename_base() + ".model"

        with open(os.path.join(output_dir, file_name), "wb") as f:
            msgpack.pack(out_dict, f)

    def reset(self):
        self.__init__()

    @property
    def params(self) -> List[Dict[str, Any]]:
        """Get parameters for each estimator."""
        return [p.get_params() for p in self._clf]

    @params.setter
    def params(self, params: List[Dict[str, Any]]):
        """Set parameters for each estimator."""
        for i, p in enumerate(params):
            self._clf[i].set_params(**p)
