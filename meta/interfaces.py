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

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from typing import Any, Dict, Iterable, List, Optional

import msgpack
import numpy as np
import os


# noinspection PyPep8Naming
class MetaClassificationModel(Output, ABC):
    """
    Base class for meta classification models.
    """

    def __init__(self):
        self._clf = []
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
    async def fit(self, X: Iterable[Iterable[float]], y: Iterable[int]):
        """
        Fit model to data in X with labels from y.

        :param X: training samples
        :param y: labels
        """
        pass

    @abstractmethod
    async def predict(self, X: Iterable[Iterable[float]]) -> Iterable[int]:
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

    async def load(self, file_name: str):
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
                        clf.__dict__[k[1:].decode("utf-8")] = np.array(clf_dict[k])
                    if k[0] == ord("s"):
                        clf.__dict__[k[1:].decode("utf-8")] = clf_dict[k].decode("utf-8")
                    else:
                        clf.__dict__[k[1:].decode("utf-8")] = clf_dict[k]
                self._clf.append(clf)

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
            out_dict["clf"].append(clf_dict)

        if file_name is None:
            file_name = self._generate_output_basename() + ".model"

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
