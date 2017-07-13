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

from output.formats import UnmaskingResult

from typing import Optional, Tuple

import numpy as np


def unmasking_result_to_numpy(result: UnmaskingResult) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert UnmaskingResult to a numpy feature matrix containing the curve data and a numpy
    array containing the class labels.
    The feature matrix rows consist of the original curve values and their first derivative.

    String labels from the given UnmaskingResult are represented by integers (starting at 0) in the
    order in which they appear in :attr:: UnmaskingResult.meta.

    :param result: UnmaskingResult to convert
    :return: numpy matrix with data samples and numpy array with integer labels (None if there are no labels)
    """

    classes = result.meta.get("classes", [])
    classes_mapping = {k: i for i, k in enumerate(classes)}

    curves = result.curves
    num_rows = len(curves)
    num_cols = max((len(curves[c]["values"]) for c in curves)) * 2

    X = np.zeros((num_rows, num_cols))
    y = np.zeros(num_rows)

    no_labels = False
    for i, c in enumerate(curves):
        if not curves[c]["values"]:
            continue

        data = np.array(curves[c]["values"])
        X[i] = np.concatenate((data, np.gradient(data)))

        if no_labels or "cls" not in curves[c]:
            no_labels = True
        else:
            y[i] = classes_mapping.get(curves[c]["cls"])

    return X, (y if not no_labels else None)
