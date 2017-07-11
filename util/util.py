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

import functools
import os
import sys


__cached_functions = []
__protected_cached_functions = []


def lru_cache(protected: bool = False, maxsize: int = 128, typed: bool = False):
    """
    Wrapped version of the functools.lru_cache decorator which
    keeps track of all decorated functions to allow clearing all caches.

    :param protected: if set to True, the cache for this function will only be cleared by
                      :function:: clear_lru_caches() if its `clear_protected`
                      parameter is set to True
    :param maxsize: maximum cache size (None for unlimited)
    :param typed: if this is True, parameters of different types will be cached separately
    """
    def decorator(func):
        func = functools.lru_cache(maxsize=maxsize, typed=typed)(func)
        if protected:
            __protected_cached_functions.append(func)
        else:
            __cached_functions.append(func)
        return func

    return decorator


def clear_lru_caches(clear_protected: bool = False):
    """
    Clear all LRU caches.

    :param clear_protected: force-clear all caches, even protected ones
    """
    for func in __cached_functions:
        func.cache_clear()
    __cached_functions.clear()

    if clear_protected:
        for func in __protected_cached_functions:
            func.cache_clear()
        __protected_cached_functions.clear()


def get_base_path():
    """
    Get application base path.

    :return: absolute path to application directory
    """
    return os.path.realpath(os.path.dirname(sys.argv[0]))
