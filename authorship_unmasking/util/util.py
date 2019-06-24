# Copyright (C) 2017-2019 Janek Bevendorff, Webis Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import os

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
    return os.path.realpath(os.path.join(os.path.dirname(__file__), '../..'))


class SoftKeyboardInterrupt(Exception):
    """
    Replacement for KeyboardInterrupt that inherits from :class:: Exception instead of
    :class:: BaseException, to avoid uncatchable stack traces when a :class:: KeyboardInterrupt
    happens within a coroutine.
    See: https://github.com/python/asyncio/issues/341
    """
    pass


async def base_coroutine(cr):
    """
    Base coroutine that wraps and waits another coroutine and catches KeyboardInterrupts.
    Caught keyboardInterrupts are re-raised as SoftKeyboardInterrupts.

    :param cr: coroutine to wrap
    :return: Return value of the wrapped coroutine
    """
    try:
        return await cr
    except KeyboardInterrupt as k:
        raise SoftKeyboardInterrupt() from k


def run_in_event_loop(coroutine):
    """
    Wrap and run coroutine in event loop.

    :param coroutine: coroutine to run in the event loop
    """
    try:
        stop_when_done = False
        loop = asyncio.get_event_loop()
    except RuntimeError:
        stop_when_done = True
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.set_default_executor(ThreadPoolExecutor(max_workers=os.cpu_count()))
    try:
        future = asyncio.ensure_future(base_coroutine(coroutine))
        loop.run_until_complete(future)
    finally:
        loop.run_until_complete(base_coroutine(loop.shutdown_asyncgens()))
        if stop_when_done:
            loop.stop()
