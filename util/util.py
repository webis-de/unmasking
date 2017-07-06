import functools


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
