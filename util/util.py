import functools


__cached_functions = []


def lru_cache(*args, **kwargs):
    """
    Wrapped version of the functools.lru_cache decorator which
    keeps track of all decorated functions to allow clearing all caches.
    """
    def decorator(func):
        func = functools.lru_cache(*args, **kwargs)(func)
        __cached_functions.append(func)
        return func

    return decorator


def clear_lru_caches():
    """
    Clear all LRU caches.
    """
    for func in __cached_functions:
        func.cache_clear()
    __cached_functions.clear()
