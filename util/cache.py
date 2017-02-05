from util.map import BoundedHashMap


class CacheMixin:
    """
    Mix-in class for statically caching data in memory.
    Before items can be stored, a cache needs to be initialized with :method:`init_cache()`.
    Caches have a limited pre-defined size which is counted in "items".
    """
    
    __cache = {}
    __cache_aliases = {}
    __next_handle = 0
    
    @classmethod
    def init_cache(cls, size: int = 0) -> int:
        """
        Initialize a new cache with given size and returns its cache handle.
        Cache handles are positive integers (starting at 0).
        
        :param size: maximum cache size (0 means unlimited)
        :return: cache handle
        """
        cls.__cache[cls.__next_handle] = BoundedHashMap(size)
        cls.__next_handle += 1
        return cls.__next_handle - 1
    
    @classmethod
    def uninit_cache(cls, handle: int):
        """
        Un-initialize and delete a cache together with all its aliases.
        
        :param handle: cache handle
        """
        if handle in cls.__cache:
            for a in cls.__cache_aliases:
                if cls.__cache_aliases[a] == handle:
                    del cls.__cache_aliases[a]
            del cls.__cache[handle]
    
    @classmethod
    def has_cache(cls, handle: int) -> bool:
        """
        Whether a cache exists for the given handle.

        :param handle: integer handle of the cache element
        :return: True if cache exists
        """
        return handle in cls.__cache
    
    @classmethod
    def set_cache_alias(cls, handle: int, alias: str) -> bool:
        """
        Associate a cache handle with a recognizable string which can be used for
        retrieving a cache with unknown handle.
        Cache aliases are global and should therefore be uniquely prefixed (e.g. with a class name).
        
        :param handle: cache handle
        :param alias: alias name
        :return: True if alias could be set, False if another handle already has this alias or handle does not exist
        """
        if handle not in cls.__cache:
            return False
        
        if alias not in cls.__cache_aliases:
            cls.__cache_aliases[alias] = handle
            return True
        elif alias in cls.__cache_aliases and cls.__cache_aliases[alias] == handle:
            return True
        
        return False
    
    @classmethod
    def resolve_cache_alias(cls, alias: str) -> int:
        """
        Get cache handle for given alias.
        
        :param alias: alias name
        :return: cache handle or -1 if alias does not exist
        """
        if alias not in cls.__cache_aliases:
            return -1
        return cls.__cache_aliases[alias]
    
    @classmethod
    def get_cache_size(cls, handle: int) -> int:
        """Get size of cache ``handle``"""
        return cls.__cache[handle].maxlen
    
    @classmethod
    def set_cache_size(cls, handle: int, size: int):
        """Set size of cache ``handle``"""
        cls.__cache[handle].maxlen = size
    
    @classmethod
    def set_cache_item(cls, handle: int, cache_key, cache_value):
        """
        Cache a key-value pair.
        Raises :class:`ValueError` if cache ``handle`` does not exist.
        
        :param handle: cache handle
        :param cache_key: key of the cached item
        :param cache_value: value of the cached item
        :return:
        """
        if handle not in cls.__cache:
            raise ValueError("Cache handle '{}' has not been initialized!".format(handle))
        cls.__cache[handle][cache_key] = cache_value
    
    @classmethod
    def get_cache_item(cls, handle: int, cache_key, default_value=None):
        """
        Get value for given key from cache ``handle``.
        Raises :class:`ValueError` if cache ``handle`` does not exist.
        
        :param handle: cache handle
        :param cache_key: key of the cached item
        :param default_value: default value to return if cache ``handle`` has no such key
        :return: cached value
        """
        if handle not in cls.__cache:
            raise ValueError("Cache handle '{}' has not been initialized!".format(handle))
        
        if cache_key in cls.__cache[handle]:
            return cls.__cache[handle][cache_key]
        
        return default_value
