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
    
    @staticmethod
    def init_cache(size: int = 0) -> int:
        """
        Initialize a new cache with given size and returns its cache handle.
        Cache handles are positive integers (starting at 0).
        
        :param size: maximum cache size (0 means unlimited)
        :return: cache handle
        """
        CacheMixin.__cache[CacheMixin.__next_handle] = BoundedHashMap(size)
        CacheMixin.__next_handle += 1
        return CacheMixin.__next_handle - 1
    
    @staticmethod
    def uninit_cache(handle: int):
        """
        Un-initialize and delete a cache together with all its aliases.
        
        :param handle: cache handle
        """
        if handle in CacheMixin.__cache:
            for a in CacheMixin.__cache_aliases:
                if CacheMixin.__cache_aliases[a] == handle:
                    del CacheMixin.__cache_aliases[a]
            del CacheMixin.__cache[handle]
    
    @staticmethod
    def has_cache(handle: int) -> bool:
        """
        Whether a cache exists for the given handle.

        :param handle: integer handle of the cache element
        :return: True if cache exists
        """
        return handle in CacheMixin.__cache
    
    @staticmethod
    def set_cache_alias(handle: int, alias: str) -> bool:
        """
        Associate a cache handle with a recognizable string which can be used for
        retrieving a cache with unknown handle.
        Cache aliases are global and should therefore be uniquely prefixed (e.g. with a class name).
        
        :param handle: cache handle
        :param alias: alias name
        :return: True if alias could be set, False if another handle already has this alias or handle does not exist
        """
        if handle not in CacheMixin.__cache:
            return False
        
        if alias not in CacheMixin.__cache_aliases:
            CacheMixin.__cache_aliases[alias] = handle
            return True
        elif alias in CacheMixin.__cache_aliases and CacheMixin.__cache_aliases[alias] == handle:
            return True
        
        return False
    
    @staticmethod
    def resolve_cache_alias(alias: str) -> int:
        """
        Get cache handle for given alias.
        
        :param alias: alias name
        :return: cache handle or -1 if alias does not exist
        """
        if alias not in CacheMixin.__cache_aliases:
            return -1
        return CacheMixin.__cache_aliases[alias]
    
    @staticmethod
    def get_cache_size(handle: int) -> int:
        """Get size of cache ``handle``"""
        return CacheMixin.__cache[handle].maxlen
    
    @staticmethod
    def set_cache_size(handle: int, size: int):
        """Set size of cache ``handle``"""
        CacheMixin.__cache[handle].maxlen = size
    
    @staticmethod
    def set_cache_item(handle: int, cache_key, cache_value):
        """
        Cache a key-value pair.
        Raises :class:`ValueError` if cache ``handle`` does not exist.
        
        :param handle: cache handle
        :param cache_key: key of the cached item
        :param cache_value: value of the cached item
        :return:
        """
        if handle not in CacheMixin.__cache:
            raise ValueError("Cache handle '{}' has not been initialized!".format(handle))
        CacheMixin.__cache[handle][cache_key] = cache_value
    
    @staticmethod
    def get_cache_item(handle: int, cache_key, default_value=None):
        """
        Get value for given key from cache ``handle``.
        Raises :class:`ValueError` if cache ``handle`` does not exist.
        
        :param handle: cache handle
        :param cache_key: key of the cached item
        :param default_value: default value to return if cache ``handle`` has no such key
        :return: cached value
        """
        if handle not in CacheMixin.__cache:
            raise ValueError("Cache handle '{}' has not been initialized!".format(handle))
        
        if cache_key in CacheMixin.__cache[handle]:
            return CacheMixin.__cache[handle][cache_key]
        
        return default_value
    
    @staticmethod
    def reset_caches():
        """
        Invalidate / clear all caches. It is recommended to do this before starting a new experiment.
        All existing cache handles and aliases will become invalid.
        """
        CacheMixin.__cache = {}
        CacheMixin.__cache_aliases = {}
        CacheMixin.__next_handle = 0
