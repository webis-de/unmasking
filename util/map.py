from collections import OrderedDict


class BoundedHashMap:
    """
    Hash map with limited capacity.
    The hash map is bounded, meaning items will be removed from the map in FIFO order
    when the number of entries exceeds the specified maximum size.

    This is class is mainly intended for fast caching of results of expensive operations in memory.
    Due to this design, testing the map for (in)equality or hashing the map for using it as a key in
    another hash map are unsupported.
    """
    
    def __init__(self, maxlen: int = 0):
        """
        :param maxlen: maximum number of elements to store. If the map grows larger than this, elements
                       will be removed in FIFO order. Use 0 for unlimited size.
        """
        self._maxlen = 0
        self.maxlen = maxlen
        self._elements = OrderedDict()
    
    @property
    def maxlen(self) -> int:
        """Maximum number of entries inside the map."""
        return self._maxlen
    
    @maxlen.setter
    def maxlen(self, maxlen: int):
        """Set maximum number of entries after which older entries are removed."""
        if maxlen < 0:
            raise ValueError
        self._maxlen = maxlen
    
    def __getitem__(self, item):
        """
        :param item: pre-computed hash value
        :return: stored value
        """
        return self._elements[item]
    
    def __contains__(self, item):
        return item in self._elements
    
    def __setitem__(self, key, value):
        self._elements[key] = value
        
        if 0 < self.maxlen < len(self._elements):
            self._elements.popitem()
    
    def __delitem__(self, key):
        # create new OrderedDict instead of using del since it's usually faster for large dictionaries
        self._elements = OrderedDict(((k, v) for (k, v) in self._elements.items() if k != key))
    
    def __iter__(self):
        return iter(self._elements)
    
    def __eq__(self, other):
        """Not implemented for performance reasons."""
        raise NotImplementedError("bounded hash map not testable for equality")
    
    def __ne__(self, other):
        """Not implemented for performance reasons."""
        raise NotImplementedError("bounded hash map not testable for inequality")
    
    def __hash__(self):
        """Not implemented for performance reasons."""
        raise TypeError("bounded hash map not hashable")
