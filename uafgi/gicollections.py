import collections.abc

# https://codereview.stackexchange.com/questions/173045/mutable-named-tuple-or-slotted-data-structure
class MutableNamedTuple(collections.abc.Sequence): 
    """Abstract Base Class for objects as efficient as mutable
    namedtuples. 
    Subclass and define your named fields with __slots__.
    """
    __slots__ = ()
    def __init__(self, *args):
        for slot, arg in zip(self.__slots__, args):
            setattr(self, slot, arg)
    def __repr__(self):
        return type(self).__name__ + repr(tuple(self))
    # more direct __iter__ than Sequence's
    def __iter__(self): 
        for name in self.__slots__:
            yield getattr(self, name)
    # Sequence requires __getitem__ & __len__:
    def __getitem__(self, index):
        return getattr(self, self.__slots__[index])
    def __len__(self):
        return len(self.__slots__)
