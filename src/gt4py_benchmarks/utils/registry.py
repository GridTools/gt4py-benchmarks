def subclass_registry(cls):
    def scr(cls):
        direct_subclasses = cls.__subclasses__()
        return set(direct_subclasses).union(*(scr(c) for c in direct_subclasses))

    cls.subclass_registry = classmethod(scr)

    return cls
