class _RegistryHelper(object):
    """Helper class."""
    # Global dictionary mapping subclass name to registered params.
    _PARAMS = {}

    @classmethod
    def Register(cls, real_cls):
        k = real_cls.__name__
        assert k not in cls._PARAMS, '{} already registered!'.format(k)
        cls._PARAMS[k] = real_cls
        return real_cls


Register = _RegistryHelper.Register


def Get(name):
    if name not in _RegistryHelper._PARAMS:
        raise LookupError('{} not found in registered params: {}'.format(
            name, list(sorted(_RegistryHelper._PARAMS.keys()))))
    p = _RegistryHelper._PARAMS[name]().Params()
    return p


def GetAll():
    return dict(_RegistryHelper._PARAMS)