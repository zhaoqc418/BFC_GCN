from .test_gcn import test_gcn
from .train_gcn import train_gcn
from .test_gcn_centraility import test_gcn

__factory__ = {
    # 'test_gcn': test_gcn,
    'test_gcn': test_gcn_centraility,

    'train_gcn': train_gcn,
}


def build_handler(phase):
    key_handler = '{}_gcn'.format(phase)
    if key_handler not in __factory__:
        raise KeyError("Unknown op:", key_handler)
    return __factory__[key_handler]
