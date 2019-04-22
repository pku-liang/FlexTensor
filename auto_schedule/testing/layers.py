"""
Base class for layers

====================================
**Author**: `Size Zheng`
"""
import tvm 


class Layer(object):
    """Base class for layers

    """
    def __init__(self):
        self.params = {}

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def parameters(self):
        return self.params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)