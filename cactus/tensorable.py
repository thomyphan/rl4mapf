from cactus.constants import *

class Tensorable:

    def __init__(self, device) -> None:
        self.device = device

    def stack(self, tensor_list):
        return torch.stack(tensor_list)
    
    def cat(self, tensor_list, dim=0):
        return torch.cat(tensor_list, dim=dim)

    def tensor(self, data, dtype):
        return torch.tensor(data, dtype=dtype, device=self.device)
    
    def bool_tensor(self, data):
        return self.tensor(data, BOOL_TYPE)
    
    def int_tensor(self, data):
        return self.tensor(data, INT_TYPE)
    
    def float_tensor(self, data):
        return self.tensor(data, FLOAT_TYPE)
    
    def as_tensor(self, data, dtype):
        return torch.as_tensor(data, dtype=dtype, device=self.device)
    
    def as_bool_tensor(self, data):
        return self.as_tensor(data, BOOL_TYPE)
    
    def as_int_tensor(self, data):
        return self.as_tensor(data, INT_TYPE)
    
    def as_float_tensor(self, data):
        return self.as_tensor(data, FLOAT_TYPE)
    
    def ones(self, dimensions, dtype):
        return torch.ones(dimensions, dtype=dtype, device=self.device)
    
    def bool_ones(self, dimensions):
        return self.ones(dimensions, BOOL_TYPE)
    
    def int_ones(self, dimensions):
        return self.ones(dimensions, INT_TYPE)
    
    def float_ones(self, dimensions):
        return self.ones(dimensions, FLOAT_TYPE)
    
    def zeros(self, dimensions, dtype):
        return torch.zeros(dimensions, dtype=dtype, device=self.device)
    
    def bool_zeros(self, dimensions):
        return self.zeros(dimensions, BOOL_TYPE)
    
    def int_zeros(self, dimensions):
        return self.zeros(dimensions, INT_TYPE)
    
    def float_zeros(self, dimensions):
        return self.zeros(dimensions, FLOAT_TYPE)
    
    def ones_like(self, other, dtype):
        return torch.ones_like(other, dtype=dtype, device=self.device)
    
    def bool_ones_like(self, other):
        return self.ones_like(other, BOOL_TYPE)
    
    def int_ones_like(self, other):
        return self.ones_like(other, INT_TYPE)
    
    def float_ones_like(self, other):
        return self.ones_like(other, FLOAT_TYPE)
    
    def zeros_like(self, other, dtype):
        return torch.zeros_like(other, dtype=dtype, device=self.device)
    
    def bool_zeros_like(self, other):
        return self.zeros_like(other, BOOL_TYPE)
    
    def int_zeros_like(self, other):
        return self.zeros_like(other, INT_TYPE)
    
    def float_zeros_like(self, other):
        return self.zeros_like(other, FLOAT_TYPE)