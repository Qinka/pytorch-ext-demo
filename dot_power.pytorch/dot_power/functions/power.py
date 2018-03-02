# functions/power.py

import torch
from torch.autograd import Function
from .._ext import dot_power

class DotPower(Function):
    def forward(self, base, exponent):
        self.save_for_backward(base,exponent)
        assert(base.is_contiguous() == True)
        assert(exponent.is_contiguous() == True)
        output = base.new().resize_as_(base).zero_()
        if input1.is_cuda:
            dot_power.dot_power_forward(base,exponent,output)
        else:
            raise NotImplementedError()
        return output

    def backward(self,grad_output):
        base,exponent = self.saved_tensors
        assert(grad_output.is_contiguous() == True)
        grad_base = base.new().resize_as_(base).zero_()
        grad_exponent =  exponent.new().resize_as_(base).zero_()
        if input1.is_cuda:
            dot_power.dot_power_backward(base,exponent,grad_output,grad_base,grad_exponent)
        else:
            raise NotImplementedError()
        return grad_base,grad_exponent
