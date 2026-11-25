import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

__all__ = ["TritonRMSNorm2dFunc"]

# Fallback implementation when Triton is not available
class TritonRMSNorm2dFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        # Simple RMS norm implementation
        rms = torch.sqrt(torch.mean(x ** 2, dim=(2, 3), keepdim=True) + eps)
        normalized = x / rms
        output = normalized * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
        
        ctx.save_for_backward(x, weight, bias, rms)
        ctx.eps = eps
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, rms = ctx.saved_tensors
        eps = ctx.eps
        
        grad_x = grad_output * weight.view(1, -1, 1, 1) / rms
        grad_weight = (grad_output * (x / rms)).sum(dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        
        return grad_x, grad_weight, grad_bias, None
