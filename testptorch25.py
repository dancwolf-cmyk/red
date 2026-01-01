import torch
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel

print("PyTorch:", torch.__version__)
print("Device:", torch.cuda.get_device_name(0))

# 强制启用 Flash Attention / Memory-efficient / Math fallback
sdp_kernel.enable_flash = True
sdp_kernel.enable_mem_efficient = True
sdp_kernel.enable_math = True

# 测试张量（512 token、8 heads）
q = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.float16)
k = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.float16)
v = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.float16)

with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_mem_efficient=True, enable_math=True):
    out = F.scaled_dot_product_attention(q, k, v)

print("Output shape:", out.shape)
print("OK —— FlashAttention 已启用（如果没报错）")
