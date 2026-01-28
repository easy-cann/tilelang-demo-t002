from pathlib import Path
import torch
import ctypes

torch.set_default_device("npu")

B, S, H, D = 1, 128, 1, 512

q = torch.randn((B, H, S, D), dtype=torch.float16)
k = torch.randn((B, H, S, D), dtype=torch.float16)
v = torch.randn((B, H, S, D), dtype=torch.float16)
output = torch.empty((B, H, S, D), dtype=torch.float16)

w1 = torch.randn((2, 64, 64), dtype=torch.float)
w2 = torch.randn((2, 64, 64), dtype=torch.float16)
w3 = torch.randn((2, 64, 512), dtype=torch.float)

SCRIPT_DIR = Path(__file__).resolve().parent
lib = ctypes.CDLL(f"{SCRIPT_DIR}/test_flash_attention.so")
result = lib.call(
    ctypes.c_void_p(q.data_ptr()),
    ctypes.c_void_p(k.data_ptr()),
    ctypes.c_void_p(v.data_ptr()),
    ctypes.c_void_p(output.data_ptr()),
    ctypes.c_void_p(w1.data_ptr()),
    ctypes.c_void_p(w1.data_ptr()),
    ctypes.c_void_p(w1.data_ptr()),
    torch.npu.current_stream()._as_parameter_
)

torch.npu.synchronize()
print(f"Kernel Output Is: {output}")

def ref_flash_attn(q, k, v):
    q = q.float()
    k = k.float()
    v = v.float()

    acc = torch.einsum("bhsd,bhkd->bhsk", q, k) * (1.0 / q.shape[-1])**0.5
    acc = acc.softmax(dim=-1)
    o = torch.einsum("bhsk,bhkd->bhsd", acc, v)
    return o.to(torch.float16)

ref_output = ref_flash_attn(q, k, v)
print(f"Golden Output Is: {ref_output}")

torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")



