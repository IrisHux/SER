import torch
import sys

print("--- 1. Check PyTorch and CUDA ---")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")

# Check if CUDA is available
is_cuda_available = torch.cuda.is_available()
print(f"Is CUDA available: {is_cuda_available}")

if not is_cuda_available:
    print("\n[ERROR] PyTorch could not detect CUDA. Please check your NVIDIA driver and PyTorch installation.")
    # If CUDA is not available, exit the script
    sys.exit()

print("\n--- 2. Get GPU Device Information ---")
# Get the default CUDA device (usually GPU 0)
device = torch.device("cuda:0")
print(f"Default CUDA device: {device}")

# Print the name of the GPU
gpu_name = torch.cuda.get_device_name(0)
print(f"GPU Name: {gpu_name}")

# Print the CUDA version PyTorch was compiled with
torch_cuda_version = torch.version.cuda
print(f"PyTorch compiled with CUDA version: {torch_cuda_version}")


print("\n--- 3. Test Data Transfer Between CPU and GPU ---")
# a. Create a tensor on the CPU
cpu_tensor = torch.tensor([1, 2, 3], device='cpu')
print(f"a. Tensor created on the CPU: {cpu_tensor}")
print(f"   - Device: {cpu_tensor.device}")

# b. Try to move the tensor to the GPU
try:
    gpu_tensor = cpu_tensor.to(device)
    print(f"\nb. Successfully moved tensor to GPU: {gpu_tensor}")
    print(f"   - Device: {gpu_tensor.device}")
except Exception as e:
    print(f"\n[ERROR] Failed to move data to GPU: {e}")
    sys.exit()


print("\n--- 4. Test Computation on GPU ---")
# a. Create two tensors on the GPU for computation
try:
    a = torch.randn(3, 3).to(device)
    b = torch.randn(3, 3).to(device)
    print(f"a. Created two 3x3 random tensors on the GPU.")
    print(f"   - Tensor a device: {a.device}")
    print(f"   - Tensor b device: {b.device}")

    # b. Perform matrix multiplication on the GPU
    print("\nb. Performing matrix multiplication on GPU (c = a * b)...")
    c = torch.matmul(a, b)
    print(f"   - Result c device: {c.device}")
    print(f"   - Computation successful!")

except Exception as e:
    print(f"\n[ERROR] Computation on GPU failed: {e}")
    sys.exit()

print("\n--- 5. Test Moving Result Back to CPU ---")
# a. Move the computation result from GPU back to CPU
try:
    result_cpu_tensor = c.cpu()
    print("a. Successfully moved the computation result back to the CPU.")
    print(f"   - Device: {result_cpu_tensor.device}")
    print("\nComputation result:")
    print(result_cpu_tensor)

except Exception as e:
    print(f"\n[ERROR] Failed to move result back to CPU: {e}")
    sys.exit()


print("\n--- All tests completed ---")
print("[SUCCESS] Your PyTorch and CUDA environment is configured correctly, and they can communicate and perform computations normally!")