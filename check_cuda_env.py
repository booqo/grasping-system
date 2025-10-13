import os
import torch
import subprocess
import ctypes

print("====== PyTorch & CUDA 环境检查 ======")

# 1. PyTorch 版本信息
torch_version = torch.__version__
torch_cuda = torch.version.cuda
is_available = torch.cuda.is_available()
device_count = torch.cuda.device_count()

print(f"PyTorch 版本       : {torch_version}")
print(f"编译的 CUDA 版本  : {torch_cuda}")
print(f"是否可用 cuda      : {is_available}")
print(f"检测到的 GPU 数量  : {device_count}")

gpu_name = None
if device_count > 0:
    try:
        gpu_name = torch.cuda.get_device_name(0)
        print("GPU 名称          :", gpu_name)
    except Exception as e:
        print("GPU 初始化失败    :", e)

# 2. nvidia-smi 信息
print("\n====== nvidia-smi 输出 ======")
driver_version = None
cuda_runtime = None
try:
    smi_out = subprocess.check_output(["nvidia-smi"], text=True)
    print(smi_out)
    for line in smi_out.splitlines():
        if "Driver Version" in line and "CUDA Version" in line:
            parts = line.split()
            driver_version = parts[2]
            cuda_runtime = parts[-1]
except Exception as e:
    print("运行 nvidia-smi 出错:", e)

# 3. 检查 PyTorch C 扩展实际链接的 CUDA 库
print("====== PyTorch 依赖的 CUDA 库 ======")
linked_libs = []
try:
    so_path = torch._C.__file__
    libs = subprocess.check_output(["ldd", so_path], text=True)
    for line in libs.splitlines():
        if "cuda" in line.lower():
            linked_libs.append(line.strip())
            print(line)
except Exception as e:
    print("检查 ldd 出错:", e)

# 4. 当前环境变量
print("\n====== 环境变量 ======")
for k in ["CUDA_HOME", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PATH"]:
    print(f"{k} = {os.environ.get(k)}")

# 5. 测试能否加载驱动的 libcuda
print("\n====== 测试 libcuda 加载 ======")
libcuda_loaded = False
try:
    libcuda = ctypes.CDLL("libcuda.so")
    print("成功加载 libcuda.so:", libcuda)
    libcuda_loaded = True
except Exception as e:
    print("加载 libcuda.so 失败:", e)

# 6. 自动诊断
print("\n====== 自动诊断 ======")
if not libcuda_loaded:
    print("❌ 无法加载驱动的 libcuda.so，说明驱动没装好或路径错误。请检查驱动安装。")

elif device_count == 0:
    print("❌ PyTorch 没检测到 GPU，请确认 GPU 是否在 nvidia-smi 中出现。")

elif device_count > 0 and not is_available:
    print("⚠️  PyTorch 能数出 GPU，但初始化失败。")
    if torch_cuda.startswith("12.1"):
        print("👉 你装的是 cu121 wheel，但驱动是", cuda_runtime, "。在 CUDA 12.8 驱动上 cu121 已知会报错。")
        print("✅ 建议切换到 cu124 wheel：")
        print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    elif torch_cuda.startswith("12.4"):
        print("👉 你已经是 cu124 版本，但还是报错。")
        if driver_version and driver_version.startswith("570"):
            print("⚠️  当前驱动是 570.x (CUDA 12.8)，可能太新。")
            print("✅ 建议降级到 550.x 驱动（CUDA 12.4），保证和 cu124 对齐。")
        else:
            print("⚠️  驱动版本未知，可能存在 ABI 不兼容。")
    else:
        print("⚠️  你的 torch CUDA 版本 =", torch_cuda, "，可能和驱动 CUDA", cuda_runtime, "不匹配。")
        print("✅ 建议安装与驱动最接近的 PyTorch wheel (cu121 或 cu124)。")

else:
    print("✅ 一切正常，CUDA 初始化成功。")
