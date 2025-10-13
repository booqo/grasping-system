# Graspnet All In One

## Description

本项目是使用Mujoco环境下的UR5机器人进行抓取仿真的验证平台，目前已经用于验证箱子中的无序抓取算法和基于Qwen2.5VLM的机械臂交互抓取算法。在公司自己的仿真平台出来之前，其他和抓取相关的代码也可以考虑使用此方法验证。上机代码不建议在该环境中验证！！！！（如有需要，开发motioncontrol下的aurorapy，如https://gitlab.fftaicorp.com/motioncontrol/aurorapy/-/tree/qwen_robot/v0.1.0?ref_type=heads）

## clone 代码

```bash
git remote add origin http://gitlab.fftaicorp.com/systemeng/graspnet.git
git checkout feature/graspnet_AIO
```

## 准备工作

### 创建conda环境：

```bash
conda create -n graspnet_AIO python=3.9
conda activate graspnet_AIO
cd graspnet-baseline
```

### 安装torch

查看CUDA版本

```bash
nvidia-smi
```

这里本文安装的CUDA版本为11.3，这里选择安装官方 PyTorch 的 `cu121` 版本

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

安装完成后，检查一下是否能正确识别 GPU：

```
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

理想输出示例（版本可能不同）：

```
1.12.1 11.3 True NVIDIA GeForce RTX 4060 Laptop GPU
```

### 编译并安装pointnet2算子（代码改编自votenet）

```bash
cd graspnet-baseline
cd pointnet2
python setup.py install
cd ../
```

### 编译并安装knn算子（代码改编自pytorch_knn_cuda）

```bash
cd knn
python setup.py install
cd ../
```

### 安装 graspnetAPI 以进行评估

```bash
cd graspnetAPI
pip install .
cd ../
```

权重文件已经下好，具体在**logs/log_rs**下

### 安装Clip

```bash

pip install git+https://github.com/openai/CLIP.git
```

### 安装其他环境依赖

```bash
pip install --no-cache-dir -r requirements.txt
```

## 运行项目

QwenVLM接入测试，服务器端必须正确运行了大模型api转发程序`qwen_openai_api.py`，

```
python main_Qwen_grasp.py 
```



箱子中物体无序抓取测试

```bash
NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia MUJOCO_GL=glfw python main_box_grasp.py
```

<video src="./assets/grasp.mp4" controls="controls" width="800" height="600"></video>

yolo抓取测试

````bash
python main_yoloWorld_sam.py
````

服务器大模型API测试

```bash
python test_api.py
```



