
import torch
import requests
from PIL import Image
import torch.nn.functional as F
from .clipseg import CLIPDensePredT
from torchvision import transforms
from PIL import Image
from typing import Optional, Union
import matplotlib.pyplot as plt
import time
import numpy as np



class BoxSegmenter:
    def __init__(self, model_path='/home/zousf/code/graspnet-baseline/utils/weights/rd64-uni.pth', device='cuda'):
        # 初始化模型[1](@ref)
        self.device = torch.device(device)
        self.model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64).to(self.device)
        self.model.eval()
        self.prompt = "a box"

        # 加载预训练权重（仅Decoder部分）[1](@ref)
        try:
            # 新版 PyTorch（≥2.4）支持 weights_only，安全性更好
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            # 旧版不支持该参数，回退到传统加载
            state_dict = torch.load(model_path, map_location=self.device)

        # 若 checkpoint 外层包了一层（常见：{'state_dict': ...}）
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        self.model.load_state_dict(state_dict, strict=False)

        #图像预处理流程[1](@ref)
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((352, 352))
        ])

    def morph_postprocess(self,mask: torch.Tensor,
                      open_ks: int = 3,# 小孔/孤立点的去除（先腐蚀再膨胀）
                      close_ks: int = 5,# 边缘闭合（先膨胀再腐蚀）
                      fill_holes: bool = True,# 填充孔洞（SciPy 可用时生效）
                      keep_largest: bool = True,# 仅保留最大连通域（SciPy 可用时生效）
                      min_area: Optional[int] = None) -> torch.Tensor: # 或设为整数，如 500，过滤小区域
        """
        对二值 mask 做形态学处理，返回 CPU 上的 bool Tensor（与输入同尺寸）。
        - 优先使用 SciPy；无 SciPy 时退化为 PyTorch 的开/闭运算。
        """
        try:
            from scipy import ndimage as ndi
            _HAS_SCIPY = True
        except Exception:
            _HAS_SCIPY = False
            import torch.nn.functional as F
        mask = mask.detach().to("cpu").bool()

        if _HAS_SCIPY:
            m = mask.numpy().astype(np.uint8)

            if open_ks and open_ks > 1:
                s = np.ones((open_ks, open_ks), dtype=np.uint8)
                m = ndi.binary_opening(m, structure=s)

            if close_ks and close_ks > 1:
                s = np.ones((close_ks, close_ks), dtype=np.uint8)
                m = ndi.binary_closing(m, structure=s)

            if fill_holes:
                m = ndi.binary_fill_holes(m)

            if keep_largest or (min_area is not None and min_area > 0):
                labels, num = ndi.label(m)
                if num > 0:
                    sizes = ndi.sum(m, labels, index=np.arange(1, num + 1))
                    if keep_largest:
                        keep_id = 1 + int(sizes.argmax())
                        m = (labels == keep_id)
                    if min_area is not None and min_area > 0:
                        keep_ids = 1 + np.nonzero(sizes >= min_area)[0]
                        if keep_ids.size > 0:
                            m = np.isin(labels, keep_ids)
                        else:
                            m = np.zeros_like(m, dtype=bool)
            return torch.from_numpy(m.astype(bool))

        else:
            # —— 无 SciPy：仅做开闭运算（PyTorch版） ——
            def _dilate_bin(t: torch.Tensor, k: int, iters: int = 1):
                if k <= 1 or iters <= 0:
                    return t
                x = t.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                pad = k // 2
                for _ in range(iters):
                    x = F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
                return (x.squeeze(0).squeeze(0) > 0.5)

            def _erode_bin(t: torch.Tensor, k: int, iters: int = 1):
                return ~_dilate_bin(~t, k, iters)

            m = mask
            if open_ks and open_ks > 1:
                m = _dilate_bin(_erode_bin(m, open_ks), open_ks)
            if close_ks and close_ks > 1:
                m = _erode_bin(_dilate_bin(m, close_ks), close_ks)
            return m.bool()

    def predict_mask(self, image_input):#, prompt):
        if isinstance(image_input, str):
            if image_input.startswith('http'):
                img = Image.open(requests.get(image_input, stream=True).raw)
            else:
                img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input,np.ndarray):
            arr = image_input

            # [fix] 如果是 CHW → HWC
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))

            # [fix] 去除 HWC 中的单通道维度
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0]  # 变为 HxW

            # [fix] 将 float 转为 uint8（支持 [0,1] 或 [0,255]）
            if arr.dtype in (np.float32, np.float64):
                minv, maxv = float(np.min(arr)), float(np.max(arr))
                if 0.0 <= minv and maxv <= 1.0:
                    arr = (arr * 255.0).round().astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).round().astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)

            # [fix] OpenCV 常见为 BGR，这里统一转为 RGB（仅对 3 通道生效）
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = arr[..., ::-1]  # BGR → RGB

            img = Image.fromarray(arr)
        else:
            raise ValueError("输入类型需为路径/PIL图像/URL/np")

        # 记录原始尺寸用于结果还原[1](@ref)
        original_size = img.size  # (width, height)

    
        img_tensor = self.transform2(img).unsqueeze(0).to(self.device)
        # 模型推理[1](@ref)

        prompt = self.prompt
        with torch.no_grad():
            pred = self.model(img_tensor, [prompt])[0]
            prob_mask = torch.sigmoid(pred[0][0])  # 概率图


        # 后处理
        mask = (prob_mask > 0.45).float()  # 二值化阈值
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=original_size[::-1],  # 目标尺寸(height, width)
            mode='bilinear'
        ).squeeze()
        mask = self.morph_postprocess(mask = mask)

        return mask # tensor bool ([H,W])
    
if __name__ == "__main__":
    mask = BoxSegmenter()
    # prompt = "bin, box, plastic bin, storage bin, container, tray, tote"
    prompt = "a box"
    test_image = Image.open("../our_data/JPEGImages/57.jpg")
    box_mask = mask.predict_mask(test_image)#,prompt=prompt)
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    # plt.figure("mask image")
    axes[0].imshow(box_mask.detach().cpu().numpy())
    axes[1].imshow(test_image)
    plt.show()

# if __name__ == "__main__":
#     import os
#     import numpy as np
#     from PIL import Image
#     import matplotlib.pyplot as plt

#     img_dir = "/home/zousf/code/graspnet-baseline/our_data/JPEGImages"
#     out_dir = "/home/zousf/code/graspnet-baseline/our_data/mask_results"
#     os.makedirs(out_dir, exist_ok=True)

#     seg = BoxSegmenter()
#     prompt = "a box"

#     for fname in sorted(os.listdir(img_dir)):
#         if not fname.lower().endswith('.jpg'):
#             continue

#         stem = os.path.splitext(fname)[0]
#         color_img = Image.open(os.path.join(img_dir, fname)).convert("RGB")

#         # 获取 mask
#         mask = seg.predict_mask(color_img)#, prompt=prompt)  # bool tensor
#         if not mask.any():
#             print(f"{fname}: 未检测到目标")
#             continue

#         # 转 numpy
#         mask_np = mask.numpy()

#         # 原图叠加可视化
#         overlay = np.array(color_img).copy()
#         overlay[mask_np] = (255, 0, 0)  # mask 区域染红

#         # 绘图：左=mask，中=叠加，右=原图
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#         axes[0].imshow(mask_np, cmap='gray')
#         axes[0].set_title("Mask")
#         axes[0].axis('off')

#         axes[1].imshow(overlay)
#         axes[1].set_title("Overlay")
#         axes[1].axis('off')

#         axes[2].imshow(color_img)
#         axes[2].set_title("Original")
#         axes[2].axis('off')

#         plt.tight_layout()
#         plt.show()

        # # 保存 mask 图（白=255，黑=0）
        # mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
        # mask_img.save(os.path.join(out_dir, f"{stem}_mask.png"))

        # # 保存叠加图
        # Image.fromarray(overlay).save(os.path.join(out_dir, f"{stem}_overlay.jpg"))

        # print(f"{fname}: mask/overlay 已保存到 {out_dir}")





# import torch
# import requests


# from clipseg import CLIPDensePredT
# from PIL import Image
# from torchvision import transforms
# from matplotlib import pyplot as plt

# # load model
# model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
# model.eval()

# # non-strict, because we only stored decoder weights (not CLIP weights)
# model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False)

# # load and normalize image
# input_image = Image.open('../our_data/JPEGImages/0.jpg')

# # or load from URL...
# # image_url = 'https://farm5.staticflickr.com/4141/4856248695_03475782dc_z.jpg'
# # input_image = Image.open(requests.get(image_url, stream=True).raw)

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     transforms.Resize((352, 352)),
# ])
# img = transform(input_image).unsqueeze(0)

# prompts = ['a black box', 'bin',  'big bin',"a box"]

# # predict
# with torch.no_grad():
#     preds = model(img.repeat(4,1,1,1), prompts)[0]

# # visualize prediction
# _, ax = plt.subplots(1, 5, figsize=(15, 4))
# [a.axis('off') for a in ax.flatten()]
# ax[0].imshow(input_image)
# [ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(4)]
# [ax[i+1].text(0, -15, prompts[i]) for i in range(4)]
# plt.show()