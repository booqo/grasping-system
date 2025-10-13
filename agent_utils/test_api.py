import base64
import os
import mimetypes
from openai import OpenAI
from .visual_utils import plot_bounding_boxes,plot_points
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import torch


client = OpenAI(base_url="http://10.18.138.8:8000/v1", api_key="EMPTY")

def _to_data_url_from_array(img: np.ndarray, mime: str = "image/jpeg") -> str:
    """将 numpy 数组转为 data URL。"""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:  # 灰度
        pil_img = Image.fromarray(img, mode="L")
    elif img.ndim == 3:
        if img.shape[2] == 1:
            pil_img = Image.fromarray(img.squeeze(-1), mode="L")
        elif img.shape[2] == 3:
            pil_img = Image.fromarray(img, mode="RGB")
        elif img.shape[2] == 4:
            pil_img = Image.fromarray(img, mode="RGBA")
        else:
            raise ValueError(f"不支持的通道数: {img.shape}")
    else:
        raise ValueError(f"不支持的数组形状: {img.shape}")

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")  # 或 "PNG"，取决于 mime
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}", pil_img

def _to_data_url_from_path(path: str) -> (str, Image.Image):
    abs_path = os.path.abspath(path)
    mime, _ = mimetypes.guess_type(abs_path)
    if mime is None:
        mime = "image/jpeg"
    with open(abs_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    pil_img = Image.open(abs_path).convert("RGB")
    return f"data:{mime};base64,{b64}", pil_img

def inference(img_input=None, prompt="", system_prompt="You are a helpful assistant", max_new_tokens=1024):
    """
    img_input 可以是：
      - str: 本地路径或 http(s) URL
      - np.ndarray: HWC 格式图像 (0~255)
      - torch.Tensor: HWC 格式张量 (0~1 或 0~255)
    """
    # 1) 统一成 PIL.Image + data_url
    if isinstance(img_input, str):
        if img_input.startswith("http://") or img_input.startswith("https://"):
            data_url = img_input
            pil_img = None
        else:  # 本地路径
            data_url, pil_img = _to_data_url_from_path(img_input)
    elif isinstance(img_input, np.ndarray):
        data_url, pil_img = _to_data_url_from_array(img_input)
    elif torch is not None and isinstance(img_input, torch.Tensor):
        arr = img_input.detach().cpu().numpy()
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        data_url, pil_img = _to_data_url_from_array(arr)
    elif img_input == None:
        messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        },
    ]
        # 3) 调用 VLM API
        resp = client.chat.completions.create(
            model="Qwen2.5-VL-7B-Instruct",
            messages=messages,
            temperature=0.2,
            max_tokens=max_new_tokens,
        )
        text = resp.choices[0].message.content

        return text

    else:
        raise TypeError(f"不支持的输入类型: {type(img_input)}")

    input_width, input_height = (pil_img.size if pil_img else (None, None))

    # 2) 组织消息
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    # 3) 调用 VLM API
    resp = client.chat.completions.create(
        model="Qwen2.5-VL-7B-Instruct",
        messages=messages,
        temperature=0.2,
        max_tokens=max_new_tokens,
    )
    text = resp.choices[0].message.content

    return text#, input_height, input_width


def getobjectbbox(image, prompt, system_prompt=None, max_new_tokens=1024):
    if system_prompt is None:
        system_prompt = """你是一个图像识别专家，专门负责从图像中检测和定位物体。你的任务是根据用户提供的提示，识别图像中的相关物体，并以json格式返回它们的二维边界框坐标。"""
    obj_json = inference(image, prompt, system_prompt, max_new_tokens)
    return obj_json

def getobjectpoints(image, prompt, system_prompt=None, max_new_tokens=1024):
    if system_prompt is None:
        system_prompt = """你是一个图像识别专家，专门负责从图像中检测和定位物体。你的任务是根据用户提供的提示，识别图像中的相关物体，并以xml格式输出其坐标。"""
    obj_xml, input_height, input_width = inference(image, prompt, system_prompt, max_new_tokens)
    return obj_xml, input_height, input_width


if __name__ == "__main__":

    image_path = "detection_visualization.jpg"

    # out, input_width, input_height = inference(image_path=image_path, prompt="请描述这张图片。")


    prompt = """任务：在输入图像中检测“香蕉”的2D边界框。"""

    prompt = """任务：以点的形式定位图箱子中香蕉，桌子和苹果的位置,以XML格式输出其坐标。
    要求："""

    response, input_height, input_width = getobjectpoints(image_path, prompt)
    image = Image.open(image_path)
    print(image.size)
    print(response)
    # image.thumbnail([640,640], Image.Resampling.LANCZOS)
    plot_points(image,response,input_width,input_height)

    # prompt = "以点的形式定位图中桌子远处的擀面杖，以XML格式输出其坐标"
    # response, input_height, input_width = inference(image_path, prompt)
    # image = Image.open(image_path)
    # image.thumbnail([640,640], Image.Resampling.LANCZOS)
    # plot_points(image, response, input_width, input_height)