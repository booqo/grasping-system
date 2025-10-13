import os
import sys
import numpy as np
import open3d as o3d
import scipy.io as scio
import torch
from PIL import Image
import spatialmath as sm
import cv2
import mujoco
import base64
import json
from openai import OpenAI
from graspnetAPI import GraspGroup
from utils.robot_agent_tools import RuntimeStore, PerceptionUse, GraspPlanUse, MotionUse
from utils.prompt import AGENT_SYS_PROMPT,AGENT_SYS_PROMPT_OPENAI
from qwen_agent.tools.base import BaseTool, register_tool
from utils.qwen_utils import parse_agent_json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from gradio_client import Client

from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv
from copy import deepcopy
from cv_process import segment_image

from utils.grasp_utils import generate_grasps, execute_grasp
# from agent_function import camera_check
from agent_utils.test_api import inference, getobjectbbox,getobjectpoints,plot_bounding_boxes,plot_points

SYSTEM_PROMPT_CATCH = '''
我即将说一句给机械臂的指令，你帮我从这句话中提取出起始物体和终止物体，并从这张图中分别找到这两个物体左上角和右下角的像素坐标，输出json数据结构。
如果只有一個物體，比如“抓取桌面上的苹果”，那就只需要找到這一個物體的左上角和右下角的像素坐标，输出json数据结构。

例如，如果我的指令是：请帮我把红色方块放在房子简笔画上。
你输出这样的格式：
[
        {"bbox_2d": [225, 341, 260, 522], "label": "红色方块"},
        {"bbox_2d": [100, 200, 200, 300], "label": "房子简笔画"}
]
注意第一個bbox_2d是起始物体，第二個是终止物体
例如，如果我的指令是：抓取桌面上的苹果。
你输出这样的格式：
[
        {"bbox_2d": [225, 341, 260, 522], "label": "红苹果"},
]
只回复json本身即可，不要回复其它内容
我现在的指令是：
'''

SYSTEM_PROMPT_POINT = """
我即将说一句给机械臂的指令，你帮我从这句话中提取出指定物品信息，并从这张图中識別相關的物體，以xml格式输出其坐标。

例如，如果我的指令是：定位图中香蕉，桌子和苹果的位置
你输出这样的格式：
```xml
<points x1="237" y1="460" alt="香蕉">香蕉</points>
<points x1="298" y1="456" alt="桌子">桌子</points>
<points x1="334" y1="397" alt="苹果">苹果</points>
```
只回复xml本身即可，不要回复其它内容
我现在的指令是：

"""

# ```xml
# <points x1="237" y1="460" alt="箱子中香蕉">箱子中香蕉</points>
# <points x1="298" y1="456" alt="桌子">桌子</points>
# <points x1="334" y1="397" alt="苹果">苹果</points>
# ```

SYSTEM_PROMPT_VQA = '''
告诉我图片中每个物体的名称、类别和作用。每个物体用一句话描述。

例如：
连花清瘟胶囊，药品，治疗感冒。
盘子，生活物品，盛放东西。
氯雷他定片，药品，治疗抗过敏。

我现在的指令是：
'''

class Qwen_agent:
    def __init__(self, system_prompt: str, client: OpenAI):
        self.client = client
        self.system_prompt = system_prompt

    def get_response(self, prompt: list, image_path=None) -> dict:
        """
        messages: 形如 [{"role":"user","content":"..."}] 或包含图像部件的 OpenAI 格式消息
        返回: {"function":[...], "response":"..."}
        """

        rsp = inference(image_path,prompt,self.system_prompt)

        msg = rsp
        print(rsp)
        raw = msg if isinstance(msg, str) else str(msg)

        # ✅ 解析“回退 JSON”，并做一次兜底
        try:
            agent_plan_output = parse_agent_json(raw)  # 期望 {"function":[...], "response":"..."}
        except Exception:
            # 兜底：当模型未按 JSON 返回时，给出空函数 + 简短回应
            agent_plan_output = {"function": [], "response": "请仅输出JSON: function与response"}

        return agent_plan_output
    
    def vlm_get_bbox_postion(self, PROMPT: str, image_path:str) -> list:
        """
        PROMPT: 
        返回: 開始物體和終止物體的bbox像素座標字典{"lable":[abs_x1,abs_y1,abs_x2,abs_y2]}
        """
        # ✅ 不要 in-place 修改外部传入的 messages；避免副作用
        response  = getobjectbbox(image=image_path,prompt=PROMPT, system_prompt=SYSTEM_PROMPT_CATCH)

        image = Image.open(image_path)
        input_height, input_width = image.size
        # image.thumbnail([640,640], Image.Resampling.LANCZOS)# 改爲縮略圖 測試用，實際運行注意註釋掉
        result = plot_bounding_boxes(image,response,input_height, input_width) # 可視化box，導出像素座標字典
        return result #{"lable":[abs_x1,abs_y1,abs_x2,abs_y2]}
    
    def vlm_get_target_point(self, PROMPT: str, image_path:str):
        """
        PROMPT: 
        返回: 以座標點的形式返回目標物體
        """
        obj_xml, input_height, input_width = getobjectpoints(image=image_path,prompt=PROMPT, system_prompt=SYSTEM_PROMPT_POINT)
        image = Image.open(image_path)
        image.thumbnail([640,640], Image.Resampling.LANCZOS)# 改爲縮略圖 測試用，實際運行注意註釋掉
        result = plot_points(image,obj_xml,input_height, input_width) # 可視化box，導出像素座標列表 [(x1,y1,"lable"),(x2,y2,"lable")...]
        return result

    def get_image_response(self, prompt: list, image_path=None) -> dict:
        """
        
        """

        return inference(image_path,prompt)
    

if __name__ == '__main__':


    # env = UR5GraspEnv()
    # env.reset()
    # camera_check.visual_camer(env)
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
    agent = Qwen_agent(system_prompt=AGENT_SYS_PROMPT_OPENAI,client=client)
    image_path = "detection_visualization.jpg"

    prompt = "请抓取桌面上的蘋果，并将其放到盤子上。"
    prompt = "我肚子餓了你給我抓點東西吃"

    result = agent.get_response(prompt,image_path)
    print(result)
    functions = result['function']
    return_msg = result['response']
    if len(functions) != 0:
        for function in functions:
            print(function)
            eval(function)

    print(return_msg)
        





# # ============== 连续抓取多个物体 ====================

#     resp = client.chat.completions.create(
#         model="Qwen2.5-VL-7B-Instruct",
#         messages=[{"role": "user", "content": "你好，简单自我介绍一下。"}],
#         temperature=0.2,
#         max_tokens=256
#     )
#     print(resp.choices[0].message.content)

#     n = 4 # 循环次数，连续抓取物体
#     for _ in range(n): 

#         for i in range(500): # 1000
#             env.step()
        
#         # 1. 获取图像和深度图
#         imgs = env.render()
#         color_img_path = imgs['img'] # MuJoCo 渲染的是 RGB
#         depth_img_path = imgs['depth']

#         # 将MuJoCo渲染的是RGB转化为OpenCV默认使用BGR颜色空间
#         color_img_path = cv2.cvtColor(color_img_path, cv2.COLOR_RGB2BGR)
#         # 保存/查看图片
#         # cv2.imwrite('color_img_path.jpg', color_img_path)
#         # cv2.imshow('color', color_img_path)
#         # cv2.waitKey(0)
        
#         # 2. SAM分割图像
#         mask_img_path = segment_image(color_img_path)

#         # 3. 获取物体的点云数据
#         end_points, cloud_o3d = get_and_process_data(color_img_path, depth_img_path, mask_img_path)

#         # 4. 获取抓取点对应的夹爪姿态
#         gg = generate_grasps(end_points, cloud_o3d, True) # True or False

#         # 5. 仿真执行抓取
#         execute_grasp(env, gg)

#     env.close()
