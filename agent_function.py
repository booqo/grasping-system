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

from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from utils.qwen_utils import parse_agent_json

from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv

from cv_process import segment_image,segment_image_with_bbox
import time
from copy import deepcopy
from openai import OpenAI
from main_Qwen_grasp import Qwen_agent,AGENT_SYS_PROMPT_OPENAI



class Qwen_Robot():
    def __init__(self,env:UR5GraspEnv,client):
        self.q0 = env.robot.get_joint()
        self.env = env
        self.qwen_agent = Qwen_agent(system_prompt=AGENT_SYS_PROMPT_OPENAI,client=client)

    def get_image(self):
        # 1. 获取图像和深度图
        imgs = self.env.render()
        color_img = imgs['img'] # MuJoCo 渲染的是 RGB
        depth_img = imgs['depth']

        # 将MuJoCo渲染的是RGB转化为OpenCV默认使用BGR颜色空间
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        # 保存/查看图片
        os.makedirs("./temp", exist_ok=True)
        cv2.imwrite('./temp/temp.jpg', color_img)

        # cv2.imshow('color', color_img)
        # cv2.waitKey(0)
        return color_img,depth_img

    def back_zero(self):
        robot = self.env.robot

        action = np.zeros(7)

        # 目标：将机器人从当前位置移动到0位（q1）
        time1 = 1
        q0 = robot.get_joint()
        q1 = self.q0 # 6个关节角度
        parameter0 = JointParameter(q0, q1)
        velocity_parameter0 = QuinticVelocityParameter(time1)
        trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
        planner1 = TrajectoryPlanner(trajectory_parameter0)
        # 执行planner_array = [planner1]
        time_array = [0.0, time1]
        planner_array = [planner1]
        total_time = np.sum(time_array)
        time_step_num = round(total_time / 0.002) + 1
        times = np.linspace(0.0, total_time, time_step_num)
        time_cumsum = np.cumsum(time_array)
        for timei in times:
            for j in range(len(time_cumsum)):
                if timei == 0.0:
                    break
                if timei <= time_cumsum[j]:
                    planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                    if isinstance(planner_interpolate, np.ndarray):
                        joint = planner_interpolate
                        robot.move_joint(joint)
                    else:
                        robot.move_cartesian(planner_interpolate)
                        joint = robot.get_joint()
                    action[:6] = joint
                    self.env.step(action)
                    break

    def head_shake(self,time1 = 1):
        self.back_zero(self.env) # 先回零
        
        robot = self.env.robot

        action = np.zeros(7)

        q0 = robot.get_joint()
        q1 = np.array([0.0, 0.0, 0.0, 0.0, -np.pi / 2, 0.0]) # 6个关节角度
        def shake(q_begin,q_end):
            parameter0 = JointParameter(q_begin, q_end)
            velocity_parameter0 = QuinticVelocityParameter(time1)
            trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
            planner1 = TrajectoryPlanner(trajectory_parameter0)
            # 执行planner_array = [planner1]
            time_array = [0.0, time1]
            planner_array = [planner1]
            total_time = np.sum(time_array)
            time_step_num = round(total_time / 0.002) + 1
            times = np.linspace(0.0, total_time, time_step_num)
            time_cumsum = np.cumsum(time_array)
            for timei in times:
                for j in range(len(time_cumsum)):
                    if timei == 0.0:
                        break
                    if timei <= time_cumsum[j]:
                        planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                        if isinstance(planner_interpolate, np.ndarray):
                            joint = planner_interpolate
                            robot.move_joint(joint)
                        else:
                            robot.move_cartesian(planner_interpolate)
                            joint = robot.get_joint()
                        action[:6] = joint
                        env.step(action)
                        break
        for _ in range(2):
            shake(q0,q1)
            shake(q1,q0)

    def head_nod(self,time1 = 1):
        self.back_zero(self.env) # 先回零
        
        robot = self.env.robot

        action = np.zeros(7)

        q0 = robot.get_joint()
        q1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -np.pi / 2]) # 6个关节角度
        def shake(q_begin,q_end):
            parameter0 = JointParameter(q_begin, q_end)
            velocity_parameter0 = QuinticVelocityParameter(time1)
            trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
            planner1 = TrajectoryPlanner(trajectory_parameter0)
            # 执行planner_array = [planner1]
            time_array = [0.0, time1]
            planner_array = [planner1]
            total_time = np.sum(time_array)
            time_step_num = round(total_time / 0.002) + 1
            times = np.linspace(0.0, total_time, time_step_num)
            time_cumsum = np.cumsum(time_array)
            for timei in times:
                for j in range(len(time_cumsum)):
                    if timei == 0.0:
                        break
                    if timei <= time_cumsum[j]:
                        planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                        if isinstance(planner_interpolate, np.ndarray):
                            joint = planner_interpolate
                            robot.move_joint(joint)
                        else:
                            robot.move_cartesian(planner_interpolate)
                            joint = robot.get_joint()
                        action[:6] = joint
                        env.step(action)
                        break
        for _ in range(2):
            shake(q0,q1)
            shake(q1,q0)

    def claw_open(self):
        action = np.zeros(7)
        for i in range(1000):
            action[-1] -= 0.2
            action[-1] = np.min([action[-1], 255])
            self.env.step(action)

    def claw_close(self):
        action = np.zeros(7)
        for i in range(1000):
            action[-1] += 0.2
            action[-1] = np.min([action[-1], 255])
            self.env.step(action)

    def move_to_coords(self,X:int, Y:int, Z:int):
        q0 = robot.get_joint()# 機器人當前關節位姿
        action = np.zeros(7)
        robot = self.env.robot
        T_wb = robot.base
        # 2.接近抓取位姿
        # 目标：从预抓取位姿直线移动到抓取点附近（T2）
        # 关键点：T2 是 T_wo 沿负 x 方向偏移 0.1m，确保安全接近物体。
        time1 = 1
        robot.set_joint(q0)
        T1 = robot.get_cartesian()
        T2 = sm.SE3(X, Y, Z)
        position_parameter1 = LinePositionParameter(T1.t, T2.t) #  位置规划（直线路径）
        attitude_parameter1 = OneAttitudeParameter(sm.SO3(T1.R), sm.SO3(T2.R)) # 姿态规划（插值旋转）
        cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1) # 组合笛卡尔参数
        velocity_parameter1 = QuinticVelocityParameter(time1) # 速度曲线（五次多项式插值）
        trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1) # 将笛卡尔空间路径和速度曲线结合，生成完整的轨迹参数
        planner2 = TrajectoryPlanner(trajectory_parameter1) # 轨迹规划器，将笛卡尔空间路径和速度曲线结合，生成完整的轨迹参数
        # 执行planner_array = [planner2]
        time_array = [0.0, time1]
        planner_array = [planner2]
        total_time = np.sum(time_array)
        time_step_num = round(total_time / 0.002) + 1
        times = np.linspace(0.0, total_time, time_step_num)
        time_cumsum = np.cumsum(time_array)
        for timei in times:
            for j in range(len(time_cumsum)):
                if timei == 0.0:
                    break
                if timei <= time_cumsum[j]:
                    planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                    if isinstance(planner_interpolate, np.ndarray):
                        joint = planner_interpolate
                        robot.move_joint(joint)
                    else:
                        robot.move_cartesian(planner_interpolate)
                        joint = robot.get_joint()
                    action[:6] = joint
                    self.env.step(action)
                    break

    def single_joint_move(self,joint_id:int, degree:float):# 不回0
        robot = self.env.robot
        action = np.zeros(7)
        time1 = 1
        q0 = robot.get_joint()
        if joint_id>5 and joint_id<0:
            print("wrong with joint number, the number of  joint should in arange from 0 to 5")
            return 
        p_degree = degree/180*np.pi
        q1 = q0[:]
        q1[joint_id] = p_degree
        parameter0 = JointParameter(q0, q1)
        velocity_parameter0 = QuinticVelocityParameter(time1)
        trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
        planner1 = TrajectoryPlanner(trajectory_parameter0)
        # 执行planner_array = [planner1]
        time_array = [0.0, time1]
        planner_array = [planner1]
        total_time = np.sum(time_array)
        time_step_num = round(total_time / 0.002) + 1
        times = np.linspace(0.0, total_time, time_step_num)
        time_cumsum = np.cumsum(time_array)
        for timei in times:
            for j in range(len(time_cumsum)):
                if timei == 0.0:
                    break
                if timei <= time_cumsum[j]:
                    planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                    if isinstance(planner_interpolate, np.ndarray):
                        joint = planner_interpolate
                        robot.move_joint(joint)
                    else:
                        robot.move_cartesian(planner_interpolate)
                        joint = robot.get_joint()
                    action[:6] = joint
                    self.env.step(action)
                    break

    def move_to_prepare(self):

        robot = self.env.robot

        action = np.zeros(7)

        # 1.机器人运动到预抓取位姿
        # 目标：将机器人从当前位置移动到预抓取姿态（q1）
        time1 = 1
        q0 = robot.get_joint()
        q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0]) # 6个关节角度
        parameter0 = JointParameter(q0, q1)
        velocity_parameter0 = QuinticVelocityParameter(time1)
        trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
        planner1 = TrajectoryPlanner(trajectory_parameter0)
        # 执行planner_array = [planner1]
        time_array = [0.0, time1]
        planner_array = [planner1]
        total_time = np.sum(time_array)
        time_step_num = round(total_time / 0.002) + 1
        times = np.linspace(0.0, total_time, time_step_num)
        time_cumsum = np.cumsum(time_array)
        for timei in times:
            for j in range(len(time_cumsum)):
                if timei == 0.0:
                    break
                if timei <= time_cumsum[j]:
                    planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                    if isinstance(planner_interpolate, np.ndarray):
                        joint = planner_interpolate
                        robot.move_joint(joint)
                    else:
                        robot.move_cartesian(planner_interpolate)
                        joint = robot.get_joint()
                    action[:6] = joint
                    self.env.step(action)
                    break

    def vlm_move(self,PROMPT='帮我把绿色方块放在小猪佩奇上', input_way='keyboard'):
        '''
        多模态抓取/搬运（自然语言描述）
        input_way：speech语音输入，keyboard键盘输入
        '''

        print('多模态大模型识别图像')
        
        # 机械臂归零
        print('机械臂归零')
        self.back_zero()
        # time.sleep(3)
        
        print('第二步，给出的指令是：', PROMPT)
        
        ## 第三步：獲得視圖
        print('第三步：獲得圖像')
        color_img,depth_img = self.get_image() # RGB image
        
        ## 第四步：将图片输入给多模态视觉大模型
        print('第四步：将图片输入给多模态视觉大模型')
        img_path = './temp/temp.jpg'
        
        n = 1
        while n < 5:
            try:
                print('    尝试第 {} 次访问多模态大模型'.format(n))
                # result = yi_vision_api(PROMPT, img_path='temp/vl_now.jpg')  # yi_vision定位能力出现波动，暂时换用QwenVL系列
                result = self.qwen_agent.vlm_get_bbox_postion(PROMPT=PROMPT,image_path=img_path)
                print('    多模态大模型调用成功！')
                print(result)
                break
            except Exception as e:
                print('    多模态大模型返回数据结构错误，再尝试一次', e)
                n += 1
        
    
        print('将像素坐标转换为机械臂坐标')
        """
        具體流程入下，獲得位姿的字典，{"lable1":[abs_x1,abs_y1,abs_x2,abs_y2],"lable2":[abs_x1,abs_y1,abs_x2,abs_y2]}
        lable1是起點的標籤，lable2是終點的標籤，先使用seg模型對圖片這倆部分進行分割，可能有物體，也可能沒有物體，有物體使用分割後的mask，沒有的話使用bbox
        最後得到倆部分相機系下的點雲，然後有倆個方案，1：抓取，需要graspnet；2：示意移動，直接獲得點雲的座標，讓機械臂直接移動到位置之後不進行操作就行
        """
        #  result 是 {"start": [x1, y1, x2, y2], "end": [x1, y1, x2, y2]}
        if not result or not isinstance(result, dict):
            print("未能获得有效的目标位置，任务终止。")
            return
        if len(result)>1:# 說明是有兩個物體，需要抓，拿放下
            k = 0
            for key, val in result.items():
                # 第一次先抓，第二次再放
                lable, bbox = key, val
                k+=1
                k = k%2


        else: # 一個目標抓取就行，先sam分割，

            for lable,bbox in result.items():
                print(lable)
                mask_image = segment_image_with_bbox(color_img,lable,bbox)
                
                end_points, cloud_o3d = get_and_process_data(color_img, depth_img, mask_image)

                gg = generate_grasps(end_points, cloud_o3d, True)

                execute_grasp(env, gg)

            ...
        


        # # 起点，机械臂坐标
        # START_X_MC, START_Y_MC = eye2hand(START_X_CENTER, START_Y_CENTER)
        # # 终点，机械臂坐标
        # END_X_MC, END_Y_MC = eye2hand(END_X_CENTER, END_Y_CENTER)
        
        # ## 第七步：吸泵吸取移动物体
        # print('第七步：吸泵吸取移动物体')
        # pump_move(mc=mc, XY_START=[START_X_MC, START_Y_MC], XY_END=[END_X_MC, END_Y_MC])
        
        # ## 第八步：收尾
        # print('第八步：任务完成')
        # GPIO.cleanup()            # 释放GPIO pin channel
        # cv2.destroyAllWindows()   # 关闭所有opencv窗口
        # # exit()

    def vlm_vqa(self,PROMPT='请数一数图中中几个方块', input_way='keyboard'):
        # 机械臂归零
        print('机械臂归零')
        # self.back_zero()
        # time.sleep(3)
        print('第二步，给出的指令是：', PROMPT)
        color_img,depth_img = self.get_image() # RGB image
        img_path = 'temp/temp.jpg'
        result = self.qwen_agent.get_image_response(PROMPT,img_path)
        print('    多模态大模型调用成功！')
        print(result)
        cv2.destroyAllWindows()   # 关闭所有opencv窗口

    def top_view_shot(self,check=False):
        '''
        拍摄一张图片并保存
        check：是否需要人工看屏幕确认拍照成功，再在键盘上按q键确认继续
        '''
        # 获取摄像头，传入0表示获取系统默认摄像头
        cap = cv2.VideoCapture(0)
        # 打开cap
        cap.open(0)
        time.sleep(0.3)
        success, img_bgr = cap.read()
        
        # 保存图像
        print('    保存至temp/vl_now.jpg')
        cv2.imwrite('temp/vl_now.jpg', img_bgr)

        # 屏幕上展示图像
        cv2.destroyAllWindows()   # 关闭所有opencv窗口
        cv2.imshow('zihao_vlm', img_bgr) 
        
        if check:
            print('请确认拍照成功，按c键继续，按q键退出')
            while(True):
                key = cv2.waitKey(10) & 0xFF
                if key == ord('c'): # 按c键继续
                    break
                if key == ord('q'): # 按q键退出
                    # exit()
                    cv2.destroyAllWindows()   # 关闭所有opencv窗口
                    raise NameError('按q退出')
        else:
            if cv2.waitKey(10) & 0xFF == None:
                pass
            
        # 关闭摄像头
        cap.release()
        # 关闭图像窗口
        # cv2.destroyAllWindows()





# ================= 数据处理并生成输入 ====================
def get_and_process_data(color_path, depth_path, mask_path):
    """
    根据给定的 RGB 图、深度图、掩码图（可以是 文件路径 或 NumPy 数组），生成输入点云及其它必要数据
    """
#---------------------------------------
    # 1. 加载 color（可能是路径，也可能是数组）
    if isinstance(color_path, str):
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    elif isinstance(color_path, np.ndarray):
        color = color_path.astype(np.float32)
        color /= 255.0
    else:
        raise TypeError("color_path 既不是字符串路径也不是 NumPy 数组！")

    # 2. 加载 depth（可能是路径，也可能是数组）
    if isinstance(depth_path, str):
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img)
    elif isinstance(depth_path, np.ndarray):
        depth = depth_path
    else:
        raise TypeError("depth_path 既不是字符串路径也不是 NumPy 数组！")

    # 3. 加载 mask（可能是路径，也可能是数组）
    if isinstance(mask_path, str):
        workspace_mask = np.array(Image.open(mask_path))
    elif isinstance(mask_path, np.ndarray):
        workspace_mask = mask_path
    else:
        raise TypeError("mask_path 既不是字符串路径也不是 NumPy 数组！")

    # print("\n=== 尺寸验证 ===")
    # print("深度图尺寸:", depth.shape)
    # print("颜色图尺寸:", color.shape[:2])
    # print("工作空间尺寸:", workspace_mask.shape)

    # 构造相机内参矩阵
    height = color.shape[0]
    width = color.shape[1]
    fovy = np.pi / 4 # 定义的仿真相机
    focal = height / (2.0 * np.tan(fovy / 2.0))  # 焦距计算（基于垂直视场角fovy和高度height）
    c_x = width / 2.0   # 水平中心
    c_y = height / 2.0  # 垂直中心
    intrinsic = np.array([
        [focal, 0.0, c_x],    
        [0.0, focal, c_y],   
        [0.0, 0.0, 1.0]
    ])
    factor_depth = 1.0  # 深度因子，根据实际数据调整

    # 利用深度图生成点云 (H,W,3) 并保留组织结构
    camera = CameraInfo(width, height, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # mask = depth < 2.0
    mask = (workspace_mask > 0) & (depth < 2.0)
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    # print(f"mask过滤后的点云数量 (color_masked): {len(color_masked)}") # 在采样前打印原始过滤后的点数

    NUM_POINT = 3000 # 10000或5000
    # 如果点数足够，随机采样NUM_POINT个点（不重复）
    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    # 如果点数不足，先保留所有点，再随机重复补足NUM_POINT个点
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs] # 提取点云和颜色

    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    # end_points = {'point_clouds': cloud_sampled}

    end_points = dict()
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud_o3d

# =================== 获取抓取预测 ====================
def generate_grasps(end_points, cloud, visual=False):
    """
    主推理流程：
    0. 数据处理并生成输入
    1. 加载网络
    2. 前向推理（进行抓取预测解码）
    3. 碰撞检测
    4. NMS 去重 + 按置信度/得分排序（降序）
    5. 对抓取预测进行垂直角度筛选
    """

    # 1. 加载网络
    net = GraspNet(input_feature_dim=0, 
                   num_view=300, 
                   num_angle=12, 
                   num_depth=4,
                   cylinder_radius=0.05, 
                   hmin=-0.02, 
                   hmax_list=[0.01, 0.02, 0.03, 0.04], 
                   is_training=False)
    net.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    checkpoint = torch.load('./logs/log_rs/checkpoint-rs.tar') # checkpoint_path
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # 2. 前向推理
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy()) 

    # 3. 碰撞检测
    COLLISION_THRESH = 0.01
    if COLLISION_THRESH > 0:
        voxel_size = 0.01
        collision_thresh = 0.01
        mfcdetector = ModelFreeCollisionDetector(np.asarray(cloud.points), voxel_size=voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        gg = gg[~collision_mask]

    # 4. NMS 去重 + 按置信度/得分排序（降序）
    gg.nms().sort_by_score()

    # 5. 返回抓取得分最高的抓取（对抓取预测的接近方向进行垂直角度限制）
    # 将 gg 转换为普通列表
    all_grasps = list(gg)
    vertical = np.array([0, 0, 1])  # 期望抓取接近方向（垂直桌面） np.array([0, 0, 1])
    angle_threshold = np.deg2rad(30)  # 30度的弧度值 np.deg2rad(30)
    filtered = []
    for grasp in all_grasps:
        # 抓取的接近方向取 grasp.rotation_matrix 的第三列[:, 0]
        approach_dir = grasp.rotation_matrix[:, 0]
        # 计算夹角：cos(angle)=dot(approach_dir, vertical)
        cos_angle = np.dot(approach_dir, vertical)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            filtered.append(grasp)
    if len(filtered) == 0:
        print("\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.")
        filtered = all_grasps
    # else:
        print(f"\nFiltered {len(filtered)} grasps within ±30° of vertical out of {len(all_grasps)} total predictions.")

    # 对过滤后的抓取根据 score 排序（降序）
    filtered.sort(key=lambda g: g.score, reverse=True)

    # 取前20个抓取（如果少于20个，则全部使用）
    top_grasps = filtered[:20]
    # top_grasps = filtered[:1]

    # 可视化过滤后的抓取，手动转换为 Open3D 物体
    grippers = [g.to_open3d_geometry() for g in top_grasps]
    # print(f"\nVisualizing top {len(top_grasps)} grasps after vertical filtering...")
    # o3d.visualization.draw_geometries([cloud, *grippers])
    # for gripper in grippers:
    #     o3d.visualization.draw_geometries([cloud, gripper])
    
    # 选择得分最高的抓取（filtered 列表已按得分降序排序）
    best_grasp = top_grasps[0]
    best_translation = best_grasp.translation
    best_rotation = best_grasp.rotation_matrix
    best_width = best_grasp.width

    # 创建一个新的 GraspGroup 并添加最佳抓取
    new_gg = GraspGroup()            # 初始化空的 GraspGroup
    new_gg.add(best_grasp)           # 添加最佳抓取
    if visual:
        grippers = new_gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
    return new_gg
    # return best_translation, best_rotation, best_width


# ================= 仿真执行抓取动作 ====================
def execute_grasp(env, gg):
    """
    执行抓取动作，控制机器人从初始位置移动到抓取位置，并完成抓取操作。

    参数:
    env (UR5GraspEnv): 机器人环境对象。
    gg (GraspGroup): 抓取预测结果。
    """
    robot = env.robot
    T_wb = robot.base

    # 0.初始准备阶段
    # 目标：计算抓取位姿 T_wo（物体相对于世界坐标系的位姿）
    # n_wc = np.array([0.0, -1.0, 0.0]) # 相机朝向
    # o_wc = np.array([-1.0, 0.0, -0.5]) # 相机朝向 [0.5, 0.0, -1.0] -> [-1.0, 0.0, -0.5]
    # t_wc = np.array([1.0, 0.6, 2.0]) # 相机的位置。2.0是相机高度，与scene.xml中保持一致。
    n_wc = np.array([0.0, -1.0, 0.0]) 
    o_wc = np.array([-1.0, 0.0, -0.5]) 
    # t_wc = np.array([0.85, 0.8, 1.6]) 
    t_wc = np.array([1.054, 0.539, 1.776]) 

    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))
    T_wo = T_wc * T_co

    action = np.zeros(7)

    # 1.机器人运动到预抓取位姿
    # 目标：将机器人从当前位置移动到预抓取姿态（q1）
    time1 = 1
    q0 = robot.get_joint()
    q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0]) # 6个关节角度
    parameter0 = JointParameter(q0, q1)
    velocity_parameter0 = QuinticVelocityParameter(time1)
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
    planner1 = TrajectoryPlanner(trajectory_parameter0)
    # 执行planner_array = [planner1]
    time_array = [0.0, time1]
    planner_array = [planner1]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

    # 2.接近抓取位姿
    # 目标：从预抓取位姿直线移动到抓取点附近（T2）
    # 关键点：T2 是 T_wo 沿负 x 方向偏移 0.1m，确保安全接近物体。
    time2 = 1
    robot.set_joint(q1)
    T1 = robot.get_cartesian()
    T2 = T_wo * sm.SE3(-0.1, 0.0, 0.0)
    position_parameter1 = LinePositionParameter(T1.t, T2.t) #  位置规划（直线路径）
    attitude_parameter1 = OneAttitudeParameter(sm.SO3(T1.R), sm.SO3(T2.R)) # 姿态规划（插值旋转）
    cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1) # 组合笛卡尔参数
    velocity_parameter1 = QuinticVelocityParameter(time2) # 速度曲线（五次多项式插值）
    trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1) # 将笛卡尔空间路径和速度曲线结合，生成完整的轨迹参数
    planner2 = TrajectoryPlanner(trajectory_parameter1) # 轨迹规划器，将笛卡尔空间路径和速度曲线结合，生成完整的轨迹参数
    # 执行planner_array = [planner2]
    time_array = [0.0, time2]
    planner_array = [planner2]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

    # 3.执行抓取
    # 目标：从 T2 移动到 T3（精确抓取位姿）。通过逐步增加 action[-1]（夹爪控制信号）闭合夹爪，抓取物体。
    time3 = 1
    T3 = T_wo
    position_parameter2 = LinePositionParameter(T2.t, T3.t)
    attitude_parameter2 = OneAttitudeParameter(sm.SO3(T2.R), sm.SO3(T3.R))
    cartesian_parameter2 = CartesianParameter(position_parameter2, attitude_parameter2)
    velocity_parameter2 = QuinticVelocityParameter(time3)
    trajectory_parameter2 = TrajectoryParameter(cartesian_parameter2, velocity_parameter2)
    planner3 = TrajectoryPlanner(trajectory_parameter2)
    # 执行planner_array = [planner3]
    time_array = [0.0, time3]
    planner_array = [planner3]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num) 
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)): 
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break
    for i in range(1000):
        action[-1] += 0.2
        action[-1] = np.min([action[-1], 255])
        env.step(action)

    # 4.提起物体
    # 目标：抓取后垂直提升物体（避免碰撞桌面）。
    time4 = 1
    T4 = sm.SE3.Trans(0.0, 0.0, 0.3) * T3 # 通过在T3的基础上向上偏移0.3单位得到的，用于控制机器人上升一定的高度
    position_parameter3 = LinePositionParameter(T3.t, T4.t)
    attitude_parameter3 = OneAttitudeParameter(sm.SO3(T3.R), sm.SO3(T4.R))
    cartesian_parameter3 = CartesianParameter(position_parameter3, attitude_parameter3)
    velocity_parameter3 = QuinticVelocityParameter(time4)
    trajectory_parameter3 = TrajectoryParameter(cartesian_parameter3, velocity_parameter3)
    planner4 = TrajectoryPlanner(trajectory_parameter3)

    # 5.水平移动物体
    # 目标：将物体水平移动到目标放置位置，保持高度不变。
    time5 = 1
    T5 = sm.SE3.Trans(1.4, 0.3, T4.t[2]) * sm.SE3(sm.SO3(T4.R)) #  通过在T4的基础上进行平移得到，这里的1.4, 0.3是场景中的固定点坐标，而不是偏移量
    position_parameter4 = LinePositionParameter(T4.t, T5.t)
    attitude_parameter4 = OneAttitudeParameter(sm.SO3(T4.R), sm.SO3(T5.R))
    cartesian_parameter4 = CartesianParameter(position_parameter4, attitude_parameter4)
    velocity_parameter4 = QuinticVelocityParameter(time5)
    trajectory_parameter4 = TrajectoryParameter(cartesian_parameter4, velocity_parameter4)
    planner5 = TrajectoryPlanner(trajectory_parameter4)

    # 6.放置物体
    # 目标：垂直下降物体到接触面（T7）。逐步减小 action[-1]（夹爪信号）以释放物体。
    time6 = 1
    T6 = sm.SE3.Trans(0.0, 0.0, -0.1) * T5 # 通过在T5的基础上向下偏移0.1单位得到的，用于控制机器人下降一定的高度
    position_parameter6 = LinePositionParameter(T5.t, T6.t)
    attitude_parameter6 = OneAttitudeParameter(sm.SO3(T5.R), sm.SO3(T6.R))
    cartesian_parameter6 = CartesianParameter(position_parameter6, attitude_parameter6)
    velocity_parameter6 = QuinticVelocityParameter(time6)
    trajectory_parameter6 = TrajectoryParameter(cartesian_parameter6, velocity_parameter6)
    planner6 = TrajectoryPlanner(trajectory_parameter6)

    # 执行planner_array = [planner4, planner5, planner6]
    time_array = [0.0, time4, time5, time6]
    planner_array = [planner4, planner5, planner6]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break
    for i in range(1000):
        action[-1] -= 0.2
        action[-1] = np.max([action[-1], 0])
        env.step(action)

    # 7.抬起夹爪
    # 目标：放置后抬起夹爪，避免碰撞物体。
    time7 = 1
    T7 = sm.SE3.Trans(0.0, 0.0, 0.1) * T6
    position_parameter7 = LinePositionParameter(T6.t, T7.t)
    attitude_parameter7 = OneAttitudeParameter(sm.SO3(T6.R), sm.SO3(T7.R))
    cartesian_parameter7 = CartesianParameter(position_parameter7, attitude_parameter7)
    velocity_parameter7 = QuinticVelocityParameter(time7)
    trajectory_parameter7 = TrajectoryParameter(cartesian_parameter7, velocity_parameter7)
    planner7 = TrajectoryPlanner(trajectory_parameter7)
    # 执行planner_array = [planner7]
    time_array = [0.0, time7]
    planner_array = [planner7]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

    # 8.回到初始位置
    # 目标：机器人返回初始姿态（q0），完成整个任务。
    time8 = 1
    q8 = robot.get_joint()
    q9 = q0
    parameter8 = JointParameter(q8, q9)
    velocity_parameter8 = QuinticVelocityParameter(time8)
    trajectory_parameter8 = TrajectoryParameter(parameter8, velocity_parameter8)
    planner8 = TrajectoryPlanner(trajectory_parameter8)
    # 执行planner_array = [planner8]
    time_array = [0.0, time8]
    planner_array = [planner8]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break


if __name__ == '__main__':

    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
    env = UR5GraspEnv()
    env.reset()

    robot= Qwen_Robot(env,client)

    # color_img,_ = robot.get_image()
    # # image_path = "detection_visualization.jpg"

    # prompt = "请抓取桌面上的蘋果，并将其放到盤子上。"
    # prompt = "抓个香蕉"
    # # prompt = "please use English to answer my question, catch the apple"
    # # prompt = "畫面中有蘋果嗎？" # 對於沒有的物品還是有bug

    # result = robot.qwen_agent.get_response(prompt,color_img)
    # print(result)
    # functions = result['function']
    # return_msg = result['response']
    # if len(functions) != 0:
    #     for function in functions:
    #         print(function)
    #         eval("robot."+ function)

    # print(return_msg)

    while True:
        for i in range(500): # 1000
            env.step()
        # color_img,_ = robot.get_image()
        color_img = None
        client_message = input("输入指令（exit/quit 退出）：").strip()
        if client_message.lower() in {"exit", "quit"}:
            print("已退出。")
            break
        # prompt = "你好,介紹一下自己"
        # print(prompt)
        prompt = client_message
        result = robot.qwen_agent.get_response(prompt,color_img)
        # print(result)
        functions = result['function']
        return_msg = result['response']
        print(return_msg)
        if len(functions) != 0:
            for function in functions:
                print(function)
                eval("robot."+ function)

        



        