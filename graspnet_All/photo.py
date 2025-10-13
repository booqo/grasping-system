import cv2
import numpy as np
import pyrealsense2 as rs
import os
import time
from datetime import datetime


def create_timestamp_folder(base_dir="captures"):
    """创建带时间戳的子文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    folder_path = os.path.join(base_dir, timestamp)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_graspnet_depth(depth_data, path):
    """保存GraspNet兼容的深度图（uint16 PNG，单位：毫米）"""
    # 转换为毫米并确保在uint16范围内
    depth_mm = np.clip(depth_data * 1000, 0, 65535).astype(np.uint16)
    cv2.imwrite(path, depth_mm)


def display_and_capture():
    # 初始化RealSense
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置相机流（与GraspNet训练尺寸一致）
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    try:
        pipeline.start(config)
        align = rs.align(rs.stream.color)  # 对齐深度到彩色帧
        print("按ENTER采集当前帧，Q键退出...")

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 转换图像数据
            color_image = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())  # 原始uint16数据

            # 获取实际物理单位数据（米）
            depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
            depth_meters = depth_data.astype(np.float32) * depth_scale

            # 显示处理
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_data, alpha=0.03),  # 仅用于显示
                cv2.COLORMAP_JET
            )

            # 显示画面
            cv2.imshow("RGB Preview", color_image)
            cv2.imshow("Depth Preview", depth_colormap)

            key = cv2.waitKey(1)

            if key == 13:  # ENTER键
                try:
                    save_folder = create_timestamp_folder()

                    # 生成文件名
                    timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
                    prefix = os.path.join(save_folder, f"capture_{timestamp}")

                    # 保存原始数据
                    cv2.imwrite(f"{prefix}_color.png", color_image)
                    np.save(f"{prefix}_depth_meters.npy", depth_meters)

                    # 保存GraspNet兼容格式
                    save_graspnet_depth(depth_meters, f"{prefix}_depth_graspnet.png")

                    print(f"采集保存到：{save_folder}")

                except Exception as e:
                    print(f"保存失败：{str(e)}")

            elif key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 创建根存储目录
    os.makedirs("captures", exist_ok=True)
    display_and_capture()