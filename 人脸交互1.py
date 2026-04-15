# -*- coding: utf-8 -*-
# 必须放在文件第一行
import cv2
import mediapipe as mp
import numpy as np

# 初始化MediaPipe面部网格模型
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 定义面部区域的颜色（BGR格式）
# 468个关键点按区域分组（参考MediaPipe官方索引定义）
COLORS = {
    'eyes': (0, 255, 255),       # 眼睛（青色）
    'eyebrows': (255, 0, 255),   # 眉毛（洋红色）
    'nose': (0, 255, 0),         # 鼻子（绿色）
    'mouth': (0, 0, 255),        # 嘴巴（红色）
    'face_contour': (255, 255, 0)# 面部轮廓（黄色）
}

# 面部各区域关键点索引范围（参考MediaPipe官方文档）
# 具体索引对应关系：https://mediapipe.dev/images/mobile/face_mesh_landmarks_index.png
FACE_REGIONS = {
    'eyes': list(range(33, 133)) + list(range(362, 463)),  # 左右眼及周围
    'eyebrows': list(range(17, 27)) + list(range(28, 33)) + list(range(336, 346)) + list(range(347, 352)),  # 左右眉毛
    'nose': list(range(1, 17)) + list(range(27, 36)) + list(range(352, 361)),  # 鼻子及周围
    'mouth': list(range(36, 41)) + list(range(42, 47)) + list(range(61, 68)) + list(range(76, 83)) + list(range(84, 91)) + list(range(91, 96)) + list(range(181, 188)) + list(range(267, 274)) + list(range(287, 294)) + list(range(308, 315)),  # 嘴巴及周围
    'face_contour': list(range(10, 17)) + list(range(67, 76)) + list(range(146, 153)) + list(range(177, 181)) + list(range(234, 241)) + list(range(288, 295)) + list(range(356, 362))  # 面部轮廓
}

def draw_face_landmarks(frame, landmarks):
    """绘制彩色高亮的面部关键点"""
    h, w, _ = frame.shape  # 图像高度和宽度
    for region, indices in FACE_REGIONS.items():
        color = COLORS[region]
        for idx in indices:
            if idx >= len(landmarks):
                continue  # 防止索引越界
            # 转换相对坐标为像素坐标
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            # 绘制高亮关键点（圆形，半径3，填充）
            cv2.circle(frame, (x, y), 3, color, -1)
            # 可选：连接相邻关键点形成区域轮廓（增强高亮效果）
            if idx < len(indices) - 1:
                next_idx = indices[idx + 1]
                if next_idx < len(landmarks):
                    x2 = int(landmarks[next_idx].x * w)
                    y2 = int(landmarks[next_idx].y * h)
                    cv2.line(frame, (x, y), (x2, y2), color, 1)
    return frame

def main():
    # 打开摄像头（0为默认摄像头）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 配置MediaPipe面部网格检测
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,  # 实时视频模式
        max_num_faces=1,          # 最多检测1张脸
        refine_landmarks=True,    # 细化关键点（提高眼睛和嘴唇区域精度）
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as facial:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("无法获取视频帧")
                break

            # 转换为RGB（MediaPipe需要RGB输入）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 处理帧并检测面部关键点
            results = facial.process(frame_rgb)

            # 绘制彩色高亮关键点
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    frame = draw_face_landmarks(frame, face_landmarks.landmark)

            # 左右翻转画面（镜像效果，符合视觉习惯）
            frame = cv2.flip(frame, 1)
            # 显示结果
            cv2.imshow('面部表情彩色高亮'.encode('utf-8').decode('gbk', errors='ignore'), frame)

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()