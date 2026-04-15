import cv2
import mediapipe as mp
import numpy as np

# 初始化MediaPipe手部检测模型
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # 用于绘制关键点（可选）

# 定义21个关键点的彩色列表（可自定义颜色）
colors = [
    (255, 0, 0),    # 0: 腕部（蓝色）
    (0, 255, 0),    # 1-4: 拇指（绿色）
    (0, 0, 255),    # 5-8: 食指（红色）
    (255, 255, 0),  # 9-12: 中指（青色）
    (255, 0, 255),  # 13-16: 无名指（洋红色）
    (0, 255, 255)   # 17-20: 小指（黄色）
]

def draw_landmarks(frame, landmarks):
    """在帧上绘制彩色关键点"""
    h, w, _ = frame.shape  # 获取帧的高度和宽度
    for idx, landmark in enumerate(landmarks):
        # 将关键点坐标从相对值转换为像素坐标
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        # 根据关键点索引选择颜色（0:腕部，1-4:拇指，5-8:食指等）
        if idx == 0:
            color = colors[0]
        elif 1 <= idx <=4:
            color = colors[1]
        elif 5 <= idx <=8:
            color = colors[2]
        elif 9 <= idx <=12:
            color = colors[3]
        elif 13 <= idx <=16:
            color = colors[4]
        else:  # 17-20
            color = colors[5]
        # 绘制圆形关键点（半径5，线宽2）
        cv2.circle(frame, (x, y), 5, color, -1)  # -1表示填充圆
    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 1. 实例化Hands模型（配置参数）
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:  # 用with语句创建实例，自动管理资源

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 2. 用实例hands调用process，显式传入image参数
            results = hands.process(image=frame_rgb)  # 关键修正：用实例+传image

            # 后续处理...
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    frame = draw_landmarks(frame, hand_landmarks.landmark)

            # 显示画面...
            cv2.imshow('手部关键点', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()