import cv2
import mediapipe as mp

# 初始化MediaPipe解决方案
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils  # 添加绘图工具
# 在初始化部分
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# 读取视频
cap = cv2.VideoCapture(0)  # 使用0作为摄像头索引

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像从BGR转换为RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理帧
    results = face_mesh.process(frame_rgb)

    # 绘制结果（面部网格）
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 使用mp_drawing绘制，并更新连接参数
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,  # 使用正确的连接属性
                landmark_drawing_spec = drawing_spec,
                connection_drawing_spec = drawing_spec
            )
    # 显示结果
    cv2.imshow('MediaPipe Face Mesh', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # 按Esc退出
        break

cap.release()
cv2.destroyAllWindows()