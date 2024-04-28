import cv2
import tensorflow as tf

def read_video(video_path, resize=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if resize is not None:
            frame = cv2.resize(frame, resize)
        # OpenCV读取的图像通道顺序为BGR，需要转换为RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    # 将图像帧转换为TensorFlow张量
    frames = tf.convert_to_tensor(frames, dtype=tf.uint8)
    return frames

# 读取视频并将其转换为TensorFlow张量
video_path = 'E:\\data\\cmp\\fvd\\gt\\video_4s_0202.mp4'
video_frames = read_video(video_path, resize=(224, 224))  # 可选的调整大小参数
print("视频帧张量的形状:", video_frames.shape)
