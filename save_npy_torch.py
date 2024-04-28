import cv2
import torch
import torchvision.transforms as transforms

def read_videos(video_paths, video_length, frame_width, frame_height):
    videos_data = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((frame_height, frame_width)),
        transforms.ToTensor()
    ])
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while(len(frames) < video_length):
            ret, frame = cap.read()
            if not ret:
                break
            # 调整帧大小
            frame = cv2.resize(frame, (frame_width, frame_height))
            # OpenCV读取的图像通道顺序为BGR，需要转换为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 将帧转换为PyTorch张量
            frame = transform(frame)
            frames.append(frame)
        print(len(frames))
        cap.release()
        # 将视频填充到指定长度，如果视频不够长，重复最后一帧
        while len(frames) < video_length:
            frames.append(frames[-1])
        # 转换为张量并添加到列表
        frames = torch.stack(frames, dim=0)
        videos_data.append(frames)
    # 堆叠多个视频张量
    videos_data = torch.stack(videos_data, dim=0)
    return videos_data

# 读取多个视频并转换为张量
# video_paths = ['video1.mp4', 'video2.mp4']  # 视频文件路径列表
video_paths = ['E:\\data\\cmp\\fvd\\gt\\video_4s_0202.mp4']  # 视频文件路径列表
video_length = 30  # 每个视频的帧数
frame_width = 128  # 每帧的宽度
frame_height = 128  # 每帧的高度
# a = load_video()
videos_tensor = read_videos(video_paths, video_length, frame_width, frame_height)
print("视频张量的形状:", videos_tensor.shape)
videos_np = videos_tensor.numpy()
import numpy as np
np.save("E:\\data\\cmp\\fvd\\npy\\1.npy", videos_np)
