import cv2
import tensorflow as tf
import numpy as np
def read_video(video_paths, video_length, resize=None):
    videos_data = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while(len(frames) < video_length):
            ret, frame = cap.read()
            if not ret:
                break
            if resize is not None:
                frame = cv2.resize(frame, resize)
            # OpenCV读取的图像通道顺序为BGR，需要转换为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        print(len(frames))
        cap.release()
        # 将视频填充到指定长度，如果视频不够长，重复最后一帧
        while len(frames) < video_length:
            frames.append(frames[-1])
        # 将图像帧转换为TensorFlow张量
        frames = tf.convert_to_tensor(np.asarray(frames), dtype=tf.float32)
        videos_data.append(frames)
    # 堆叠多个视频张量
    videos_data = tf.stack(videos_data, axis=0)
    return videos_data

# 读取视频并将其转换为TensorFlow张量
# video_length = 15  # 每个视频的帧数
# video_paths = ['E:\\data\\cmp\\fvd\\gt\\video_4s_0202.mp4', 'E:\\data\\cmp\\fvd\\gt\\video_4s_0202.mp4']
# videos_data = read_video(video_paths, video_length, resize=(224, 224))  # 可选的调整大小参数
# print("视频帧张量的形状:", videos_data.shape)
#  (2, 15, 224, 224, 3)
# [NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3]
# 16 15 64 64 3
import tensorflow.compat.v1 as tf
from frechet_video_distance import *
tf.disable_v2_behavior()
with tf.Graph().as_default():
    video_length = 40  # 每个视频的帧数
    import os
    gt_dir = "E:\\data\\cmp\\fvd\\gt"
    my_dir = "E:\\data\\cmp\\fvd\\recon_videos"
    S2G_dir = "E:\\data\\cmp\\fvd\\S2G_oliver15_crop256"
    SDT_dir = "E:\\data\\cmp\\fvd\\SDT_oliver15_crop256"
    CMP = S2G_dir
    video_paths1 = sorted([os.path.join(gt_dir, x) for x in os.listdir(gt_dir)])
    video_paths2 = sorted([os.path.join(CMP, x) for x in os.listdir(my_dir)])
    videos_data1 = read_video(video_paths1, video_length, resize=(224, 224))  # 可选的调整大小参数
    videos_data2 = read_video(video_paths2, video_length, resize=(224, 224))  # 可选的调整大小参数
    print("视频帧张量的形状:", videos_data1.shape, videos_data2.shape)
    first_set_of_videos = videos_data1
    second_set_of_videos = videos_data2
    b1 = preprocess(first_set_of_videos, (224, 224))
    b2 = preprocess(second_set_of_videos, (224, 224))
    c1 = create_id3_embedding(b1)
    c2 = create_id3_embedding(b2)
    result = calculate_fvd(c1, c2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        print(sess.run(c1))
        print(sess.run(c2))
        print("FVD is: %.2f." % sess.run(result))

'''
video_paths1 = ['E:\\data\\cmp\\fvd\\gt\\video_4s_0202.mp4', 'E:\\data\\cmp\\fvd\\gt\\video_4s_0203.mp4']
video_paths2 = ['E:\\data\\cmp\\fvd\\recon_videos\\video_4s_0202.mp4', 'E:\\data\\cmp\\fvd\\recon_videos\\video_4s_0203.mp4']
FVD is: 1089.53.
FVD is: 847.49.


video_paths1 = ['E:\\data\\cmp\\fvd\\gt\\video_4s_0202.mp4', 'E:\\data\\cmp\\fvd\\gt\\video_4s_0203.mp4']
video_paths2 = ['E:\\data\\cmp\\fvd\\S2G_oliver15_crop256\\video_4s_0202.mp4', 'E:\\data\\cmp\\fvd\\S2G_oliver15_crop256\\video_4s_0203.mp4']
2 FVD is: 803.37.
15 FVD is: 627.20.
40 
video_paths1 = ['E:\\data\\cmp\\fvd\\gt\\video_4s_0202.mp4', 'E:\\data\\cmp\\fvd\\gt\\video_4s_0203.mp4']
video_paths2 = ['E:\\data\\cmp\\fvd\\SDT_oliver15_crop256\\video_4s_0202.mp4', 'E:\\data\\cmp\\fvd\\SDT_oliver15_crop256\\video_4s_0203.mp4']
FVD is: 1058.05.
FVD is: 693.47.
'''