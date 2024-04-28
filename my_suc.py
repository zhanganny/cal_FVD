import tensorflow.compat.v1 as tf
from frechet_video_distance import *
tf.disable_v2_behavior()

NUMBER_OF_VIDEOS = 16
VIDEO_LENGTH = 15
with tf.Graph().as_default():
    first_set_of_videos = tf.zeros([NUMBER_OF_VIDEOS, VIDEO_LENGTH, 224, 224, 3])
    second_set_of_videos = tf.ones([NUMBER_OF_VIDEOS, VIDEO_LENGTH, 224, 224, 3]) * 255
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
videos.name: sub:0 module_name fvd_kinetics-400_id3_module_sub_0
a?????? Tensor("fvd_kinetics-400_id3_module_sub_0/StatefulPartitionedCall:0", shape=(16, 400), dtype=float32)

videos.name: sub_1:0 module_name fvd_kinetics-400_id3_module_sub_1_0
a?????? Tensor("fvd_kinetics-400_id3_module_sub_1_0/StatefulPartitionedCall:0", shape=(16, 400), dtype=float32)
[[ 0.7422105   2.829727    0.39993525 ... -0.38092658  0.65539014
  -1.4021297 ]
 [ 0.7422105   2.829727    0.39993525 ... -0.38092658  0.65539014
  -1.4021297 ]
 [ 0.7422105   2.829727    0.39993525 ... -0.38092658  0.65539014
  -1.4021297 ]
 ...
 [ 0.7422105   2.829727    0.39993525 ... -0.38092658  0.65539014
  -1.4021297 ]
 [ 0.7422105   2.829727    0.39993525 ... -0.38092658  0.65539014
  -1.4021297 ]
 [ 0.7422112   2.8297267   0.39993495 ... -0.38092652  0.6553905
  -1.4021292 ]]
[[ 0.7422105   2.829727    0.39993525 ... -0.38092658  0.65539014
  -1.4021297 ]
 [ 0.7422105   2.829727    0.39993525 ... -0.38092658  0.65539014
  -1.4021297 ]
 [ 0.7422105   2.829727    0.39993525 ... -0.38092658  0.65539014
  -1.4021297 ]
 ...
 [ 0.7422105   2.829727    0.39993525 ... -0.38092658  0.65539014
  -1.4021297 ]
 [ 0.7422105   2.829727    0.39993525 ... -0.38092658  0.65539014
  -1.4021297 ]
 [ 0.7422112   2.8297267   0.39993495 ... -0.38092652  0.6553905
  -1.4021292 ]]

'''