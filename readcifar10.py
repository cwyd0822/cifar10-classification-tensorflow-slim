"""
对Cifar10打包成tfrecord的文件进行读取
"""

# 引入tensorflow
import tensorflow as tf

'''
batchsize: 每次读取的图片数量
type: 0表示训练集，1表示测试集
aug_data: 是否需要数据增强

返回batchsize大小的图片和相应的标签列表
'''
# 定义读取数据函数
def read(batchsize=64, type=1, aug_data=1):
    # 通过TFRecordReader读取.tfrecord数据集
    reader = tf.TFRecordReader()
    # 当type=0时，表示训练从训练集中进行读取
    if type == 0:
        # tfrecord是Tensorflow支持的数据集格式
        file_list = ["data/train.tfrecord"]
    # 当type=1时，表示测试从测试集中进行读取
    if type == 1:
        file_list = ["data/test.tfrecord"]
    filename_queue = tf.train.string_input_producer(
        file_list, num_epochs=None, shuffle=True
    )
    _, serialized_example = reader.read(filename_queue)

    batch = tf.train.shuffle_batch([serialized_example], batchsize, capacity=batchsize * 10,
                                   min_after_dequeue=batchsize * 5)

    feature = {'image': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.int64)}

    features = tf.parse_example(batch, features=feature)

    images = features["image"]

    # 将序列化的数据解码成uint8类型的数据
    img_batch = tf.decode_raw(images, tf.uint8)
    # 转成float32数据，是一个向量
    img_batch = tf.cast(img_batch, tf.float32)
    # 再把向量reshape成列表
    img_batch = tf.reshape(img_batch, [batchsize, 32, 32, 3])

    # 只对训练数据进行数据增强
    if type == 0 and aug_data == 1:
        # 图片随机裁剪
        distorted_image = tf.random_crop(img_batch,
                                         [batchsize, 28, 28, 3])
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.8,
                                                   upper=1.2)
        distorted_image = tf.image.random_hue(distorted_image,
                                              max_delta=0.2)
        distorted_image = tf.image.random_saturation(distorted_image,
                                                     lower=0.8,
                                                     upper=1.2)
        img_batch = tf.clip_by_value(distorted_image, 0, 255)

    img_batch = tf.image.resize_images(img_batch, [32, 32])
    label_batch = tf.cast(features['label'], tf.int64)

    # 归一化到-1,1
    img_batch = tf.cast(img_batch, tf.float32) / 128.0 - 1.0
    # 返回处理好的图片和标签
    return img_batch, label_batch


