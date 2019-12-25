import tensorflow as tf
import readcifar10
slim = tf.contrib.slim
import os
import resnet


# 定义网络结构
# image：一张图像
# 返回10维的向量
def model(image, keep_prob=0.8, is_training=True):
    batch_norm_params = {
        "is_training": is_training,
        "epsilon": 1e-5,  # 防止除以0
        "decay": 0.997,  # 衰减系数
        'scale': True,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
        [slim.conv2d],  # 设置卷积默认参数的初始化
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,  # 激活函数初始化
        weights_regularizer=slim.l2_regularizer(0.0001),  # l2正则
        normalizer_fn=slim.batch_norm,  # batch norm层参数初始化
        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.max_pool2d], padding="SAME"):  # 设置默认的max pooling的初始化
            net = slim.conv2d(image, 32, [3, 3], scope='conv1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')  # 进行2倍下采样
            net = slim.conv2d(net, 64, [3, 3], scope='conv3')
            net = slim.conv2d(net, 64, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
            net = slim.conv2d(net, 128, [3, 3], scope='conv5')
            net = slim.conv2d(net, 128, [3, 3], scope='conv6')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool3')
            net = slim.conv2d(net, 256, [3, 3], scope='conv7')
            net = tf.reduce_mean(net, axis=[1, 2])  # nhwc--n11c 对特征图求均值
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1024)
            slim.dropout(net, keep_prob)  # 添加dropout层
            net = slim.fully_connected(net, 10)
    return net  # 10 dim vec


def loss(logits, label):
    one_hot_label = slim.one_hot_encoding(label, 10)  # 对label进行one hot编码
    slim.losses.softmax_cross_entropy(logits, one_hot_label)  # 交叉熵损失

    # 正则化的损失
    reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)  # 正则化的loss集合
    l2_loss = tf.add_n(reg_set)
    slim.losses.add_loss(l2_loss)

    totalloss = slim.losses.get_total_loss()

    return totalloss, l2_loss


# 优化器
def func_optimal(batchsize, loss_val):
    # 从0开始
    global_step = tf.Variable(0, trainable=False)
    # 定义指数衰减率的学习率
    lr = tf.train.exponential_decay(0.01,
                                    global_step,
                                    decay_steps=50000 // batchsize,  # 衰减步长
                                    decay_rate=0.95,  #
                                    staircase=False)  # 以平滑的形式
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        op = tf.train.AdamOptimizer(lr).minimize(loss_val, global_step)
    return global_step, op, lr


def train():
    """
    训练的函数
    :return:
    """
    batchsize = 64
    folder_log = 'logdirs'  # 日志存放的路径
    folder_model = 'model'  # Model存放的路径

    if not os.path.exists(folder_log):
        os.mkdir(folder_log)

    if not os.path.exists(folder_model):
        os.mkdir(folder_model)

    tr_summary = set()
    te_summary = set()

    # data
    tr_im, tr_label = readcifar10.read(batchsize, 0, 1)  # 训练样本
    te_im, te_label = readcifar10.read(batchsize, 1, 0)  # 测试样本

    # 定义网络
    # 定义数据
    input_data = tf.placeholder(tf.float32, shape=[None, 32, 32, 3],
                                name='input_data')
    # 定义标签
    input_label = tf.placeholder(tf.int64, shape=[None],
                                name='input_label')
    keep_prob = tf.placeholder(tf.float32, shape=None,
                                name='keep_prob')

    is_training = tf.placeholder(tf.bool, shape=None,
                               name='is_training')
    # logits = resnet.model_resnet(input_data, keep_prob=keep_prob, is_training=is_training)
    logits = model(input_data, keep_prob=keep_prob, is_training=is_training)

    # 定义loss

    total_loss, l2_loss = loss(logits, input_label)

    tr_summary.add(tf.summary.scalar('train total loss', total_loss))
    te_summary.add(tf.summary.scalar('test l2_loss', l2_loss))

    tr_summary.add(tf.summary.scalar('train total loss', total_loss))
    te_summary.add(tf.summary.scalar('test l2_loss', l2_loss))

    # 获取accurancy
    pred_max = tf.argmax(logits, 1)
    correct = tf.equal(pred_max, input_label)
    accurancy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tr_summary.add(tf.summary.scalar('train accurancy', accurancy))
    te_summary.add(tf.summary.scalar('test accurancy', accurancy))
    # op
    global_step, op, lr = func_optimal(batchsize, total_loss)
    tr_summary.add(tf.summary.scalar('train lr', lr))
    te_summary.add(tf.summary.scalar('test lr', lr))

    tr_summary.add(tf.summary.image('train image', input_data * 128 + 128))  # 这里的图片数据是处理过的
    te_summary.add(tf.summary.image('test image', input_data * 128 + 128))

    with tf.Session() as sess:
        # 对全局变量和局部变量进行初始化
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))

        # 启动多线程管理器
        tf.train.start_queue_runners(sess=sess,
                                     coord=tf.train.Coordinator())


        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        ckpt = tf.train.latest_checkpoint(folder_model)

        if ckpt:
            saver.restore(sess, ckpt)

        epoch_val = 100

        # 合并日志信息
        tr_summary_op = tf.summary.merge(list(tr_summary))
        te_summary_op = tf.summary.merge(list(te_summary))

        summary_writer = tf.summary.FileWriter(folder_log, sess.graph)

        for i in range(50000 * epoch_val):
            # 每次获取一个batch size的数据
            train_im_batch, train_label_batch = \
                sess.run([tr_im, tr_label])
            feed_dict = {
                input_data: train_im_batch,
                input_label: train_label_batch,
                keep_prob: 0.8,
                is_training: True
            }

            _, global_step_val, \
            lr_val, \
            total_loss_val, \
            accurancy_val, tr_summary_str = sess.run([op,
                                      global_step,
                                      lr,
                                      total_loss,
                                      accurancy, tr_summary_op],
                     feed_dict=feed_dict)

            summary_writer.add_summary(tr_summary_str, global_step_val)

            # 每隔100次打印
            if i % 100 == 0:
                print("{},{},{},{}".format(global_step_val,
                                           lr_val, total_loss_val,
                                           accurancy_val))

            if i % (50000 // batchsize) == 0:
                test_loss = 0
                test_acc = 0
                for ii in range(10000//batchsize):
                    test_im_batch, test_label_batch = \
                        sess.run([te_im, te_label])
                    feed_dict = {
                        input_data: test_im_batch,
                        input_label: test_label_batch,
                        keep_prob: 1.0,
                        is_training: False
                    }

                    total_loss_val, global_step_val, \
                    accurancy_val, te_summary_str = sess.run([total_loss,global_step,
                                              accurancy, te_summary_op],
                                             feed_dict=feed_dict)

                    summary_writer.add_summary(te_summary_str, global_step_val)

                    test_loss += total_loss_val
                    test_acc += accurancy_val

                print('test：', test_loss * batchsize / 10000,
                      test_acc * batchsize / 10000)

            if i % 1000 == 0:
                saver.save(sess, "{}/model.ckpt{}".format(folder_model, str(global_step_val)))
    return

if __name__ == '__main__':
    train()