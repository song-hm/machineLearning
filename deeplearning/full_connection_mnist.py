import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

def full_connection():
    """
    用全连接来对手写数字进行识别
    :return:
    """
    with tf.variable_scope("prepare_data"):
        # 1、准备数据
        mnist_data = input_data.read_data_sets("./mnist_data", one_hot=True)
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="feature")
        y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="label")

    with tf.variable_scope("create_model"):
        # 2、构造模型
        weights = tf.Variable(initial_value=tf.random_normal(shape=[784, 10]), name="weights")
        bias = tf.Variable(initial_value=tf.random_normal(shape=[10]), name="bias")
        y_predict = tf.matmul(x, weights) + bias

    with tf.variable_scope("loss_function"):
        # 3、构造损失函数
        error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict), name="error")

    with tf.variable_scope("optimizer"):
        # 4、优化损失
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(error, name="optimizer")

    # 2_收集变量
    tf.summary.scalar("error", error)
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("bias", bias)
    # 3_合并变量
    merge_all = tf.summary.merge_all()

    with tf.variable_scope("accuracy"):
        # 5、准确率计算
        # 1）比较输出结果的最大值所在位置和真实值所在的位置
        # np.argmax()
        equal_list = tf.equal(tf.argmax(y_true, 1, name="y_true"), tf.argmax(y_predict, 1, name="y_predict"),
                              name="equal_list")
        # 求平均
        accuracy = tf.reduce_mean(tf.cast(equal_list, dtype=tf.float32), name="accuracy")

    # 创建saver
    saver = tf.train.Saver()

    # 初始化变量
    init = tf.global_variables_initializer()
    # 开启会话
    with tf.Session() as sess:
        sess.run(init)
        # 1_创建事件文件
        file_writer = tf.summary.FileWriter("./tmp/mnist/", graph=sess.graph)
        image, label = mnist_data.train.next_batch(3000)
        print("训练前，损失为%f" % sess.run(error, feed_dict={x: image, y_true: label}))
        # 开始训练
        for i in range(5000):
            _, loss, accuracy_value = sess.run([optimizer, error, accuracy], feed_dict={x: image, y_true: label})
            if (i+1) % 100 == 0:
                print("第%d次训练的损失为%f,准确率为%f" % (i+1, loss, accuracy_value))

            # 运行合并变量操作
            summary = sess.run(merge_all, feed_dict={x: image, y_true: label})
            # 将每次迭代后的事件写入事件文件
            file_writer.add_summary(summary, i)

            # 保存模型
            if (i+1) % 500 == 0:
                saver.save(sess, "./tmp/mnist/full_connection.ckpt")

    return None

def test_full_connect():
    """
    测试模型
    :return:
    """
    with tf.variable_scope("prepare_data"):
        # 1、准备数据
        mnist_data = input_data.read_data_sets("./mnist_data", one_hot=True)
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="feature")
        y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="label")

    with tf.variable_scope("create_model"):
        # 2、构造模型
        weights = tf.Variable(initial_value=tf.random_normal(shape=[784, 10]), name="weights")
        bias = tf.Variable(initial_value=tf.random_normal(shape=[10]), name="bias")
        y_predict = tf.matmul(x, weights) + bias

    with tf.variable_scope("loss_function"):
        # 3、构造损失函数
        error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict), name="error")

    # 2_收集变量
    tf.summary.scalar("error", error)
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("bias", bias)
    # 3_合并变量
    merge_all = tf.summary.merge_all()

    with tf.variable_scope("accuracy"):
        # 5、准确率计算
        # 1）比较输出结果的最大值所在位置和真实值所在的位置
        # np.argmax()
        equal_list = tf.equal(tf.argmax(y_true, 1, name="y_true"), tf.argmax(y_predict, 1, name="y_predict"),
                              name="equal_list")
        # 求平均
        accuracy = tf.reduce_mean(tf.cast(equal_list, dtype=tf.float32), name="accuracy")

    # 创建saver
    saver = tf.train.Saver()

    # 初始化变量
    init = tf.global_variables_initializer()
    # 开启会话
    with tf.Session() as sess:
        sess.run(init)
        # 1_创建事件文件
        file_writer = tf.summary.FileWriter("./tmp/mnist/", graph=sess.graph)

        # 加载模型
        if  os.path.exists("./tmp/mnist/checkpoint"):
            model = saver.restore(sess, "./tmp/mnist/full_connection.ckpt")

        for i in range(100):
            # 每次哪一个样本测试
            mnist_x, mnist_y = mnist_data.test.next_batch(1)
            print("第%d个样本的，真实值为：%d, 模型预测为：%d" %
                  (i+1,
                   tf.argmax(sess.run(y_true, feed_dict={x: mnist_x, y_true: mnist_y}), 1, name="y_true").eval(),
                   tf.argmax(sess.run(y_predict, feed_dict={x: mnist_x, y_true: mnist_y}), 1, name="y_predict").eval(),
                   ))

    return None

if __name__ == '__main__':
    # 模型训练
    # full_connection()
    # 模型测试
    test_full_connect()


