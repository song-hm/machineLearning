import tensorflow as tf
import os

class Cifar(object):
    def __init__(self):

        # 初始化操作
        self.height = 32
        self.width = 32
        self.channels = 3

        # 字节数
        self.image_bytes = self.height * self.width * self.channels
        self.label_bytes = 1
        self.all_bytes = self.image_bytes + self.label_bytes

    def read_and_decode(self, file_list):

        # 1、构造文件名队列
        file_queue = tf.train.string_input_producer(file_list)

        # 2、读取与解码
        # 读取阶段
        reader = tf.FixedLengthRecordReader(self.all_bytes)
        # key文件名， value一个样本
        key, value = reader.read(file_queue)
        print("key:\n", key)
        print("value:\n", value)
        # 解码
        decoded = tf.decode_raw(value, tf.uint8)
        print("decoded:\n", decoded)

        # 将目标值与特征值切片分开
        label = tf.slice(decoded, [0], [self.label_bytes])
        image = tf.slice(decoded, [1], [self.image_bytes])
        print("label:\n", label)
        print("feature:\n", image)
        # 调整图像形状
        image_reshaped = tf.reshape(image, shape=[self.channels, self.height, self.width])
        print("image_reshaped:\n", image_reshaped)

        # 转置 将图片的顺序转为height,width,channels
        image_transpose = tf.transpose(image_reshaped, [1, 2, 0])
        print("image_transpose:\n", image_transpose)
        # 调整图像类型
        image_cast = tf.cast(image_transpose, dtype=tf.float32)

        # 3、批处理
        label_batch, image_batch = tf.train.batch([label, image_cast], batch_size=100, num_threads=1, capacity=100)
        print("label_batch：\n", label_batch)
        print("image_batch：\n", image_batch)

        # 开启会话
        with tf.Session() as sess:
            # 开启线程
            coordinator = tf.train.Coordinator()
            runners = tf.train.start_queue_runners(sess=sess, coord=coordinator)
            key_new, value_new, decoded_new, label_new, image_new, image_reshaped_new, image_transpose_new = \
                sess.run([key, value, decoded, label, image, image_reshaped, image_transpose])
            label_value, image_value = sess.run([label_batch, image_batch])
            print("key_new:\n", key_new)
            # print("value_new:\n", value_new)
            print("decoded_new:\n", decoded_new)
            print("label_new:\n", label_new)
            print("image_new:\n", image_new)
            print("image_reshaped_new:\n", image_reshaped_new)
            print("image_transpose_new:\n", image_transpose_new)
            print("label_value:\n", label_value)
            print("image_value:\n", image_value)

            # 关闭线程
            coordinator.request_stop()
            coordinator.join(runners)

        return None

if __name__ == '__main__':
    file_name = os.listdir("./cifar-10-batches-bin")
    # print("file_name:\n", file_name)
    # 构造文件名路径列表
    file_list = [os.path.join("./cifar-10-batches-bin", file) for file in file_name if file[-3:] == "bin"]
    # print("file_list:\n", file_list)

    # 实例化Cifar
    cifar = Cifar()
    cifar.read_and_decode(file_list)
