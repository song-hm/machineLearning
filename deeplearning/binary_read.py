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

    def read_and_decode(self):
        """
        读取二进制文件
        :param file_list:
        :return:
        """
        # 1、构造文件名队列
        file_name = os.listdir("./cifar-10-batches-bin")
        # print("file_name:\n", file_name)
        # 构造文件名路径列表
        file_list = [os.path.join("./cifar-10-batches-bin/", file) for file in file_name if file[-3:] == "bin"]
        # print("file_list:\n", file_list)
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
            # print("decoded_new:\n", decoded_new)
            # print("label_new:\n", label_new)
            # print("image_new:\n", image_new)
            # print("image_reshaped_new:\n", image_reshaped_new)
            # print("image_transpose_new:\n", image_transpose_new)
            print("label_value:\n", label_value)
            print("image_value:\n", image_value)

            # 关闭线程
            coordinator.request_stop()
            coordinator.join(runners)

        return label_value, image_value

    def writer_to_tfrecords(self, image_batch, label_batch):
        """
        将样本特征值和目标值一起写入tfrecords文件
        :param image:
        :param label:
        :return:
        """
        with tf.python_io.TFRecordWriter("cifar10.tfrecords") as writer:
            # 循环构造Example对象，并序列化写入文件
            for i in range(100):
                image = image_batch[i].tostring()
                label = label_batch[i][0]
                # print("tfrecords-image:\n", image)
                # print("tfrecords-label:\n", label)
                example = tf.train.Example(features=tf.train.Features(
                    feature={"image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                             "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])), }))
                # 将序列化后的example写入文件
                # example.SerializeToString()
                writer.write(example.SerializeToString())
        return None

    def read_records(self):
        """
        读取tfrecords文件
        :return:
        """
        # 1、构造文件名队列
        file_queue = tf.train.string_input_producer(["cifar10.tfrecords"])

        # 2、读取与解码
        # 读取
        reader = tf.TFRecordReader()
        key, value = reader.read(file_queue)
        # 解析example
        feature = tf.parse_single_example(value, features={"image": tf.FixedLenFeature([], tf.string),
                                                           "label": tf.FixedLenFeature([], tf.int64), })
        image = feature["image"]
        label = feature["label"]
        print("image:\n", image)
        print("label:\n", label)
        # 解码
        image_decoded = tf.decode_raw(image, tf.float32)
        print("image_decoded:\n", image_decoded)
        # 图像形状的调整
        image_reshape = tf.reshape(image_decoded, shape=[self.height, self.width, self.channels])
        print("image_reshape:\n", image_reshape)

        # 3、构造批处理队列
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=100, num_threads=2, capacity=100)
        print("image_batch:\n", image_batch)
        print("label_batch:\n", label_batch)

        # 开启会话
        with tf.Session() as sess:
            # 开启线程
            coordinator = tf.train.Coordinator()
            runners = tf.train.start_queue_runners(sess=sess, coord=coordinator)

            image_new, label_new, image_decoded_new, image_reshape_new = sess.run([image,
                                                                                   label, image_decoded, image_reshape])
            image_value, label_value = sess.run([image_batch, label_batch])

            print("image_new:\n", image_new)
            print("label_new:\n", label_new)
            print("image_decoded_new:\n", image_decoded_new)
            print("image_reshape_new:\n", image_reshape_new)
            print("image_value:\n", image_value)
            print("label_value:\n", label_value)

            # 线程回收
            coordinator.request_stop()
            coordinator.join(runners)

        return None

if __name__ == '__main__':
    # 实例化Cifar
    cifar = Cifar()
    # label_value, image_value = cifar.read_and_decode()
    # cifar.writer_to_tfrecords(image_value, label_value)
    cifar.read_records()

