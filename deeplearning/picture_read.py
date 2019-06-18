import tensorflow as tf
import os

def picture_read(file_list):
    """
    狗图片读取
    :return:
    """
    # 1、构造文件名队列
    file_queue = tf.train.string_input_producer(file_list)

    # 2、读取与解码
    #  读取阶段
    reader = tf.WholeFileReader()
    # key文件名 value一张图片的原始编码形式
    key, value = reader.read(file_queue)
    print("key:\n", key)
    print("value:\n", value)
    # 解码阶段
    image = tf.image.decode_jpeg(value)
    print("image:\n", image)

    # 图像的形状和大小的修改
    resize_images = tf.image.resize_images(image, [200, 200])
    print("resize_images:\n", resize_images)
    # 静态形状的修改
    resize_images.set_shape(shape=[200, 200, 3])

    # 3、批处理
    image_batch = tf.train.batch([resize_images], batch_size=100, num_threads=1, capacity=100)
    print("image_batch:\n", image_batch)

    # 开启会话
    with tf.Session() as sess:
        # 开启线程
        # 线程协调员
        coordinator = tf.train.Coordinator()
        runners = tf.train.start_queue_runners(sess, coord=coordinator)
        key_new, value_new, image_new, resize_images, image_batch_new = sess.run([key, value, image,
                                                                                  resize_images, image_batch])
        print("key_new:\n", key_new)
        # print("value_new:\n", value_new)
        print("image_new:\n", image_new)
        print("resize_images:\n", resize_images)
        print("image_batch_new:\n", image_batch_new)

        # 回收线程
        coordinator.request_stop()
        coordinator.join(runners)
    return None

if __name__ == '__main__':
    # 构造路径加文件名的列表
    filename = os.listdir("./dog")
    # print(filename)
    # 拼接路径加文件名
    file_list = [os.path.join("./dog", file) for file in filename]
    # print(file_list)
    # 狗图片读取
    picture_read(file_list)

