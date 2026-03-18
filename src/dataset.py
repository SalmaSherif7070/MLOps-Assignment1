import tensorflow as tf
import os

IMG_SIZE = 256

def load_paired_images(inp_path, tar_path):
    inp = tf.io.read_file(inp_path)
    inp = tf.image.decode_jpeg(inp)
    inp = tf.image.resize(inp, [IMG_SIZE, IMG_SIZE])
    inp = (tf.cast(inp, tf.float32) / 127.5) - 1

    tar = tf.io.read_file(tar_path)
    tar = tf.image.decode_jpeg(tar)
    tar = tf.image.resize(tar, [IMG_SIZE, IMG_SIZE])
    tar = (tf.cast(tar, tf.float32) / 127.5) - 1

    return inp, tar

def create_dataset(dir_a, dir_b, batch=2):
    files_a = sorted([f for f in os.listdir(dir_a) if f.endswith('.jpg')])
    paths_a = [os.path.join(dir_a, f) for f in files_a]
    paths_b = [os.path.join(dir_b, f.replace('_A.jpg', '_B.jpg')) for f in files_a]

    ds = tf.data.Dataset.from_tensor_slices((paths_a, paths_b))
    ds = ds.map(load_paired_images, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.shuffle(400).batch(batch).prefetch(tf.data.AUTOTUNE)