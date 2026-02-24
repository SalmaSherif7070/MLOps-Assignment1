import tensorflow as tf

IMG_SIZE = 256

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1] // 2
    inp = image[:, :w, :]
    tar = image[:, w:, :]

    inp = (tf.cast(inp, tf.float32) / 127.5) - 1
    tar = (tf.cast(tar, tf.float32) / 127.5) - 1
    return inp, tar

def create_dataset(path, batch=2):
    ds = tf.data.Dataset.list_files(path + "/*.jpg")
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.shuffle(400).batch(batch).prefetch(tf.data.AUTOTUNE)