import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("generator.h5")

img = tf.io.read_file("sample.jpg")
img = tf.image.decode_jpeg(img)
img = tf.image.resize(img, [256,256])
img = (tf.cast(img, tf.float32) / 127.5) - 1
img = img[None,...]

pred = model(img, training=False)[0]
plt.imshow((pred+1)/2)
plt.axis("off")
plt.show()