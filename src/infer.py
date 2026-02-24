import tensorflow as tf
import matplotlib.pyplot as plt
import os

model = tf.keras.models.load_model("generator.h5")
test_dir = "data/sketch2pokemon/testA"

for filename in sorted(os.listdir(test_dir)):
    if not filename.endswith('.jpg'):
        continue
    
    img_path = os.path.join(test_dir, filename)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [256, 256])
    img = (tf.cast(img, tf.float32) / 127.5) - 1
    img = img[None, ...]
    
    pred = model(img, training=False)[0]
    plt.imshow((pred + 1) / 2)
    plt.title(filename)
    plt.axis("off")
    plt.show()