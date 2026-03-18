import tensorflow as tf

def down(filters, size, bn=True):
    x = tf.keras.Sequential()
    x.add(tf.keras.layers.Conv2D(filters, size, 2, "same", use_bias=False))
    if bn: x.add(tf.keras.layers.BatchNormalization())
    x.add(tf.keras.layers.LeakyReLU())
    return x

def up(filters, size, drop=False):
    x = tf.keras.Sequential()
    x.add(tf.keras.layers.Conv2DTranspose(filters, size, 2, "same", use_bias=False))
    x.add(tf.keras.layers.BatchNormalization())
    if drop: x.add(tf.keras.layers.Dropout(0.5))
    x.add(tf.keras.layers.ReLU())
    return x

def Generator():
    inputs = tf.keras.Input([256,256,3])
    d = [down(64,4,False), down(128,4), down(256,4), down(512,4)]
    u = [up(256,4), up(128,4), up(64,4)]
    x, skips = inputs, []
    for layer in d:
        x = layer(x); skips.append(x)
    skips = reversed(skips[:-1])
    for layer, skip in zip(u, skips):
        x = layer(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    out = tf.keras.layers.Conv2DTranspose(3,4,2,"same",activation="tanh")(x)
    return tf.keras.Model(inputs, out)

def Discriminator():
    inp = tf.keras.Input([256,256,3])
    tar = tf.keras.Input([256,256,3])
    x = tf.keras.layers.Concatenate()([inp, tar])
    x = down(64,4,False)(x)
    x = down(128,4)(x)
    x = down(256,4)(x)
    out = tf.keras.layers.Conv2D(1,4,1,"same")(x)
    return tf.keras.Model([inp,tar], out)