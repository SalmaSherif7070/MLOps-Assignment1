import tensorflow as tf
from dataset import create_dataset
from model import Generator, Discriminator

LAMBDA = 100
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

G = Generator()
D = Discriminator()

g_opt = tf.keras.optimizers.Adam(2e-4, 0.5)
d_opt = tf.keras.optimizers.Adam(2e-4, 0.5)

@tf.function
def train_step(inp, tar):
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        gen = G(inp, training=True)

        d_real = D([inp, tar], training=True)
        d_fake = D([inp, gen], training=True)

        g_gan_loss = loss(tf.ones_like(d_fake), d_fake)
        l1_loss = tf.reduce_mean(tf.abs(tar - gen))
        g_loss = g_gan_loss + LAMBDA * l1_loss

        d_loss = (
            loss(tf.ones_like(d_real), d_real) +
            loss(tf.zeros_like(d_fake), d_fake)
        )

    g_opt.apply_gradients(
        zip(g_tape.gradient(g_loss, G.trainable_variables), G.trainable_variables)
    )
    d_opt.apply_gradients(
        zip(d_tape.gradient(d_loss, D.trainable_variables), D.trainable_variables)
    )

    return g_loss, d_loss, l1_loss

train_ds = create_dataset("data/pokemon_pix2pix_dataset/train")

for epoch in range(5):
    for inp, tar in train_ds:
        train_step(inp, tar)
    print(f"Epoch {epoch+1} done")

G.save("generator.h5")