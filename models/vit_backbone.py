import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


class Mlp(layers.Layer):
    def __init__(self, hidden_units, drop=0.1):
        super().__init__()
        self.layers = []
        for dim in hidden_units:
            self.layers.append(layers.Dense(dim, activation=tf.nn.gelu))
            self.layers.append(layers.Dropout(drop))

    def call(self, x):
        for f in self.layers:
            x = f(x)
        return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
class TransformerLayer(layers.Layer):
    def __init__(self, projection_dim, transformer_units, num_heads=4, drop=0.1):
        super().__init__()
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.mha1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=drop)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = Mlp(transformer_units)

    def call(self, x):
        x1 = self.ln1(x)
        x2 = x + self.mha1(x1, x1)
        x3 = self.ln2(x2)
        x3 = self.mlp(x3)
        return x2 + x3

class VitBackbone(tf.keras.Model):
    def __init__(self, image_size, patch_size, projection_dim, transformer_layers, transformer_units):
        super(VitBackbone, self).__init__()
        self.patches = Patches(patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.encoded_patches = PatchEncoder(num_patches, projection_dim)
        self.transformer_list = []
        for _ in range(transformer_layers):
            self.transformer_list.append(TransformerLayer(projection_dim, transformer_units))
        self.last_ln = keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Flatten(),
            ])

    def call(self, inputs):
        x = inputs
        x = self.patches(x)
        x = self.encoded_patches(x)
        for f in self.transformer_list:
            x = f(x)
        x = self.last_ln(x)
        x = tf.keras.activations.relu(x)
        return x
