import tensorflow as tf
from models.resnet import ResNetBackbone


class MLP(tf.keras.layers.Layer):
    """
    A simple 2-layer MLP
    """
    def __init__(self, hidden_features, out_features, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_features, activation=tf.nn.gelu)
        self.dense2 = tf.keras.layers.Dense(out_features)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        y = self.dropout(x)
        return y
    
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, projection_dim, num_heads=4, dropout_rate=0.1):
        super(AttentionLayer, self).__init__()
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(projection_dim * 2, projection_dim, dropout_rate)
            
    def call(self, x):
        # Layer normalization 1.
        x1 = self.norm1(x) # encoded_patches
        # Create a multi-head attention layer.
        attention_output = self.attn(x1, x1)
        # Skip connection 1.
        x2 = tf.keras.layers.Add()([attention_output, x]) #encoded_patches
        # Layer normalization 2.
        x3 = self.norm2(x2)
        # MLP.
        x3 = self.mlp(x3)
        # Skip connection 2.
        y = tf.keras.layers.Add()([x3, x2])
        return y

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, projection_dim, num_heads=4, num_blocks=1, dropout_rate=0.1, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.blocks = [AttentionLayer(projection_dim, num_heads, dropout_rate) for _ in range(num_blocks)]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(0.5)        
        
    def position_encoding(self, d, n):    
        """
        return nxd
        """
        vals_i = tf.cast(tf.reshape(tf.range(d), (1,-1)), tf.float32)
        vals_i = tf.tile(vals_i, (n, 1))
        pos = tf.cast(tf.reshape(tf.range(n), (1,-1)), tf.float32)
        pos = tf.transpose(tf.tile(pos, (d, 1)))                
        sins  = tf.math.sin(pos / tf.math.pow(10000.0, 2.0*vals_i / tf.cast(d, tf.float32)))
        cosins  = tf.math.cos(pos / tf.math.pow(10000.0, 2.0*vals_i / tf.cast(d, tf.float32)))
        pe =  tf.where(tf.equal(tf.math.floormod(vals_i, 2),0), sins, cosins) 
        return pe
    
    def call(self, x ):
        # adding positiona encoding
        d = tf.shape(x)[2]
        n = tf.shape(x)[1]
        b = tf.shape(x)[0]
        pos = tf.reshape(self.position_encoding(d, n), (1, n, d))
        pos = tf.tile(pos, [b,1,1])        
        x = pos + x           
        for block in self.blocks:
            x = block(x)            
        x = self.norm(x)
        y = self.dropout(x)
        return y        
    
    
        
class ResNetAttBackbone(tf.keras.Model):            
    def __init__(self, block_sizes, filters, use_bottleneck = False, se_factor = 0, kernel_regularizer = None, num_att_heads=4, num_att_blocks=2, **kwargs) :
        super(ResNetAttBackbone, self).__init__(**kwargs)
        self.encoder = ResNetBackbone(block_sizes, filters, use_bottleneck, se_factor, kernel_regularizer = kernel_regularizer, name = 'encoder')
        self.att_block = AttentionBlock(projection_dim=512, num_heads=num_att_heads, num_blocks=num_att_blocks)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        
    def call(self, inputs):
        x = inputs
        x = self.encoder(x)
        b = tf.shape(x)[0]
        c = tf.shape(x)[-1]
        x = tf.reshape(x, [b, -1, c])
        x = self.att_block(x)
        x = self.avg_pool(x)
        return x
    
if __name__ == '__main__':
    bkbone = ResNetAttBackbone([3,4,6,3], [64,128, 256, 512])
    x = tf.random.uniform(shape=(10, 224, 224, 3), minval=0, maxval=1)
    y = bkbone(x)
    input()
