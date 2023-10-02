import models.resnet as resnet
import tensorflow as tf
import configparser
import os

class SketchSimSiam(tf.keras.Model):
    def __init__(self, config_data, config_model):
        super().__init__()        
        self.CROP_SIZE = config_data.getint('CROP_SIZE')
        self.PROJECT_DIM =  config_model.getint('PROJECT_DIM')
        self.WEIGHT_DECAY = config_model.getfloat('WEIGHT_DECAY')
        self.LATENT_DIM  = config_model.getint('LATENT_DIM')        
        self.CHANNELS = 3        
        print('{} {} {} {}'.format(self.CROP_SIZE, self.PROJECT_DIM, self.WEIGHT_DECAY, self.LATENT_DIM))
        self.encoder = self.get_encoder()
        self.predictor = self.get_predictor()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        
    def get_input_shape(self):
        return (self.CROP_SIZE, self.CROP_SIZE, self.CHANNELS)
                    
    def get_encoder(self):
        # Input and backbone.
        inputs = tf.keras.layers.Input((self.CROP_SIZE, self.CROP_SIZE, self.CHANNELS))                
        x = inputs / 127.5 - 1
        #the backbone can be an input to the clas SimSiam
        # bkbone = resnet.ResNetBackbone([3,4,6,3], [64,128, 256, 512], kernel_regularizer = tf.keras.regularizers.l2(self.WEIGHT_DECAY))
        bkbone = resnet.ResNet([3,4,6,3], [64,128, 256, 512], 1000, True)
        # Load the pre-trained weights
        pretrain_epoch = 100
        bkbone.build(input_shape=(1, 224, 224, 3))
        bkbone.load_weights(f'/home/wcampos/tests/ssl/saved_models/imagenet1k/resnet50/model_weights_epoch{pretrain_epoch}.h5')
        bkbone = bkbone.layers[-3]
        #bkbone = simple.Backbone()
        x = bkbone(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # Projection head.
        x = tf.keras.layers.Dense(
            self.PROJECT_DIM, 
            use_bias=False, 
            kernel_regularizer=tf.keras.regularizers.l2()
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
            self.PROJECT_DIM,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(self.WEIGHT_DECAY)
        )(x)
        outputs = tf.keras.layers.BatchNormalization()(x)

        return tf.keras.Model(inputs, outputs, name="encoder")


    def get_predictor(self):
        model = tf.keras.Sequential(
            [
                # Note the AutoEncoder-like structure.
                tf.keras.layers.Input((self.PROJECT_DIM,)),
                tf.keras.layers.Dense(
                    self.LATENT_DIM,
                    use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(self.WEIGHT_DECAY),
                ),
                tf.keras.layers.ReLU(),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(self.PROJECT_DIM ),
            ],
            name="predictor",
        )
        return model

    def compute_loss(self, p, z):
        # The authors of SimSiam emphasize the impact of
        # the `stop_gradient` operator in the paper as it
        # has an important role in the overall optimization.
        z = tf.stop_gradient(z)
        p = tf.math.l2_normalize(p, axis=1)
        z = tf.math.l2_normalize(z, axis=1)
        # Negative cosine similarity (minimizing this is
        # equivalent to maximizing the similarity).
        return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack the data.
        ds_one, ds_two = data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
            p1, p2 = self.predictor(z1), self.predictor(z2)
            # Note that here we are enforcing the network to match
            # the representations of two differently augmented batches
            # of data.
            loss = self.compute_loss(p1, z2) / 2 + self.compute_loss(p2, z1) / 2

        # Compute gradients and update the parameters.
        learnable_params = (
            self.encoder.trainable_variables + self.predictor.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
if __name__ == '__main__' :    
    config = configparser.ConfigParser()
    config.read('example.ini')
    config_model = config['SIMSIAM']
    config_data = config['DATA']
    simsiam = SimSiam(config_data, config_model)        
    simsiam.load_weights(config_data.get('MODEL_NAME'))
    for v in simsiam.encoder.trainable_variables :
        print(v.numpy)