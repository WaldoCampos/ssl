import models.resnet as resnet
from improc.imagenet_augmentation import fancy_pca, rescale_shorter_side
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset using TFDS
dataset_name = "imagenet1k"  # Replace with the name of your dataset
(train_dataset, validation_dataset, test_dataset), info = tfds.load(
    name=dataset_name,
    data_dir='/home/wcampos/datasets/',
    split=["train", "val", "test"],
    with_info=True,
    as_supervised=True,
)

# # Preprocess and augment the data
# def preprocess_image(image, label):
#     # Add your preprocessing code here
#     image = tf.image.resize(image, (224, 224))
#     image = tf.cast(image, tf.float32) / 127.5 - 1
#     return image, label

imgnet_mean=[0.485, 0.456, 0.406]
imgnet_std=[0.229, 0.224, 0.225]
def preprocess_image_training(image, label):
    image = rescale_shorter_side(image)
    image = tf.image.random_crop(image, (224, 224, 3))
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image / 255, tf.float32)
    means = tf.cast(tf.reshape(imgnet_mean, [1,1,3]), tf.float32)
    stds = tf.cast(tf.reshape(imgnet_std, [1,1,3]), tf.float32)
    image -= means
    image /= stds
    # image = image.numpy()
    # image = fancy_pca(image)
    # for idx in range(3):
    #     image[:, :, idx] -= imgnet_mean[idx]
    #     image[:, :, idx] /= imgnet_std[idx]
    # image = tf.convert_to_tensor(image, dtype=tf.float32)
    return image, label

def preprocess_image_val_test(image, label):
    image = rescale_shorter_side(image, 224)
    image = tf.keras.layers.CenterCrop(224, 224)(image)
    image = tf.cast(image / 255, tf.float32)
    means = tf.cast(tf.reshape(imgnet_mean, [1,1,3]), tf.float32)
    stds = tf.cast(tf.reshape(imgnet_std, [1,1,3]), tf.float32)
    image -= means
    image /= stds
    return image, label

def view(ds, n_rows, n_cols) :    
    _, ax = plt.subplots(n_rows, n_cols)
    for i in range(n_rows) :
        for j in range(n_cols) :
            ax[i,j].set_axis_off()

    ds = next(ds.as_numpy_iterator())
    imgs, _ = ds
    for i, img in enumerate(imgs):                                            
        sketch = img.astype(np.uint8)
        ax[i // n_cols][i % n_cols].imshow(sketch)
    plt.savefig('/home/wcampos/tests/ssl/my_plot.png')


train_dataset = train_dataset.map(preprocess_image_training)
validation_dataset = validation_dataset.map(preprocess_image_val_test)
test_dataset = test_dataset.map(preprocess_image_val_test)

# Batch and shuffle the datasets
batch_size = 64
train_dataset = train_dataset.batch(batch_size).shuffle(buffer_size=512, seed=1234)
validation_dataset = validation_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# view(train_dataset, 8, 8)

# Define your model architecture
class Pretrain(tf.keras.Model):
    def __init__(self, config_model, config_data, n_classes, **kwargs):
        super(Pretrain, self).__init__(**kwargs)
        self.CROP_SIZE = config_data.getint('CROP_SIZE')
        self.PROJECT_DIM =  config_model.getint('PROJECT_DIM')
        self.WEIGHT_DECAY = config_model.getfloat('WEIGHT_DECAY')
        self.MARGIN = config_model.getfloat('MARGIN')
        self.CHANNELS = 3
        self.encoder = self.get_encoder()
        self.classifier = tf.keras.layers.Dense(n_classes)

    def get_encoder(self, name="encoder"):
        inputs = tf.keras.layers.Input((self.CROP_SIZE, self.CROP_SIZE, self.CHANNELS))
        x = inputs # / 127.5 - 1
        #the backbone can be an input to the clas SimSiam
        bkbone = resnet.ResNetBackbone([3,4,6,3], [64,128, 256, 512], kernel_regularizer = tf.keras.regularizers.l2(self.WEIGHT_DECAY))
        x = bkbone(x)
        # Projection head.
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.PROJECT_DIM)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(self.PROJECT_DIM)(x)
        outputs = tf.math.l2_normalize(x, axis=1)

        return tf.keras.Model(inputs, outputs, name=name)
    
    def call(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        x = tf.keras.layers.Softmax()(x)
        return x
    
import configparser
import tensorflow as tf
import numpy as np

AUTO = tf.data.AUTOTUNE

# Cargamos el archivo de config para el modelo
config = configparser.ConfigParser()
config.read('/home/wcampos/rvdl/tarea2/siamese_networks/config/sketchy.ini')
config_model = config['RESNET']
config_data = config['DATA']
dataset_name = config_data.get('DATASET')

model = Pretrain(config_model, config_data, n_classes=1000)

# model = resnet.ResNet([3,4,6,3], [64,128, 256, 512], 1000, True)

# # Compile the model
# model.compile(optimizer=tf.keras.optimizers.Adam(weight_decay=1e-4),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model = tf.keras.applications.resnet50.ResNet50(weights=None)

# Create a ModelCheckpoint callback to save checkpoints every epochs
checkpoint = tf.keras.callbacks.ModelCheckpoint('/home/wcampos/rvdl/tarea2/saved_models/pretrain_photos_imagenet/best_acc/ckp',
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                mode='max',
                                                save_weights_only=True,
                                                save_freq='epoch',
                                                )
# Create the ReduceLROnPlateau callback
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3
)

# Create the SGD optimizer with weight decay and momentum
sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, weight_decay=0.0001)

# Compile the model with the SGD optimizer and add the callback
model.compile(optimizer=sgd_optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# To continue training
CONTINUE = True
if CONTINUE:
    last_epoch = 10 # Ultima epoch que entreno el modelo
    model.build(input_shape=(batch_size, 224, 224, 3))
    model.load_weights('/home/wcampos/rvdl/tarea2/saved_models/pretrain_photos_imagenet/best_acc/ckp')

# Train the model
history = model.fit(train_dataset,
                    epochs=50,  # Adjust the number of epochs
                    validation_data=validation_dataset,
                    callbacks=[checkpoint, reduce_lr_callback],
                    verbose=2,
                    initial_epoch=last_epoch if CONTINUE else 0
                    )

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(validation_dataset)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')