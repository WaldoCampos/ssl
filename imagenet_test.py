import tensorflow as tf
import tensorflow_datasets as tfds
import models.resnet as resnet
from improc.imagenet_augmentation import rescale_shorter_side

imgnet_mean=[0.485, 0.456, 0.406]
imgnet_std=[0.229, 0.224, 0.225]
def preprocess_image_val_test(image, label):
    image = rescale_shorter_side(image, 224)
    image = tf.keras.layers.CenterCrop(224, 224)(image)
    image = tf.cast(image / 255, tf.float32)
    means = tf.cast(tf.reshape(imgnet_mean, [1,1,3]), tf.float32)
    stds = tf.cast(tf.reshape(imgnet_std, [1,1,3]), tf.float32)
    image -= means
    image /= stds
    return image, label

dataset_name = "imagenet1k"  # Replace with the name of your dataset
(train_dataset, validation_dataset, test_dataset), info = tfds.load(
    name=dataset_name,
    data_dir='/home/wcampos/datasets/',
    split=["train", "val", "test"],
    with_info=True,
    as_supervised=True,
)
batch_size = 1024
test_dataset = test_dataset.map(preprocess_image_val_test).batch(batch_size)
validation_dataset = validation_dataset.map(preprocess_image_val_test).batch(batch_size)

model = resnet.ResNet([3,4,6,3], [64,128, 256, 512], 1000, True)

sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, weight_decay=0.0001)

model.compile(optimizer=sgd_optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.build(input_shape=(batch_size, 224, 224, 3))

model.load_weights('/home/wcampos/tests/ssl/saved_models/imagenet1k/resnet50/model_weights_epoch24.h5')

# Test
test_loss, test_accuracy = model.evaluate(validation_dataset)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')