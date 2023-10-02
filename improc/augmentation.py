import tensorflow as tf
import tensorflow_addons as tfa
import math

class DataAugmentation():
    def __init__(self, config : dict):
        self.config = config
                        
    def flip_random_crop(self, image):
        # With random crops we also apply horizontal flipping.
        image = tf.image.random_flip_left_right(image)
        size_resize = int(self.config.getint('CROP_SIZE')*1.15)
        image = tf.image.resize_with_pad(image, size_resize, size_resize)
        image = tf.image.random_crop(image, (self.config.getint('CROP_SIZE'), self.config.getint('CROP_SIZE'), 3))
        return image
        
    def color_jitter(self, x, strength=[0.4, 0.4, 0.4, 0.1]):
        x = tf.image.random_brightness(x, max_delta=0.8 * strength[0])
        x = tf.image.random_contrast(
            x, lower=1 - 0.8 * strength[1], upper=1 + 0.8 * strength[1]
        )
        x = tf.image.random_saturation(
            x, lower=1 - 0.8 * strength[2], upper=1 + 0.8 * strength[2]
        )
        x = tf.image.random_hue(x, max_delta=0.2 * strength[3])
        # Affine transformations can disturb the natural range of
        # RGB images, hence this is needed.
        x = tf.clip_by_value(x, 0, 255)
        return x
    
    
    def color_drop(self, x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 3])
        return x
    
    
    def random_apply(self, func, x, p):
        if tf.random.uniform([], minval=0, maxval=1) < p:
            return func(x)
        else:
            return x
    
    def custom_augment(self, image):
        # As discussed in the SimCLR paper, the series of augmentation
        # transformations (except for random crops) need to be applied
        # randomly to impose translational invariance.        
        if self.config.get('DATASET') == 'MNIST' :            
            image = tf.image.grayscale_to_rgb(image)
            
        image = self.flip_random_crop(image)
        image = self.random_apply(self.color_jitter, image, p=0.8)
        image = self.random_apply(self.color_drop, image, p=0.2)        
        return image

    def random_sized_crop(self, x, img_size=224, min=0.8, max=1):
        crop_size = tf.cast(tf.math.ceil(
            img_size * tf.random.uniform([], minval=min, maxval=max)), tf.int32)
        x = tf.image.random_crop(x, (crop_size, crop_size, 3))
        x = tf.image.resize(x, (img_size, img_size))
        return x
    
    def random_line_skip(self, tensor, skip=0.1):
        tensor = tf.transpose(tensor, [1, 0, 2])
        num_rows = tf.shape(tensor)[0]
        num_selected_rows = tf.cast(tf.math.ceil(
            skip * tf.cast(num_rows, tf.float32)), tf.int32)
        indices = tf.range(num_rows)
        shuffled_indices = tf.random.shuffle(indices)
        selected_indices = shuffled_indices[:num_selected_rows]
        mask = tf.scatter_nd(tf.expand_dims(selected_indices, axis=1), tf.ones_like(
            selected_indices), tf.shape(tf.zeros(num_rows)))
        mask = tf.cast(mask, dtype=tf.bool)
        selected_rows = tf.where(tf.expand_dims(mask, axis=1), tf.cast(
            tf.fill(tf.shape(tensor), 0), tf.uint8), tensor)
        selected_rows = tf.transpose(selected_rows, [1, 0, 2])
        return selected_rows

    def random_rotation(self, x, angle=30):
        angle = math.radians(angle)
        rotation_angle = tf.random.uniform([], minval=-angle, maxval=angle)
        x = tfa.image.rotate(
            x, rotation_angle, fill_mode='constant', fill_value=0)
        x = tf.cast(x, tf.uint8)
        return x
    
    def sketch_augment(self, image):
        image = tf.image.grayscale_to_rgb(image)
        # line skip, rotation, flip, crop

        image = tf.image.resize(image, (224, 224))
        image = tf.cast(image, tf.uint8)

        image = self.random_apply(self.random_line_skip, image, 0.5)
        image = self.random_apply(self.random_rotation, image, 0.5)
        image = self.random_apply(tf.image.flip_left_right, image, 0.5)
        image = self.random_sized_crop(image)
        return image
    
    def image_augment(self, image):
        # As discussed in the SimCLR paper, the series of augmentation
        # transformations (except for random crops) need to be applied
        # randomly to impose translational invariance.                
        
        image = self.flip_random_crop(image)        
        image = self.random_apply(self.color_jitter, image, p=0.8)
        image = self.random_apply(self.color_drop, image, p=0.2)
        image = tf.expand_dims(image, axis = 0)
        image = tfa.image.random_cutout(image, (14,14), constant_values= 0)
        image = tf.squeeze(image, axis = 0)
        return image
    
    def get_augmentation_fun(self):
        if self.config.get('DATASET') == 'QD' :
            return self.sketch_augment
        
        if self.config.get('DATASET') == 'IMAGENET' :
            return self.image_augment
        
        return None 
    