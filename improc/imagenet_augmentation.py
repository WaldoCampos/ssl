import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx

def fancy_pca(img, alpha_std=0.1):
    orig_img = img.astype(float).copy()

    img = img / 255.0  # rescale to 0 to 1 range

    # flatten image to columns of RGB
    img_rs = img.reshape(-1, 3)
    # img_rs shape (640000, 3)

    # center mean
    img_centered = img_rs - np.mean(img_rs, axis=0)

    # paper says 3x3 covariance matrix
    img_cov = np.cov(img_centered, rowvar=False)

    # eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)

#     eig_vals [0.00154689 0.00448816 0.18438678]

#     eig_vecs [[ 0.35799106 -0.74045435 -0.56883192]
#      [-0.81323938  0.05207541 -0.57959456]
#      [ 0.45878547  0.67008619 -0.58352411]]

    # sort values and vector
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))

    # get 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))
    # according to the paper alpha should only be draw once per augmentation (not once per channel)
    alpha = np.random.normal(0, alpha_std)

    # broad cast to speed things up
    m2[:, 0] = alpha * eig_vals[:]

    add_vect = np.matrix(m1) * np.matrix(m2)
    orig_img /= 255.0
    for idx in range(3):   # RGB
        orig_img[..., idx] += add_vect[idx]

    orig_img = np.clip(orig_img, 0.0, 255.0)
    # orig_img /= 255
    orig_img *= 255.0
    return orig_img


def fancy_pca_tf(img, alpha_std=0.1):
    orig_img = tf.cast(tf.identity(img), tf.float32)
    img = tf.cast(img, tf.float32)
    img = img / 255.0
    img_rs = tf.reshape(img, [-1, 3])
    img_centered = img_rs - tf.math.reduce_mean(img_rs, axis=0)
    img_cov = tfp.stats.covariance(img_centered)
    eig_vals, eig_vecs = tf.linalg.eigh(img_cov)
    # sorted_indices = tf.argsort(eig_vals)
    # eig_vals = tf.sort(eig_vals)
    # eig_vecs = tf.gather(eig_vecs, sorted_indices, axis=1)
    alpha = tf.random.normal((1,), mean=0.0, stddev=alpha_std)
    eig_vals = tf.expand_dims(alpha * eig_vals, axis=-1)
    add_vect = tf.matmul(eig_vecs, eig_vals)
    orig_img /= 255.0
    orig_img += tf.reshape(add_vect, [1,1,3])
    orig_img *= 255.0
    orig_img = tf.clip_by_value(orig_img, 0.0, 255.0)
    return orig_img

def rescale_shorter_side(img, shorter_side_length=None):
    if not shorter_side_length:
        shorter_side_length = tf.cast(tf.random.uniform(shape=[], minval=256, maxval=480, dtype=tf.int32), tf.float64)
    height, width, _ = tf.split(tf.cast(tf.shape(img), tf.float64), 3)
    minval = tf.math.minimum(height, width)
    factor = shorter_side_length / minval
    new_shape = tf.cast(tf.concat([height * factor, width * factor], axis=0), tf.int32)
    img = tf.image.resize(img, new_shape)
    return img

if __name__ == '__main__':
    img = tf.random.uniform((532532, 532, 3), minval=0, maxval=255, dtype=tf.int32)
    img = rescale_shorter_side(img, 224)
    print(tf.shape(img).numpy())

    # from PIL import Image
    # image = Image.open('/home/wcampos/tests/tesis/augmentations/waldo.jpg')
    # image_array  = tf.convert_to_tensor(image)
    # output = fancy_pca_tf(image_array)
    # output = tf.cast(output, tf.uint8)
    # output = tf.io.encode_jpeg(output)
    # tf.io.write_file('/home/wcampos/tests/tesis/augmentations/waldo_pca.jpg', output)

    # input = tf.random.uniform((3,3,3), minval=0, maxval=255, dtype=tf.int32)
    # print("Input original:")
    # print(input.numpy())
    # output_1 = fancy_pca(input.numpy())
    # print("Output numpy:")
    # print(output_1)
    # output_2 = fancy_pca_tf(input)
    # print("Output tf:")
    # print(output_2.numpy())
