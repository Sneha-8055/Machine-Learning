# # import the necessary libraries
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from itertools import product

# # set the param 
# plt.rc('figure', autolayout=True)
# plt.rc('image', cmap='magma')

# # define the kernel
# kernel = tf.constant([[-1, -1, -1],
#                     [-1,  8, -1],
#                     [-1, -1, -1],
#                    ])

# # load the image
# image = tf.io.read_file('ganesh.jpg')
# image = tf.io.decode_jpeg(image, channels=1)
# image = tf.image.resize(image, size=[300, 300])

# # plot the image
# img = tf.squeeze(image).numpy()
# plt.figure(figsize=(5, 5))
# plt.imshow(img, cmap='gray')
# plt.axis('off')
# plt.title('Original Gray Scale image')
# plt.show();


# # Reformat
# image = tf.image.convert_image_dtype(image, dtype=tf.float32)
# image = tf.expand_dims(image, axis=0)
# kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
# kernel = tf.cast(kernel, dtype=tf.float32)

# # convolution layer
# conv_fn = tf.nn.conv2d

# image_filter = conv_fn(
#     input=image,
#     filters=kernel,
#     strides=1, # or (1, 1)
#     padding='SAME',
# )

# plt.figure(figsize=(15, 5))

# # Plot the convolved image
# plt.subplot(1, 3, 1)

# plt.imshow(
#     tf.squeeze(image_filter)
# )
# plt.axis('off')
# plt.title('Convolution')

# # activation layer
# relu_fn = tf.nn.relu
# # Image detection
# image_detect = relu_fn(image_filter)

# plt.subplot(1, 3, 2)
# plt.imshow(
#     # Reformat for plotting
#     tf.squeeze(image_detect)
# )

# plt.axis('off')
# plt.title('Activation')

# # Pooling layer
# pool = tf.nn.pool
# image_condense = pool(input=image_detect, 
#                              window_shape=(2, 2),
#                              pooling_type='MAX',
#                              strides=(2, 2),
#                              padding='SAME',
#                             )

# plt.subplot(1, 3, 3)
# plt.imshow(tf.squeeze(image_condense))
# plt.axis('off')
# plt.title('Pooling')
# plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread('ganesh.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (300, 300))

# Show original image
plt.figure(figsize=(5,5))
plt.imshow(image, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')
plt.show()

# Define kernel (edge detection)
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Convolution function
def convolution(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad = kh // 2
    padded = np.pad(image, pad, mode='constant')
    output = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output

# Apply convolution
conv_img = convolution(image, kernel)

# ReLU activation
relu_img = np.maximum(0, conv_img)

# Max pooling function
def max_pool(image, size=2, stride=2):
    h, w = image.shape
    output = np.zeros((h//2, w//2))

    for i in range(0, h-size+1, stride):
        for j in range(0, w-size+1, stride):
            region = image[i:i+size, j:j+size]
            output[i//2, j//2] = np.max(region)

    return output

# Apply pooling
pool_img = max_pool(relu_img)

# Plot results
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(conv_img, cmap='gray')
plt.title('Convolution')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(relu_img, cmap='gray')
plt.title('ReLU Activation')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(pool_img, cmap='gray')
plt.title('Max Pooling')
plt.axis('off')

plt.show()
