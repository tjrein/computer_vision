from PIL import Image
import numpy as np
import imageio
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.models import Model
import cv2

tf.enable_eager_execution()

height = 200
width = 200

def preprocess_image(img):
    img = vgg19.preprocess_input(img)
    return img

def deprocess(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def load_image(filename):
    image = Image.open(filename).resize((height, width))
    return np.float32(image)


def calculate_content_loss(content_features, const_content_features):
    test_l_content = 0.5 * K.sum(K.square(content_features - const_content_features))

    return test_l_content

def calculate_gram_matrix(features):
    squeezed = K.squeeze(features, 0)
    test = K.batch_flatten(K.permute_dimensions(squeezed, (2, 0, 1)))
    gram = K.dot(test, K.transpose(test))

    return gram

def calculate_style_loss(style_features, const_style_features):
    style_loss = 0
    for i in range(len(style_features)):
        feature_shape = style_features[i].shape.as_list()
        N = feature_shape[-1]
        M = feature_shape[1] * feature_shape[2]

        const_gram = calculate_gram_matrix(const_style_features[i])
        style_gram = calculate_gram_matrix(style_features[i])

        test = K.sum(K.square(style_gram - const_gram))
        factor = 1 / (4.0 * N**2 * M**2)
        gram_error = factor * test
        style_loss += gram_error * 1/5

    return style_loss

def calculate_complete_gradient(features_tup, models_tup, constructed_img):
    content_features, style_features = features_tup
    content_model, style_model = models_tup
    with tf.GradientTape() as gradient_tape:
        constructed_content_features = content_model(constructed_img)
        constructed_style_features = style_model(constructed_img)

        loss = 10 * calculate_content_loss(content_features, constructed_content_features) + \
               100 * calculate_style_loss(style_features, constructed_style_features)

    return gradient_tape.gradient(loss, constructed_img)

def calculate_content_gradient(content_features, constructed_img, content_model):
    with tf.GradientTape() as gradient_tape:
        loss = calculate_content_loss(content_features, constructed_img, content_model)
    return gradient_tape.gradient(loss, constructed_img)

def calculate_style_gradient(style_features, constructed_img, style_model):
    with tf.GradientTape() as gradient_tape:
        loss = calculate_style_loss(style_features, constructed_img, style_model)
    return gradient_tape.gradient(loss, constructed_img)

content_img = load_image("./content1.jpg")
style_img = load_image("./style_1_orig.jpg")

#initial whitenoise img
constructed_img = np.random.normal(size=(height, width, 3))
constructed_img -= constructed_img.min()
constructed_img /= constructed_img.max()/255.0
constructed_img = np.around(constructed_img).astype("float32")

content_layers = ['block4_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

#add dimension for tensorflow
content_img =  np.expand_dims(content_img, axis=0)
style_img = np.expand_dims(style_img, axis=0)
constructed_img = np.expand_dims(constructed_img, axis=0)

content_img = preprocess_image(content_img)
style_img = preprocess_image(style_img)
constructed_img = preprocess_image(constructed_img)

vgg = vgg19.VGG19(include_top=False, weights='imagenet')

style_outputs = []
for layer_name in style_layers:
    style_outputs.append(vgg.get_layer(layer_name).output)

content_outputs = []
for layer_name in content_layers:
    content_outputs.append(vgg.get_layer(layer_name).output)

content_model = Model(inputs=vgg.inputs, outputs=content_outputs)
style_model = Model(inputs=vgg.inputs, outputs=style_outputs)

#content features
content_features = content_model(content_img)

#style features
style_features = style_model(style_img)

#use tfe.Variable so optimizer can update image
constructed_img = tfe.Variable(style_img, dtype=tf.float32)
opt = tf.train.AdamOptimizer(learning_rate=1.0)

for i in range(900):
    features_tup = (content_features, style_features)
    models_tup = (content_model, style_model)
    gradients = calculate_complete_gradient(features_tup, models_tup, constructed_img)
    opt.apply_gradients([(gradients, constructed_img)])

new_output_img = np.squeeze(constructed_img.numpy(), axis=0)
x = new_output_img.copy()
x = deprocess(x)

imageio.imwrite("./test_output_img3.jpg", x)








#num_channels = int(features.shape[3])
#matrix = tf.reshape(features, shape=[-1, num_channels])
#gram = tf.matmul(tf.transpose(matrix), matrix)
