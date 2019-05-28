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

tf.enable_eager_execution()

def load_image(filename):
    image = Image.open(filename).resize((128, 128))
    return np.float32(image)


def calculate_content_loss(content_features, constructed_img, content_model):
    const_content_features = content_model(constructed_img)

    l_content = 0.5 * np.sum((content_features.numpy() - const_content_features.numpy()) ** 2)
    test_l_content = 0.5 * K.sum(K.square(content_features - const_content_features))


    print("test_l_content", test_l_content)

    return test_l_content

def calculate_style_loss(style_features, constructed_img, style_model):
    const_style_features = style_model(constructed_img)

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
        style_loss += gram_error

    print("style_loss", style_loss)
    return style_loss

def calculate_content_gradient(content_features, constructed_img, content_model):
    with tf.GradientTape() as gradient_tape:
        loss = calculate_content_loss(content_features, constructed_img, content_model)
    return gradient_tape.gradient(loss, constructed_img)

def calculate_style_gradient(style_features, constructed_img, style_model):
    with tf.GradientTape() as gradient_tape:
        loss = calculate_style_loss(style_features, constructed_img, style_model)
    return gradient_tape.gradient(loss, constructed_img)


def calculate_gram_matrix(features):
    squeezed = K.squeeze(features, 0)
    test = K.batch_flatten(K.permute_dimensions(squeezed, (2, 0, 1)))
    #test = K.batch_flatten(K.permute_dimensions(squeezed, (2, 0, 1)))
    other_test = K.dot(test, K.transpose(test))

    num_channels = int(features.shape[3])
    matrix = tf.reshape(features, shape=[-1, num_channels])

    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram

content_img = load_image("./content1.jpg")
style_img = load_image("./style_1_orig.jpg")

#initial whitenoise img
constructed_img = np.random.normal(size=(128, 128, 3))
constructed_img -= constructed_img.min()
constructed_img /= constructed_img.max()/255.0
constructed_img = np.around(constructed_img).astype("float32")

content_layers = ['block4_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

#add dimension for tensorflow
content_img =  np.expand_dims(content_img, axis=0)
style_img = np.expand_dims(style_img, axis=0)
constructed_img = np.expand_dims(constructed_img, axis=0)




vgg = vgg19.VGG19(include_top=False, weights='imagenet')

style_outputs = [vgg.get_layer(name).output for name in style_layers]
content_outputs = [vgg.get_layer(name).output for name in content_layers]
#composite_outputs = style_outputs + content_outputs


content_model = Model(inputs=vgg.inputs, outputs=content_outputs)
style_model = Model(inputs=vgg.inputs, outputs=style_outputs)
#composite_model = Model(inputs=vgg.inputs, outputs=composite_outputs)

#content features
content_features = content_model(content_img)
#constructed_content_outputs = content_model(constructed_img)

#style features
style_features = style_model(style_img)

#use tfe.Variable so optimizer can update image
constructed_img = tfe.Variable(constructed_img, dtype=tf.float32)
opt = tf.train.AdamOptimizer(learning_rate=15.0)

#calculate_style_loss(style_features, constructed_img, style_model)


for i in range(300):
    gradients = calculate_style_gradient(style_features, constructed_img, style_model)
    opt.apply_gradients([(gradients, constructed_img)])

new_output_img = np.squeeze(constructed_img.numpy(), axis=0)
imageio.imwrite("./new_output_img.jpg", new_output_img)




#output_content = content_model(content_tensor)
#output_style = style_model(style_tensor)

#print("output content", output_content)
#style_features = [style_layer for style_layer in output_style]
#content_feature = [content_layer for content_layer in output_content]
#print("style_features", content_features)
#print("output style", output_style)

#stack_images = np.concatenate([style_tensor, content_tensor], axis=0)
#model_outputs = composite_model(stack_images)
#style_features = [style_layer[0] for style_layer in model_outputs[:num_style_layers]]
#content_features = [content_layer[1] for content_layer in model_outputs[num_style_layers:]]




#content_outputs = composite_model(content_tensor)

#print("content_outputs", content_outputs)

#print("constructed_img", content_tensor)
#print("dtypes", content_img.dtype, style_img.dtype, constructed_img.dtype)


#imageio.imwrite("./test_output.jpg", content_img)

#imageio.imwrite("./style1_cropped.jpg", style_img)






#w1 = tfe.Variable(2.0)
#w2 = tfe.Variable(3.0)

#def weighted_sum(x1, x2):
#    return w1 * x1 + w2 * x2

#s = weighted_sum(5., 7.)
#print(s) # 31

#with tf.GradientTape() as tape:
#    s = weighted_sum(5., 7.)

#[w1_grad] = tape.gradient(s, [w1])
#print(w1_grad.numpy()) # 5.0 = gradient of s with regards to w1 = x1

#[test_w1_grad] = K.gradients(s, [w1])

#with tf.Session() as sess:
#    print(sess.run(test_w1_grad))
