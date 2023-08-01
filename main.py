import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import numpy as np
from sklearn.metrics import classification_report
import torchvision

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism

print_config()


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import cv2
import matplotlib.pyplot as plt
import gc
from keras.engine.base_layer import Layer
from keras import layers
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import roc_curve, auc



image_size = 256
emb_dim = 128
epochs = 300
t = 0.1
neg_num = 10
batch_size = 10
map_size = 8
test_mode = True



directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
md5 = "0bc7306e7427e00ad1c5526a6677552d"

compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
data_dir = os.path.join(root_dir, "MedNIST")
if not os.path.exists(data_dir):
    download_and_extract(resource, compressed_file, root_dir, md5)

set_determinism(seed=0)
keras.utils.set_random_seed(0)
tf.random.set_seed(0)



train_normal = np.load('data/train_normal.npy')
test_normal = np.load('data/test_normal.npy')
abnormal = np.load('data/abnormal.npy')


def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]

    return keras.layers.Lambda(func)


num_ind = [[i, i + batch_size] if i < batch_size else [i, i - batch_size] for i in range(2 * batch_size)]
@keras.utils.register_keras_serializable(package='mypackage4', name='contrast_loss')
def contrast_loss(t):

    def loss(y_true, y_pred):
        o = tf.reshape(crop(1, 0, 1)(y_pred), (batch_size, map_size, map_size//2, 2*emb_dim))
        o_cor = crop(1, 1, neg_num+1)(y_pred)
        out1 = crop(3, 0, emb_dim)(o)
        out2 = crop(3, emb_dim, 2*emb_dim)(o)
        out1_cor = crop(4, 0, emb_dim)(o_cor)
        out2_cor = crop(4, emb_dim, 2*emb_dim)(o_cor)
        out1 = K.l2_normalize(out1, axis=-1)
        out2 = K.l2_normalize(out2, axis=-1)
        out1_cor = K.l2_normalize(tf.transpose(out1_cor, perm=[0, 2, 3, 1, 4]), axis=-1)
        out2_cor = K.l2_normalize(tf.transpose(out2_cor, perm=[0, 2, 3, 1, 4]), axis=-1)

        out1_neg = tf.reshape(tf.matmul(out2_cor, tf.reshape(out1, (batch_size, map_size, map_size//2, emb_dim, 1))), (batch_size, map_size, map_size//2, neg_num))
        out2_neg = tf.reshape(tf.matmul(out1_cor, tf.reshape(out2, (batch_size, map_size, map_size//2, emb_dim, 1))), (batch_size, map_size, map_size//2, neg_num))
        out1_neg = K.mean(tf.transpose(out1_neg, perm=[0, 3, 1, 2]), axis=[2, 3])
        out2_neg = K.mean(tf.transpose(out2_neg, perm=[0, 3, 1, 2]), axis=[2, 3])
        denum1_neg = K.sum(K.exp(out1_neg / t), axis=1)
        denum2_neg = K.sum(K.exp(out2_neg / t), axis=1)
        # num = K.exp(K.sum(out1 * out2, axis=1) / t)
        comb = K.concatenate((out1, out2), axis=0)
        mat = K.exp(K.mean(tf.transpose(tf.matmul(tf.transpose(comb, perm=[1, 2, 0, 3]), tf.transpose(comb, perm=[1, 2, 3, 0])), perm=[2, 3, 0, 1]), axis=[2, 3]) / t)
        num = tf.gather_nd(mat, num_ind)
        denum = K.sum(mat, axis=1) - np.exp(1/t) - num
        denum1 = denum[0:batch_size] + denum1_neg
        denum2 = denum[batch_size:2*batch_size] + denum2_neg
        # denum1 = denum1_neg
        # denum2 = denum2_neg
        num1 = num[0:batch_size]
        num2 = num[batch_size:2*batch_size]
        l1 = -K.log(num1 / denum1) - K.log(num2 / denum2)
        return l1 / 2.0

    return loss

def feature_extractor(inp_size, t):

    backbone = keras.applications.EfficientNetV2B0(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=inp_size,
    pooling=None,
    include_preprocessing=False,
    )

    # x1 = inp_layer1
    # x2 = inp_layer2
    # for layer in backbone.layers:
    #     x1 = layer(x1)
    #     x2 = layer(x2)

    x1 = backbone.layers[-1].output[:(neg_num+1)*batch_size]
    x2 = backbone.layers[-1].output[(neg_num+1)*batch_size:]

    fin = keras.layers.Dense(emb_dim)
    x1 = fin(x1)
    x2 = fin(x2)

    o = keras.layers.Concatenate(axis=-1)([x1[:batch_size], x2[:batch_size]])
    o_cor = keras.layers.Concatenate(axis=-1)([tf.reshape(x1[batch_size:], (batch_size, neg_num, map_size, map_size//2, emb_dim)), tf.reshape(x2[batch_size:], (batch_size, neg_num, map_size, map_size//2, emb_dim))])
    o = keras.layers.Concatenate(axis=1)([tf.reshape(o, (batch_size, 1, map_size, map_size//2, 2*emb_dim)), o_cor])

    model = keras.models.Model(backbone.inputs, o)
    model.compile(optimizer='adam',
                  loss=contrast_loss(t),
                  )

    return model





def autoencoder(inp_size, t):
    inp1 = keras.layers.Input(shape=inp_size)
    inp2 = keras.layers.Input(shape=inp_size)

    # inp1_cor = keras.layers.Input(shape=inp_size)
    # inp2_cor = keras.layers.Input(shape=inp_size)

    e1 = keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer='glorot_normal')
    b1 = keras.layers.BatchNormalization()
    a1 = keras.layers.Activation('relu')
    e2 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal')
    b2 = keras.layers.BatchNormalization()
    a2 = keras.layers.Activation('relu')

    c1 = keras.layers.Concatenate(axis=-1)

    m1 = keras.layers.MaxPooling2D(pool_size=(2, 2))

    e3 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal')
    b3 = keras.layers.BatchNormalization()
    a3 = keras.layers.Activation('relu')

    ad1 = keras.layers.Add()

    e4 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal')
    b4 = keras.layers.BatchNormalization()
    a4 = keras.layers.Activation('relu')

    c2 = keras.layers.Concatenate(axis=-1)

    m2 = keras.layers.MaxPooling2D(pool_size=(2, 2))


    e5 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal')
    b5 = keras.layers.BatchNormalization()
    a5 = keras.layers.Activation('relu')

    ad2 = keras.layers.Add()

    e6 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal')
    b6 = keras.layers.BatchNormalization()
    a6 = keras.layers.Activation('relu')

    c3 = keras.layers.Concatenate(axis=-1)

    m3 = keras.layers.MaxPooling2D(pool_size=(2, 2))

    e7 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal')
    b7 = keras.layers.BatchNormalization()
    a7 = keras.layers.Activation('relu')

    ad3 = keras.layers.Add()

    e8 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal')
    b8 = keras.layers.BatchNormalization()
    a8 = keras.layers.Activation('relu')

    c4 = keras.layers.Concatenate(axis=-1)

    m4 = keras.layers.AveragePooling2D(pool_size=(2, 2))

    fin = keras.layers.Dense(emb_dim)


    n1 = e1(inp1)
    n1 = b1(n1)
    n1 = a1(n1)

    n2 = e2(n1)
    n2 = b2(n2)
    n2 = a2(n2)

    n3 = c1([n1, n2])
    n3 = m1(n3)

    n4 = e3(n3)
    n4 = b3(n4)
    n4 = a3(n4)

    n4 = ad1([n3, n4])

    n5 = e4(n4)
    n5 = b4(n5)
    n5 = a4(n5)

    n6 = c2([n4, n5])
    n6 = m2(n6)

    n7 = e5(n6)
    n7 = b5(n7)
    n7 = a5(n7)

    n7 = ad2([n6, n7])

    n8 = e6(n7)
    n8 = b6(n8)
    n8 = a6(n8)

    n9 = c3([n7, n8])
    n9 = m3(n9)

    n10 = e7(n9)
    n10 = b7(n10)
    n10 = a7(n10)

    n10 = ad3([n9, n10])

    n11 = e8(n10)
    n11 = b8(n11)
    n11 = a8(n11)

    n12 = c4([n10, n11])
    n12 = m4(n12)

    o1 = fin(n12)

    #################################################

    n1 = e1(inp2)
    n1 = b1(n1)
    n1 = a1(n1)

    n2 = e2(n1)
    n2 = b2(n2)
    n2 = a2(n2)

    n3 = c1([n1, n2])
    n3 = m1(n3)

    n4 = e3(n3)
    n4 = b3(n4)
    n4 = a3(n4)

    n4 = ad1([n3, n4])

    n5 = e4(n4)
    n5 = b4(n5)
    n5 = a4(n5)

    n6 = c2([n4, n5])
    n6 = m2(n6)

    n7 = e5(n6)
    n7 = b5(n7)
    n7 = a5(n7)

    n7 = ad2([n6, n7])

    n8 = e6(n7)
    n8 = b6(n8)
    n8 = a6(n8)

    n9 = c3([n7, n8])
    n9 = m3(n9)

    n10 = e7(n9)
    n10 = b7(n10)
    n10 = a7(n10)

    n10 = ad3([n9, n10])

    n11 = e8(n10)
    n11 = b8(n11)
    n11 = a8(n11)

    n12 = c4([n10, n11])
    n12 = m4(n12)

    o2 = fin(n12)

    ########################################


    o = keras.layers.Concatenate(axis=-1)([o1[:batch_size], o2[:batch_size]])
    o_cor = keras.layers.Concatenate(axis=-1)([tf.reshape(o1[batch_size:], (batch_size, neg_num, map_size, map_size//2, emb_dim)), tf.reshape(o2[batch_size:], (batch_size, neg_num, map_size, map_size//2, emb_dim))])
    o = keras.layers.Concatenate(axis=1)([tf.reshape(o, (batch_size, 1, map_size, map_size//2, 2*emb_dim)), o_cor])

    model = keras.models.Model([inp1, inp2], o)
    model.compile(optimizer='adam',
                  loss=contrast_loss(t),
                  )

    return model


def cutpaste(image, low, high, rotation_prob, blur_prob, paste_right):
    width = np.random.randint(low, high)
    hight = np.random.randint(low, high)

    start_x = np.random.randint(0, image.shape[0] - width)
    start_y = np.random.randint(0, image.shape[1] - hight)

    im_patch = image[start_x:start_x+width, start_y:start_y+hight]

    if np.random.uniform(0, 1) < rotation_prob:
        zx = 1
        zy = 1
        theta = np.random.uniform(0, 360)
        tx = 0
        ty = 0
        im_patch = tf.keras.preprocessing.image.apply_affine_transform(image, theta=theta, tx=tx, ty=ty, shear=0, zx=zx, zy=zy)[start_x:start_x+width, start_y:start_y+hight]

    tar_x = np.random.randint(0, image.shape[0] - width)
    tar_y = np.random.randint(0, image.shape[1]//2 - hight)

    if paste_right:
        dist = np.sqrt((tar_x + width//2 - image.shape[0]//2)**2 + (tar_y + hight//2)**2)
    else:
        dist = np.sqrt((tar_x + width//2 - image.shape[0]//2)**2 + (tar_y + hight//2 - image.shape[1]//2)**2)

    if dist > image_size//2:
        tar_x = np.random.randint(0, image.shape[0] - width)
        if paste_right:
            tar_y = np.random.randint(0, min(max(image.shape[1]//2 - np.abs(tar_x + width//2 - image.shape[1]//2) - hight//2, 1), image.shape[1]//2 - hight))
        else:
            tar_y = np.random.randint(min(np.abs(tar_x + width//2 - image.shape[1]//2), image.shape[1]//2 - hight - 1), image.shape[1]//2 - hight)

    if paste_right:
        tar = image[:, image.shape[1]//2:].copy()
    else:
        tar = image[:, :image.shape[1]//2].copy()
    tar[tar_x:tar_x+width, tar_y:tar_y+hight] = im_patch

    if np.random.uniform(0, 1) < blur_prob:
        tar = gaussian_filter(tar, sigma=3)

    return tar


def test():
    loss_normal = np.zeros((test_normal.shape[0]))
    for step in range(5):
        gc.collect()
        print('step: ' + str(step))
        targets_right = []
        targets_left = []
        for i in range(test_normal.shape[0]):
            zx = 1
            zy = 1
            theta = np.random.uniform(-45, 45)
            tx = 0
            ty = 0
            # rot = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0], [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0], [0, 0, 1]])
            # zoom = np.array([[1/zx, 0, 0], [0, 1/zy, 0], [0, 0, 1]])
            # tran = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
            # aff = tf.matmul(zoom, tf.matmul(tran, rot))
            tar_right = tf.keras.preprocessing.image.apply_affine_transform(test_normal[i], theta=theta, tx=tx, ty=ty, shear=0, zx=zx, zy=zy)[:, image_size//2:]
            tar_left = tf.keras.preprocessing.image.apply_affine_transform(test_normal[i], theta=-theta, tx=tx, ty=ty, shear=0, zx=zx, zy=zy)[:, :image_size//2]
            targets_right.append(tar_right)
            targets_left.append(tar_left)

        targets_right = np.array(targets_right)
        targets_left = np.array(targets_left)

        temp_loss = None

        for b in range(test_normal.shape[0]//batch_size):
            tar_right = targets_right[b*batch_size:(b+1)*batch_size]
            tar_left = targets_left[b*batch_size:(b+1)*batch_size]

            for i in range(neg_num):
                tar_right = np.concatenate((tar_right, targets_right[b*batch_size:(b+1)*batch_size]), axis=0)
                tar_left = np.concatenate((tar_left, targets_left[b*batch_size:(b+1)*batch_size]), axis=0)

    # model.fit([targets_right, targets_left], [targets_right, targets_left], verbose=1, batch_size=batch_size)
            out = model.predict(np.concatenate((tar_right, np.flip(tar_left, axis=-2)), axis=0), batch_size=2*batch_size*(neg_num+1))[:, 0, :, :, :]

    # plt.imshow(targets_right[0].reshape(image_size, image_size//2), cmap='gray')
    # plt.show()
    # plt.imshow(targets_left[0].reshape(image_size, image_size//2), cmap='gray')
    # plt.show()
    # plt.imshow(tar2_right[0].reshape(image_size, image_size//2), cmap='gray')
    # plt.show()
    # plt.imshow(tar2_left[0].reshape(image_size, image_size//2), cmap='gray')
    # plt.show()
            out1 = out[:, :, :, :emb_dim]
            out1_norm = np.linalg.norm(out1, axis=-1).reshape((out1.shape[0], out1.shape[1], out1.shape[2], 1))
            out1_norm = np.concatenate(tuple([out1_norm for _ in range(emb_dim)]), axis=-1)
            out1 = out1 / out1_norm

            out2 = out[:, :, :, emb_dim:2*emb_dim]
            out2_norm = np.linalg.norm(out2, axis=-1).reshape((out2.shape[0], out2.shape[1], out2.shape[2], 1))
            out2_norm = np.concatenate(tuple([out2_norm for _ in range(emb_dim)]), axis=-1)
            out2 = out2 / out2_norm

            if temp_loss is None:
                temp_loss = -np.mean(np.sum(out1*out2, axis=-1), axis=(1, 2))
            else:
                temp_loss = np.concatenate((temp_loss, -np.mean(np.sum(out1*out2, axis=-1), axis=(1, 2))), axis=0)

        loss_normal += temp_loss

    loss_abnormal = np.zeros((abnormal.shape[0]))
    for step in range(5):
        gc.collect()
        print('step: ' + str(step))
        targets_right = []
        targets_left = []
        for i in range(abnormal.shape[0]):
            zx = 1
            zy = 1
            theta = np.random.uniform(0, 360)
            tx = 0
            ty = 0
        # rot = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0], [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0], [0, 0, 1]])
        # zoom = np.array([[1/zx, 0, 0], [0, 1/zy, 0], [0, 0, 1]])
        # tran = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        # aff = tf.matmul(zoom, tf.matmul(tran, rot))
            tar_right = tf.keras.preprocessing.image.apply_affine_transform(abnormal[i], theta=theta, tx=tx, ty=ty, shear=0, zx=zx, zy=zy)[:, image_size//2:]
            tar_left = tf.keras.preprocessing.image.apply_affine_transform(abnormal[i], theta=-theta, tx=tx, ty=ty, shear=0, zx=zx, zy=zy)[:, :image_size//2]
            targets_right.append(tar_right)
            targets_left.append(tar_left)

        targets_right = np.array(targets_right)
        targets_left = np.array(targets_left)

        temp_loss = None

        for b in range(abnormal.shape[0]//batch_size):
            tar_right = targets_right[b*batch_size:(b+1)*batch_size]
            tar_left = targets_left[b*batch_size:(b+1)*batch_size]

            for i in range(neg_num):
                tar_right = np.concatenate((tar_right, targets_right[b*batch_size:(b+1)*batch_size]), axis=0)
                tar_left = np.concatenate((tar_left, targets_left[b*batch_size:(b+1)*batch_size]), axis=0)

    # model.fit([targets_right, targets_left], [targets_right, targets_left], verbose=1, batch_size=batch_size)
            out = model.predict(np.concatenate((tar_right, np.flip(tar_left, axis=-2)), axis=0), batch_size=2*batch_size*(neg_num+1))[:, 0, :, :, :]
    # plt.imshow(targets_right[0].reshape(image_size, image_size//2), cmap='gray')
    # plt.show()
    # plt.imshow(targets_left[0].reshape(image_size, image_size//2), cmap='gray')
    # plt.show()
    # plt.imshow(tar2_right[0].reshape(image_size, image_size//2), cmap='gray')
    # plt.show()
    # plt.imshow(tar2_left[0].reshape(image_size, image_size//2), cmap='gray')
    # plt.show()
            out1 = out[:, :, :, :emb_dim]
            out1_norm = np.linalg.norm(out1, axis=-1).reshape((out1.shape[0], out1.shape[1], out1.shape[2], 1))
            out1_norm = np.concatenate(tuple([out1_norm for _ in range(emb_dim)]), axis=-1)
            out1 = out1 / out1_norm

            out2 = out[:, :, :, emb_dim:2*emb_dim]
            out2_norm = np.linalg.norm(out2, axis=-1).reshape((out2.shape[0], out2.shape[1], out2.shape[2], 1))
            out2_norm = np.concatenate(tuple([out2_norm for _ in range(emb_dim)]), axis=-1)
            out2 = out2 / out2_norm

            if temp_loss is None:
                temp_loss = -np.mean(np.sum(out1*out2, axis=-1), axis=(1, 2))
            else:
                temp_loss = np.concatenate((temp_loss, -np.mean(np.sum(out1*out2, axis=-1), axis=(1, 2))), axis=0)

        loss_abnormal += temp_loss

    return loss_normal, loss_abnormal


model = feature_extractor((image_size, image_size//2, 3), t)
custom_objects = {"loss": contrast_loss(0.1)}
with keras.saving.custom_object_scope(custom_objects):
    model = keras.models.load_model('./newmodel.keras')



for step in range(epochs):
    print('step: ' + str(step))
    if step>=1:
        model.save('./newmodel.keras')
    for batch_num in range(train_normal.shape[0]//batch_size):
        train_batch = train_normal[np.random.choice(train_normal.shape[0], batch_size)]
        targets_right = []
        targets_left = []
        targets_right_corrupt = []
        targets_left_corrupt = []
        for i in range(train_batch.shape[0]):
            # print(i)
            zx = 1
            zy = 1
            theta = np.random.uniform(-45, 45)
            tx = 0
            ty = 0
        # rot = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0], [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0], [0, 0, 1]])
        # zoom = np.array([[1/zx, 0, 0], [0, 1/zy, 0], [0, 0, 1]])
        # tran = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        # aff = tf.matmul(zoom, tf.matmul(tran, rot))
            tar_right = tf.keras.preprocessing.image.apply_affine_transform(train_batch[i], theta=theta, tx=tx, ty=ty, shear=0, zx=zx, zy=zy)[:, image_size//2:]
            tar_left = tf.keras.preprocessing.image.apply_affine_transform(train_batch[i], theta=-theta, tx=tx, ty=ty, shear=0, zx=zx, zy=zy)[:, :image_size//2]
            targets_right.append(tar_right)
            targets_left.append(tar_left)

            temp_left = []
            temp_right = []
            new_im = np.concatenate((tar_left, tar_right), axis=1)
            for __ in range(neg_num):
                corrupt_image_right = cutpaste(new_im, 15, 40, 0.5, 0.5, True)
                corrupt_image_left = cutpaste(new_im, 15, 40, 0.5, 0.5, False)
                # plt.imshow(np.concatenate((corrupt_image_left, corrupt_image_right), axis=1).reshape(image_size, image_size), cmap='gray')
                # plt.show()
                temp_right.append(corrupt_image_right)
                temp_left.append(corrupt_image_left)

            targets_right_corrupt.append(temp_right)
            targets_left_corrupt.append(temp_left)

            # tar_right = tf.keras.preprocessing.image.apply_affine_transform(train_batch[i], theta=theta+45, tx=tx, ty=ty, shear=0, zx=zx, zy=zy)[:, image_size//2:]
            # tar_left = tf.keras.preprocessing.image.apply_affine_transform(train_batch[i], theta=-theta-45, tx=tx, ty=ty, shear=0, zx=zx, zy=zy)[:, :image_size//2]
            # targets_right.append(tar_right)
            # targets_left.append(tar_left)

        targets_right = np.array(targets_right)
        targets_left = np.array(targets_left)
        targets_right_corrupt = np.array(targets_right_corrupt)
        targets_left_corrupt = np.array(targets_left_corrupt)

        right_inp = np.concatenate((targets_right, targets_right_corrupt.reshape((batch_size*neg_num, image_size, image_size//2, 3))), axis=0)
        left_inp = np.flip(np.concatenate((targets_left, targets_left_corrupt.reshape((batch_size*neg_num, image_size, image_size//2, 3))), axis=0), axis=-2)

        del targets_right
        del targets_left
        del targets_right_corrupt
        del targets_left_corrupt

        gc.collect()
        # plt.imshow(left_inp[0].reshape(image_size, image_size//2), cmap='gray')
        # plt.show()
    # plt.imshow(targets_left[0].reshape(image_size, image_size//2), cmap='gray')
    # plt.show()

    # plt.imshow(tar2_right[0].reshape(image_size, image_size//2), cmap='gray')
    # plt.show()
    # plt.imshow(tar2_left[0].reshape(image_size, image_size//2), cmap='gray')
    # plt.show()
        model.fit(np.concatenate((right_inp, left_inp), axis=0), np.concatenate((right_inp, left_inp), axis=0), verbose=1, batch_size=2*batch_size*(neg_num+1), shuffle=False)

        # model.fit([targets_right[batch_size:], targets_left[batch_size:]], [targets_left[batch_size:], targets_right[batch_size:]], verbose=1, batch_size=batch_size)

        del right_inp
        del left_inp

        gc.collect()

    if test_mode:
        if step >= 1 and step % 10 == 0:
            loss_normal, loss_abnormal = test()

            diff = np.concatenate((loss_normal, loss_abnormal))
            test_labels = np.array([0 if i < loss_normal.shape[0] else 1 for i in range(loss_normal.shape[0] + loss_abnormal.shape[0])])
            print(np.mean(loss_normal))
            print(np.mean(loss_abnormal))

            fpr, tpr, thresholds = roc_curve(test_labels, diff, pos_label=1)
            AUC = auc(fpr, tpr)
            print('AUC: ' + str(AUC))

    # print('step: ' + str(step))
    # targets_right = []
    # targets_left = []
    # for i in range(train_normal.shape[0]):
    #     zx = 1
    #     zy = 1
    #     theta = 0
    #     tx = 0
    #     ty = 0
    #     # rot = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0], [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0], [0, 0, 1]])
    #     # zoom = np.array([[1/zx, 0, 0], [0, 1/zy, 0], [0, 0, 1]])
    #     # tran = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    #     # aff = tf.matmul(zoom, tf.matmul(tran, rot))
    #     tar_right = tf.keras.preprocessing.image.apply_affine_transform(train_normal[i], theta=theta, tx=tx, ty=ty, shear=0, zx=zx, zy=zy)[:, image_size//2:]
    #     tar_left = tf.keras.preprocessing.image.apply_affine_transform(train_normal[i], theta=-theta, tx=tx, ty=ty, shear=0, zx=zx, zy=zy)[:, :image_size//2]
    #     targets_right.append(tar_right)
    #     targets_left.append(tar_left)

    # targets_right = np.array(targets_right)
    # targets_left = np.array(targets_left)
    # model.fit([targets_right, targets_left], [targets_left, targets_right], verbose=1, batch_size=batch_size)
