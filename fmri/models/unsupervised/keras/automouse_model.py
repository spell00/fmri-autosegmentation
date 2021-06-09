# Original source: https://github.com/TLKline/AutoTKV_MouseMRI
# Libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, add
from tensorflow.keras.layers import Dropout, BatchNormalization
from fmri.models.unsupervised.keras.metrics import dice_coef, dice_coef_loss
from tensorflow.keras.optimizers import Adam


def automouseTKV_model(img_rows, img_cols, wd):
    inputs = Input((img_rows, img_cols, 1))

    conv1 = Conv2D(32, (7, 7), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.5)(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Conv2D(32, (7, 7), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (5, 5), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.5)(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = Conv2D(64, (5, 5), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.5)(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.5)(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.5)(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Conv2D(1028, (3, 3), activation='relu', padding='same')(pool5)
    conv6 = Dropout(0.5)(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = add([UpSampling2D(size=(2, 2))(conv6), conv5])
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Dropout(0.5)(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = add([UpSampling2D(size=(2, 2))(conv7), conv4])
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Dropout(0.5)(conv8)
    conv8 = BatchNormalization(axis=1)(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = add([UpSampling2D(size=(2, 2))(conv8), conv3])
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Dropout(0.5)(conv9)
    conv9 = BatchNormalization(axis=1)(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    up10 = add([UpSampling2D(size=(2, 2))(conv9), conv2])
    conv10 = Conv2D(64, (5, 5), activation='relu', padding='same')(up10)
    conv10 = Dropout(0.5)(conv10)
    conv10 = BatchNormalization(axis=1)(conv10)
    conv10 = Conv2D(32, (5, 5), activation='relu', padding='same')(conv10)

    up11 = add([UpSampling2D(size=(2, 2))(conv10), conv1])
    conv11 = Conv2D(32, (7, 7), activation='relu', padding='same')(up11)
    conv11 = Dropout(0.5)(conv11)
    conv11 = BatchNormalization(axis=1)(conv11)
    conv11 = Conv2D(32, (7, 7), activation='relu', padding='same')(conv11)

    conv12 = Conv2D(1, (1, 1), activation='sigmoid')(conv11)

    model = Model(inputs, conv12)

    return model