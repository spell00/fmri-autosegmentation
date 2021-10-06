# Original source: https://github.com/TLKline/AutoTKV_MouseMRI
# Libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, add
from tensorflow.keras.layers import Dropout, BatchNormalization
from fmri.models.unsupervised.keras.metrics import dice_coef, dice_coef_loss
from tensorflow.keras.optimizers import Adam
import torch
from torch import nn


def automouseTKV_model(img_rows, img_cols):
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


class AutomouseTKVModel(nn.Module):
    def __init__(self, **kwargs):
        super(AutomouseTKVModel, self).__init__(**kwargs)

        # inputs = Input((img_rows, img_cols, 1))
        # (W - K + 2P) / S + 1
        # (256 - 7 +
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (7, 7), padding=3),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (7, 7), padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5), padding=2),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 1024, (3, 3), padding=0),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, (3, 3), padding=0),
            nn.ReLU(),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, (3, 3), padding=1),
            nn.ReLU(),
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, (3, 3), padding=1),
            nn.ReLU(),
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, (3, 3), padding=1),
            nn.ReLU(),
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(64, 64, (5, 5), padding=2),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, (5, 5), padding=2),
            nn.ReLU(),
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(32, 32, (7, 7), padding=3),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, (7, 7), padding=3),
            nn.ReLU(),
        )

        self.conv12 = nn.Conv2d(1, 1, (1, 1))

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        up7 = torch.add(nn.Upsample(scale_factor=2)(conv6), conv5)
        conv7 = self.conv7(up7)

        up8 = torch.add(nn.Upsample(scale_factor=2)(conv7), conv4)
        conv8 = self.conv8(up8)

        up9 = torch.add(nn.Upsample(scale_factor=2)(conv8), conv3)
        conv9 = self.conv9(up9)

        up10 = torch.add(nn.Upsample(scale_factor=2)(conv9), conv2)
        conv10 = self.conv10(up10)

        up11 = torch.add(nn.Upsample(scale_factor=2)(conv10), conv1)
        conv11 = self.conv11(up11)

        conv12 = self.conv12(nn.Upsample(scale_factor=2)(conv11))

        return torch.sigmoid(conv12), torch.Tensor([0.])

    def random_init(self, init_func=torch.nn.init.xavier_uniform_):
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


