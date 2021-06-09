import os
import nibabel as nib
from datetime import datetime
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import nibabel as nib

from fmri.models.unsupervised.keras import automouse_model as mouse_model

from numpy.random import seed
import tensorflow as tf

from skopt.space import Real
from skopt import gp_minimize
from tensorboard.plugins.hparams import api as hp
from fmri.utils.dataset import resize_data
from sklearn.metrics import confusion_matrix
from PIL import Image
from fmri.models.unsupervised.keras.metrics import jaccard_distance_loss, dice_coef_loss, dice_coef

seed(42)
tf.random.set_seed(42)

os.makedirs(f'logs/keras_ae_3d/hparam_tuning', exist_ok=True)

HP_LR = hp.HParam('lr', hp.RealInterval(1e-5, 1e-3))
HP_WD = hp.HParam('wd', hp.RealInterval(1e-8, 1e-3))

HPARAMS = [HP_LR, HP_WD]
with tf.summary.create_file_writer(f'logs/keras_ae_3d/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=HPARAMS,
        metrics=[
            hp.Metric('train_accuracy', display_name='Train Accuracy'),
            hp.Metric('valid_accuracy', display_name='Valid Accuracy'),
            hp.Metric('test_accuracy', display_name='Test Accuracy'),
            hp.Metric('train_loss', display_name='Train Loss'),
            hp.Metric('valid_loss', display_name='Valid Loss'),
            hp.Metric('test_loss', display_name='Test Loss')
        ],
    )

from scipy.ndimage import zoom

def get_fmri(path, binarize=False):
    all_paths = os.listdir(path)
    xs = []
    for p in all_paths:
        x = nib.load(f'{path}/{p}').dataobj
        x = np.array(x)
        x = resize_data(x, [256, 256, x.shape[-1]])
        # x = np.sum(x, -1)
        if binarize:
            x[x != 0] = 1
        for im in range(x.shape[2]):
            xs += [x[:, :, im]]
    xs = np.stack(xs)
    return xs


class Train:
    def __init__(
            self,
            images_dir,
            labels_dir,
            criterion='binary_crossentropy',
            get_data_function=get_fmri,
            n_channels=1,
            save_train_models=True,
            verbose=0
    ):
        self.verbose = verbose
        self.save_train_models = save_train_models
        self.labels = get_data_function(labels_dir, binarize=True)
        self.data = get_data_function(images_dir)

        self.nb_classes = 2
        self.input_shape = [self.data[0].shape[1], n_channels]
        self.criterion = criterion
        self.step = 0

    def compute_confusion_matrix(self, y_test, y_classes):
        ys = np.round(y_test.reshape([y_test.shape[0], -1]), 0)
        y_classes = y_classes.reshape([y_classes.shape[0], -1])
        tp = np.sum([[1 if true == 1 and pred == 1 else 0 for true, pred in zip(t, c)] for t, c in zip(ys, y_classes)])
        tn = np.sum([[1 if true == 0 and pred == 0 else 0 for true, pred in zip(t, c)] for t, c in zip(ys, y_classes)])
        fp = np.sum([[1 if true == 0 and pred == 1 else 0 for true, pred in zip(t, c)] for t, c in zip(ys, y_classes)])
        fn = np.sum([[1 if true == 1 and pred == 0 else 0 for true, pred in zip(t, c)] for t, c in zip(ys, y_classes)])

        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        jaccard = jaccard_distance_loss(y_true=ys, y_pred=np.array(y_classes, dtype=np.double))
        dice = dice_coef_loss(y_true=ys, y_pred=np.array(y_classes, dtype=np.double))
        if self.verbose:
            print(f"specificity: {specificity}")
            print(f"sensitivity: {sensitivity}")
            print(f"jaccard: {jaccard}")
            print(f"dice: {dice}")

        return sensitivity, specificity, jaccard, dice

    def train(self, params):
        lr = params[0]
        wd = params[1]
        path = f'{self.criterion}/' \
               f'{"{:.8f}".format(lr)}/' \
               f'{"{:.8f}".format(wd)}/' \
               f'{str(datetime.now().strftime("%m%d%Y-%H%M%S"))}'
        filepath = f'saved_models/keras/{path}'
        log_filepath = f"logs/keras_ae_3d/{path}"
        hparams_filepath = f"logs/keras_ae_3d/hparam_tuning/{path}"
        os.makedirs(filepath, exist_ok=True)
        os.makedirs(log_filepath, exist_ok=True)
        # 5 Fold-CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []
        self.model_name = '/'.join([
            f'keras_ae',
            f'dice',
            f'adam',
            f'{lr}',
            f'{wd}',
        ])

        for i, (train_indices, test_indices) in enumerate(skf.split(list(range(len(self.labels))), np.array([1 for _ in range(len(self.labels))]))):
            # Just plot the first iteration, it will already be crowded if doing > 100 optimization iterations
            if i > 0:
                break
            if self.verbose:
                print(f"CV: {i}")
            maximum = np.max([
                np.max(self.data),
            ])
            self.data = self.data / maximum

            x_train = self.data[train_indices]
            y_train = self.labels[train_indices]
            x_test = self.data[test_indices]
            y_test = self.labels[test_indices]
            # x_train_conv = np.reshape(x_train, (x_train.shape[0], x_train.shape[2], x_train.shape[3], 1))
            # x_test_conv = np.reshape(x_test, (x_test.shape[0], x_test.shape[2], x_test.shape[3], 1))
            # y_train_conv = np.reshape(y_train, (y_train.shape[0], y_train.shape[2], y_train.shape[3], 1))
            # y_test_conv = np.reshape(y_test, (y_test.shape[0], y_test.shape[2], y_test.shape[3], 1))

            model = mouse_model.automouseTKV_model(self.input_shape[0], self.input_shape[0], wd=wd)
            model.compile(optimizer=Adam(lr=lr, decay=wd), loss="categorical_crossentropy", metrics=[dice_coef, 'accuracy'])

            model.summary()

            callbacks = []
            callbacks += [keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=10,
                verbose=self.verbose,
                min_lr=1e-12,
                min_delta=0.0001,
                cooldown=0,
                mode='min'
            )]
            if i == 0:
                callbacks += [keras.callbacks.TensorBoard(
                    log_dir=log_filepath,
                    histogram_freq=1,
                    write_graph=True,
                    write_images=False,
                    update_freq="epoch",
                    profile_batch=2,
                    embeddings_freq=0,
                    embeddings_metadata=None,
                )]
            if self.save_train_models:
                callbacks += [tf.keras.callbacks.ModelCheckpoint(
                    filepath,
                    monitor="val_loss",
                    verbose=self.verbose,
                    save_best_only=True,
                    save_weights_only=False,
                    mode="auto",
                    save_freq="epoch",
                    options=None,
                )]
            callbacks += [keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=20,
                verbose=self.verbose,
                mode='auto'
            )]
            # y_integers = np.argmax(y_train_conv.view(y_train_conv.shape[0], -1), axis=1)
            # class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
            # d_class_weights = dict(enumerate(class_weights))
            history = model.fit(
                x=x_train,
                y=y_train,
                batch_size=2,
                verbose=self.verbose,
                epochs=1000,
                validation_split=0.2,
                # class_weight=d_class_weights,
                callbacks=callbacks
            )

            model = tf.keras.models.load_model(filepath, custom_objects={
                'dice_coef': dice_coef,
                # 'dice_coef_loss': dice_coef_loss
            })
            model.compile(optimizer=Adam(lr=lr, decay=wd), loss="categorical_crossentropy", metrics=[dice_coef, 'accuracy'])
            test_loss, test_dice, test_acc = model.evaluate(x_test, y_test, verbose=self.verbose)
            y_classes = model.predict(x_test)
            y_classes_bin = np.round(y_classes, 0)

            os.makedirs(f'views/{self.model_name}/nifti', exist_ok=True)
            os.makedirs(f'views/{self.model_name}/jpeg', exist_ok=True)
            for j, (target, img, recon, bin_recon) in enumerate(zip(y_test, x_test, y_classes, y_classes_bin)):
                if j == 10:
                    break
                target_arr = Image.fromarray(target * 255)
                img_arr = Image.fromarray(img * 255)
                recon_arr = Image.fromarray(recon[:, :, -1] * 255)
                bin_recon_arr = Image.fromarray(bin_recon[:, :, -1] * 255)
                target_arr.convert('LA').save(f'views/{self.model_name}/jpeg/target_valid_{j}.png')
                img_arr.convert('LA').save(f'views/{self.model_name}/jpeg/img_valid_{j}.png')
                recon_arr.convert('LA').save(f'views/{self.model_name}/jpeg/reconstruct_valid_{j}.png')
                bin_recon_arr.convert('LA').save(f'views/{self.model_name}/jpeg/bin_recon_valid_{j}.png')

                target = nib.Nifti1Image(target, np.eye(4))
                img = nib.Nifti1Image(img, np.eye(4))
                recon = nib.Nifti1Image(recon, np.eye(4))
                bin_recon = nib.Nifti1Image(bin_recon, np.eye(4))

                target.to_filename(filename=f'views/{self.model_name}/nifti/target_valid_{j}.nii.gz')
                img.to_filename(filename=f'views/{self.model_name}/nifti/image_valid_{j}.nii.gz')
                recon.to_filename(filename=f'views/{self.model_name}/nifti/reconstruct_valid_{j}.nii.gz')
                bin_recon.to_filename(filename=f'views/{self.model_name}/nifti/bin_reconstruct_valid_{j}.nii.gz')
                del target, img, recon, bin_recon, target_arr, img_arr, recon_arr, bin_recon_arr

            self.compute_confusion_matrix(y_test, y_classes)
            train_acc = history.history['accuracy']
            valid_acc = history.history['val_accuracy']
            train_loss = history.history['loss']
            valid_loss = history.history['val_loss']

            best_epoch = np.argmax(valid_loss)

            train_losses.append(train_loss[best_epoch])
            train_accuracies.append(train_acc[best_epoch])
            valid_losses.append(valid_loss[best_epoch])
            valid_accuracies.append(valid_acc[best_epoch])
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            del test_loss, test_dice, test_acc, history, model

            # TODO Delete this piece of code when memory leak is found. It forces the release of all GPU memory
            from numba import cuda
            device = cuda.get_current_device()
            device.reset()
        # TODO Return the best loss to enable Bayesian Optimisation
        self.step += 1
        with tf.summary.create_file_writer(hparams_filepath).as_default():
            hp.hparams({
                'lr': lr,
                'wd': wd
            })  # record the values used in this trial
            tf.summary.scalar('train_accuracy', np.mean(train_accuracies), step=1)
            tf.summary.scalar('valid_accuracy', np.mean(valid_accuracies), step=1)
            tf.summary.scalar('test_accuracy', np.mean(test_accuracies), step=1)
            tf.summary.scalar('train_loss', np.mean(train_losses), step=1)
            tf.summary.scalar('valid_loss', np.mean(valid_loss), step=1)
            tf.summary.scalar('test_loss', np.mean(test_losses), step=1)
        del model, y_classes, y_classes_bin
        return np.mean(test_losses)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str, default="data\\canis_intensities.csv",
                        help="Path to intensities csv file")
    parser.add_argument("--labels_path", type=str, default="data\\canis_labels.csv",
                        help="Path to labels csv file")
    parser.add_argument("--verbose", type=str, default=1)
    args = parser.parse_args()
    train = Train(args.images_path, args.labels_path, verbose=args.verbose)
    space = [
        Real(1e-5, 1e-3, "log-uniform", name='lr'),
        Real(1e-8, 1e-3, "log-uniform", name='wd'),
    ]

    test_mean = gp_minimize(train.train, space, n_calls=100, random_state=42)
    print(test_mean)
