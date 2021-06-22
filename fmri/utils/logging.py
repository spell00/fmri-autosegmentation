from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import numpy as np


class TensorboardLogging:

    def __init__(self, hparams_filepath, params):
        self.params = params
        self.hparams_filepath = hparams_filepath
        HP_LR = hp.HParam('lr', hp.RealInterval(1e-6, 1e-3))
        HP_WD = hp.HParam('wd', hp.RealInterval(1e-8, 1e-3))
        HP_L1 = hp.HParam('l1', hp.RealInterval(0., 1e-3))
        HP_N_RES = hp.HParam('n_res', hp.IntInterval(1, 500))
        HP_N_RES_CHANNEL = hp.HParam('n_res_channel', hp.IntInterval(1, 1024))
        HP_N_CHANNEL = hp.HParam('n_channel', hp.IntInterval(1, 1024))
        HP_Z_DIM = hp.HParam('z_dim', hp.IntInterval(1, 1024))
        HP_SCALE = hp.HParam('scale', hp.RealInterval(0., 0.05))
        HPARAMS = [HP_LR, HP_WD, HP_L1, HP_N_RES, HP_N_RES_CHANNEL, HP_N_CHANNEL, HP_Z_DIM, HP_SCALE]
        with tf.summary.create_file_writer(hparams_filepath).as_default():
            hp.hparams_config(
                hparams=HPARAMS,
                metrics=[
                    hp.Metric('train_accuracy', display_name='Train Accuracy'),
                    hp.Metric('valid_accuracy', display_name='Valid Accuracy'),

                    hp.Metric('train_loss', display_name='Train Loss'),
                    hp.Metric('valid_loss', display_name='Valid Loss'),

                    hp.Metric('train_specificity', display_name='Train Specificity'),
                    hp.Metric('valid_specificity', display_name='Valid Specificity'),

                    hp.Metric('train_sensitivity', display_name='Train Sensitivity'),
                    hp.Metric('valid_sensitivity', display_name='Valid Sensitivity'),

                    hp.Metric('train_jaccard', display_name='Train Jaccard'),
                    hp.Metric('valid_jaccard', display_name='Valid Jaccard'),

                    hp.Metric('train_dice', display_name='Train Dice'),
                    hp.Metric('valid_dice', display_name='Valid Dice'),

                    hp.Metric('train_recon', display_name='Train Recon'),
                    hp.Metric('valid_recon', display_name='Valid Recon'),

                    hp.Metric('train_kld', display_name='Train KLD'),
                    hp.Metric('valid_kld', display_name='Valid KLD'),

                ],
            )

    def logging(self, traces):
        lr = self.params['learning_rate']
        wd = self.params['weight_decay']
        l1 = self.params['l1']
        n_res_channel = self.params['n_res_channel']
        n_channel = self.params['n_channel']
        n_res = self.params['n_res']
        z_dim = self.params['z_dim']
        scale = self.params['scale']
        best_epoch = np.argmin(traces['losses']['train'])
        with tf.summary.create_file_writer(self.hparams_filepath).as_default():
            hp.hparams({
                'lr': lr,
                'wd': wd,
                'l1': l1,
                'n_res': n_res,
                'z_dim': z_dim,
                'n_res_channel': n_res_channel,
                'n_channel': n_channel,
                'scale': scale,
            })  # record the values used in this trial
            tf.summary.scalar('train_accuracy', traces['acc']['train'][best_epoch], step=1)
            tf.summary.scalar('valid_accuracy',  traces['acc']['valid'][best_epoch], step=1)

            tf.summary.scalar('train_loss', traces['losses']['train'][best_epoch], step=1)
            tf.summary.scalar('valid_loss',  traces['losses']['valid'][best_epoch], step=1)

            tf.summary.scalar('train_specificity', traces['specificity']['train'][best_epoch], step=1)
            tf.summary.scalar('valid_specificity',  traces['specificity']['valid'][best_epoch], step=1)

            tf.summary.scalar('train_sensitivity', traces['sensitivity']['train'][best_epoch], step=1)
            tf.summary.scalar('valid_sensitivity',  traces['sensitivity']['valid'][best_epoch], step=1)

            tf.summary.scalar('train_jaccard', traces['jaccard']['train'][best_epoch], step=1)
            tf.summary.scalar('valid_jaccard',  traces['jaccard']['valid'][best_epoch], step=1)

            tf.summary.scalar('train_dice', traces['dice']['train'][best_epoch], step=1)
            tf.summary.scalar('valid_dice',  traces['dice']['valid'][best_epoch], step=1)

            tf.summary.scalar('train_recon', traces['recon']['train'][best_epoch], step=1)
            tf.summary.scalar('valid_recon',  traces['recon']['valid'][best_epoch], step=1)
            try:
                tf.summary.scalar('train_kld', traces['kld']['train'][best_epoch], step=1)
                tf.summary.scalar('valid_kld',  traces['kld']['valid'][best_epoch], step=1)
            except:
                pass
