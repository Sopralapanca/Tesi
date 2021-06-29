import yaml

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import History, EarlyStopping
import tensorflow as tf
import sys

import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from telemanom.utility import create_lstm_model, create_esn_model

# suppress tensorflow CPU speedup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger('telemanom')


class Model:
    def __init__(self, config, run_id, channel):
        """
        Loads/trains RNN and predicts future telemetry values for a channel.

        Args:
            config (obj): Config object containing parameters for processing
                and model training
            run_id (str): Datetime referencing set of predictions in use
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Attributes:
            config (obj): see Args
            chan_id (str): channel id
            run_id (str): see Args
            y_hat (arr): predicted channel values
            model (obj): trained RNN model for predicting channel values
        """

        self.config = config
        self.chan_id = channel.id
        self.run_id = run_id
        self.y_hat = np.array([])
        self.model = None

        if not self.config.train and not self.config.train_only:
            try:
                logger.info('Loading pre-trained model')
                if self.config.model_type == "ESN":
                    hp = {}
                    if self.config.load_hp:
                        logger.info('Loading hp id: {}'.format(self.config.hp_and_weights_id))
                        #metti poi config
                        #path = os.path.join("hp", self.config.hp_and_weights_id, "config/{}.yaml".format(self.chan_id))
                        path = os.path.join("hp", self.config.hp_and_weights_id, "{}.yaml".format(self.chan_id))
                        with open(path, 'r') as file:
                            hp = yaml.load(file, Loader=yaml.BaseLoader)

                    self.model = create_esn_model(channel, self.config, hp)

                    self.model.compile(loss=self.config.loss_metric,
                                       optimizer=self.config.optimizer)

                    self.model.load_weights(os.path.join('data', self.config.use_id,
                                                         'models', self.chan_id + '.h5'))

                else:
                    self.model = load_model(os.path.join('data', self.config.use_id,
                                                         'models', self.chan_id + '.h5'))
            except (FileNotFoundError, OSError) as e:
                path = os.path.join('data', self.config.use_id, 'models',
                                    self.chan_id + '.h5')
                logger.warning('Training new model, couldn\'t find existing '
                               'model at {}'.format(path))

                print("dentro exception: {}".format(e))
                self.train_new(channel)
                self.save()

        if not self.config.train and self.config.train_only:
            logger.info("error in the configuration file, check the flags")
            sys.exit("error in the configuration file, check the flags")

        if self.config.train and self.config.train_only:
            self.train_new(channel)
            self.save()


    def train_new(self, channel):
        """
        Train ESN, LSTM or CNN model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """



        if self.config.model_type == "LSTM":
            cbs = [History(), EarlyStopping(monitor='val_loss',
                                            patience=self.config.patience,
                                            min_delta=self.config.min_delta,
                                            verbose=0)]

            self.model = create_lstm_model(channel,self.config)
            self.model.compile(loss=self.config.loss_metric,
                               optimizer=self.config.optimizer)

            self.history = self.model.fit(channel.X_train,
                                          channel.y_train,
                                          batch_size=self.config.lstm_batch_size,
                                          epochs=self.config.epochs,
                                          validation_split=self.config.validation_split,
                                          callbacks=cbs,
                                          verbose=True)


        if self.config.model_type == "ESN":
            cbs = [History(), EarlyStopping(monitor='val_loss',
                                            patience=self.config.esn_patience,
                                            min_delta=self.config.min_delta,
                                            verbose=0)]

            hp = {}
            if self.config.load_hp:
                path = os.path.join("hp",self.config.hp_id, "config/{}.yaml".format(self.chan_id))
                with open(path, 'r') as file:
                    hp = yaml.load(file, Loader=yaml.BaseLoader)

                logger.info('units: {}'.format(hp["units"]))
                logger.info('input_scaling: {}'.format(hp["input_scaling"]))
                logger.info('radius: {}'.format(hp["radius"]))
                logger.info('leaky: {}'.format(hp["leaky"]))
                logger.info('connectivity_recurrent: {}'.format(hp["connectivity_recurrent"]))
                logger.info('connectivity_input: {}'.format(hp["connectivity_input"]))
                logger.info('return_sequences: {}'.format(hp["return_sequences"]))
            else:
                logger.info("default hp".format(self.config.model_type))

            self.model = create_esn_model(channel,self.config, hp)


            self.history = self.model.fit(channel.X_train,
                                          channel.y_train,
                                          batch_size=self.config.lstm_batch_size,
                                          epochs=self.config.esn_epochs,
                                          validation_split=self.config.validation_split,
                                          callbacks=cbs,
                                          verbose=True)


        if self.config.model_type == "CNN":
            pass



        logger.info('validation_loss: {}'.format(self.history.history["val_loss"][-1]))

    def save(self):
        """
        Save trained model, loss and validation loss graphs .
        """

        if self.config.save_graphs:
            plt.figure()
            plt.plot(self.history.history["loss"], label="Training Loss")
            plt.plot(self.history.history["val_loss"], label="Validation Loss")
            plt.title(f'Training and validation loss model: {self.config.model_type} channel: {self.chan_id}')

            plt.legend()

            plt.savefig(os.path.join('data', self.run_id, 'images',
                                     '{}_loss.png'.format(self.chan_id)))
            plt.show()
            plt.close()

        if self.config.model_type == "ESN":
            self.model.save_weights(os.path.join('data', self.run_id, 'models',
                                         '{}.h5'.format(self.chan_id)))
        else:
            self.model.save(os.path.join('data', self.run_id, 'models',
                                         '{}.h5'.format(self.chan_id)))

    def aggregate_predictions(self, y_hat_batch, method='first'):
        """
        Aggregates predictions for each timestep. When predicting n steps
        ahead where n > 1, will end up with multiple predictions for a
        timestep.

        Args:
            y_hat_batch (arr): predictions shape (<batch length>, <n_preds)
            method (string): indicates how to aggregate for a timestep - "first"
                or "mean"
        """

        agg_y_hat_batch = np.array([])

        for t in range(len(y_hat_batch)):

            start_idx = t - self.config.n_predictions
            start_idx = start_idx if start_idx >= 0 else 0

            # predictions pertaining to a specific timestep lie along diagonal
            y_hat_t = np.flipud(y_hat_batch[start_idx:t+1]).diagonal()

            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)

    def batch_predict(self, channel):
        """
        Used trained LSTM model to predict test data arriving in batches.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Returns:
            channel (obj): Channel class object with y_hat values as attribute
        """

        num_batches = int((channel.y_test.shape[0] - self.config.l_s)
                          / self.config.batch_size)
        if num_batches < 0:
            raise ValueError('l_s ({}) too large for stream length {}.'
                             .format(self.config.l_s, channel.y_test.shape[0]))


        # simulate data arriving in batches, predict each batch
        for i in range(0, num_batches + 1):
            prior_idx = i * self.config.batch_size
            idx = (i + 1) * self.config.batch_size

            if i + 1 == num_batches + 1:
                # remaining values won't necessarily equal batch size
                idx = channel.y_test.shape[0]

            X_test_batch = channel.X_test[prior_idx:idx]
            y_hat_batch = self.model.predict(X_test_batch)
            self.aggregate_predictions(y_hat_batch)


        self.y_hat = np.reshape(self.y_hat, (self.y_hat.size,))

        channel.y_hat = self.y_hat

        np.save(os.path.join('data', self.run_id, 'y_hat', '{}.npy'
                             .format(self.chan_id)), self.y_hat)

        return channel
