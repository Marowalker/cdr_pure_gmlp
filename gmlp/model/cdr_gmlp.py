import tensorflow as tf
import constants
from constants import *
import numpy as np
from gmlp.gmlp import gMLP
from gmlp.simple_gmlp import gMLPLayer
import os
import keras.backend as K
from data_utils import *


def f1_macro(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


class CDRgMLPModel:
    def __init__(self, depth, chem_emb, dis_emb, wordnet_emb, cdr_emb, rel_emb):
        if not os.path.exists(TRAINED_MODELS):
            os.makedirs(TRAINED_MODELS)

        self.depth = depth
        self.triple_emb = tf.concat([chem_emb, dis_emb], axis=0)
        self.wordnet_emb = wordnet_emb
        self.cdr_emb = tf.concat([cdr_emb, rel_emb], axis=0)

        self.max_length = constants.MAX_LENGTH
        self.num_of_words = countVocab()
        self.num_of_pos = countNumPos()
        self.num_of_synset = countNumSynset()
        self.num_of_depend = countNumRelation()
        self.num_of_class = len(constants.ALL_LABELS)
        self.trained_models = constants.TRAINED_MODELS

    def _add_inputs(self):
        self.input_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype='int32')
        self.pos_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype='int32')
        self.synset_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype='int32')
        self.triple_ids = tf.keras.layers.Input(shape=(2,), dtype='int32')
        # self.position_1_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype='int32')
        # self.position_2_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype='int32')

    def _bert_layer(self):

        pos_emb = tf.keras.layers.Embedding(self.num_of_pos + 1, 6)(self.pos_ids)

        synset_emb = tf.keras.layers.Embedding(self.wordnet_emb.shape[0], 18, weights=[self.wordnet_emb],
                                               trainable=False)(self.synset_ids)

        triple_emb = tf.keras.layers.Embedding(self.triple_emb.shape[0], constants.TRIPLE_W2V_DIM,
                                               weights=[self.triple_emb], trainable=False)(self.triple_ids)

        # positions_1_emb = tf.keras.layers.Embedding(self.max_length * 2, 25)(
        #     self.position_1_ids)
        # positions_2_emb = tf.keras.layers.Embedding(self.max_length * 2, 25)(
        #     self.position_2_ids)
        # position_emb = tf.concat([positions_1_emb, positions_2_emb], axis=-1)

        word_emb = tf.keras.layers.Embedding(self.num_of_words + self.num_of_depend + 2, 16,
                                             weights=[self.cdr_emb], trainable=False)(self.input_ids)

        word_x = gMLP(dim=constants.INPUT_W2V_DIM, depth=self.depth, seq_len=self.max_length,
                      activation=tf.nn.swish)(word_emb)
        pos_x = gMLP(dim=6, depth=self.depth, seq_len=self.max_length, activation=tf.nn.swish)(pos_emb)
        synset_x = gMLP(dim=18, depth=self.depth, seq_len=self.max_length, activation=tf.nn.swish)(synset_emb)
        triple_x = gMLP(dim=constants.TRIPLE_W2V_DIM, depth=self.depth, seq_len=2, activation=tf.nn.swish)(triple_emb)

        pos_x = tf.keras.layers.Flatten(data_format="channels_first")(pos_x)
        pos_x = tf.keras.layers.LayerNormalization()(pos_x)
        # pos_x = tf.keras.layers.Dropout(constants.DROPOUT)(pos_x)
        pos_x = tf.keras.layers.Dense(6)(pos_x)

        synset_x = tf.keras.layers.Flatten(data_format="channels_first")(synset_x)
        synset_x = tf.keras.layers.LayerNormalization()(synset_x)
        # synset_x = tf.keras.layers.Dropout(constants.DROPOUT)(synset_x)
        synset_x = tf.keras.layers.Dense(18)(synset_x)

        word_x = tf.keras.layers.Flatten(data_format="channels_first")(word_x)
        word_x = tf.keras.layers.LayerNormalization()(word_x)
        # pos_x = tf.keras.layers.Dropout(constants.DROPOUT)(pos_x)
        word_x = tf.keras.layers.Dense(constants.INPUT_W2V_DIM)(word_x)

        triple_x = tf.keras.layers.Flatten(data_format="channels_first")(triple_x)
        triple_x = tf.keras.layers.LayerNormalization()(triple_x)
        # triple_x = tf.keras.layers.Dropout(constants.DROPOUT)(triple_x)
        triple_x = tf.keras.layers.Dense(constants.TRIPLE_W2V_DIM)(triple_x)

        # position_x = tf.keras.layers.Flatten(data_format="channels_first")(position_x)
        # position_x = tf.keras.layers.LayerNormalization()(position_x)
        # # position_x = tf.keras.layers.Dropout(constants.DROPOUT)(position_x)
        # position_x = tf.keras.layers.Dense(50)(position_x)

        x = tf.keras.layers.concatenate([word_x, pos_x, synset_x, triple_x])
        # x = tf.keras.layers.concatenate([head_x, e1_x, e2_x, pos_x, synset_x, triple_x])

        out = tf.keras.layers.Dropout(DROPOUT)(x)
        out = tf.keras.layers.Dense(128)(out)
        out = tf.keras.layers.Dense(128)(out)
        out = tf.keras.layers.Dense(len(constants.ALL_LABELS), activation='softmax')(out)
        return out

    @staticmethod
    def f1_score(y_true, y_pred):
        return f1_macro(y_true, y_pred)

    def _add_train_ops(self):
        self.model = tf.keras.Model(
            inputs=[self.input_ids, self.pos_ids, self.synset_ids, self.triple_ids],
            # inputs=[self.input_ids, self.head_mask, self.e1_mask, self.e2_mask, self.pos_ids, self.synset_ids,
            #         self.triple_ids],
            outputs=self._bert_layer())
        self.optimizer = tf.keras.optimizers.Adam(lr=4e-6)
        self.model.compile(optimizer=self.optimizer,
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=['accuracy', self.f1_score])
        print(self.model.summary())

    def _train(self, train_data, val_data):

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', mode='max',
                                                          patience=constants.PATIENCE)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=TRAINED_MODELS,
            save_weights_only=True,
            monitor='val_f1_score',
            mode='max',
            save_best_only=True)

        self.model.fit(x=(train_data.words, train_data.poses, train_data.synsets, train_data.triples),
                       y=train_data.labels,
                       validation_data=((val_data.words, val_data.poses, val_data.synsets, val_data.triples),
                                        val_data.labels),
                       batch_size=16, epochs=constants.EPOCHS, callbacks=[early_stopping, model_checkpoint_callback])

        # self.model.save_weights(TRAINED_MODELS)

    def plot_model(self):
        tf.keras.utils.plot_model(self.model, to_file='model.png', show_shapes=False, show_layer_names=True,
                                  rankdir='TB',
                                  expand_nested=False, dpi=300)

    def build(self, train_data, val_data):
        with tf.device('/device:GPU:0'):
            self._add_inputs()
            self._add_train_ops()
            self._train(train_data, val_data)
            # self.plot_model()

    def predict(self, test_data):
        self.model.load_weights(TRAINED_MODELS)
        pred = self.model.predict([test_data.words, test_data.poses, test_data.synsets, test_data.triples])
        y_pred = []
        for logit in pred:
            y_pred.append(np.argmax(logit))
        return y_pred
