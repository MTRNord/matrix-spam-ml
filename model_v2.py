import os
import time
from datetime import datetime

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from nltk.corpus import stopwords
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

vocab_size = 1000
embedding_dim = 16
# embedding_dim = 32
# max_length = 120
max_length = None
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"


logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

hypermodel_logdir = "logs/scalars/" + \
    datetime.now().strftime("%Y%m%d-%H%M%S") + "_hypermodel"
hypermodel_tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=hypermodel_logdir)

hypertuner_logdir = "hypertuner_logs/scalars/" + \
    datetime.now().strftime("%Y%m%d-%H%M%S")
hypertuner_tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=hypertuner_logdir)
# Define the checkpoint directory to store the checkpoints.
checkpoint_dir = './training_checkpoints'
# Define the name of the checkpoint files.
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

progress_bar = tf.keras.callbacks.ProgbarLogger()


class SpamDectionModel(tf.keras.Model):
    def __init__(self, vectorize_layer, vocab_size, embedding_dim, max_length, hp_units, hp_dropout, hp_l2):
        super(SpamDectionModel, self).__init__()
        #self.input_layer = tf.keras.Input(shape=(1,), dtype=tf.string)
        self.vectorize_layer = vectorize_layer
        self.embedding = tf.keras.layers.Embedding(
            vocab_size + 1, embedding_dim, input_length=max_length, name="text_input")
        self.glob_average_pooling_1d = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(hp_dropout,)
        self.dense1 = tf.keras.layers.Dense(units=hp_units, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(hp_l2))
        # tf.keras.layers.Dense(6, activation='relu',
        #                      kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        self.dense2 = tf.keras.layers.Dense(
            1, activation='sigmoid', name="score_output")

    @tf.function
    def call(self, x, training=False):
        #x = self.input_layer(x)
        x = self.vectorize_layer(x)
        x = self.embedding(x)
        x = self.glob_average_pooling_1d(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense1(x)
        if training:
            x = self.dropout(x, training=training)
        return self.dense2(x)


class SpamDectionHyperModel(kt.HyperModel):
    def __init__(self, vectorize_layer, vocab_size, embedding_dim, max_length,):
        super(SpamDectionHyperModel, self).__init__()
        self.vectorize_layer = vectorize_layer
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length

    def build(self, hp):
        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 6-512
        hp_units = hp.Int('units', min_value=6, max_value=512, step=12)
        hp_dropout = hp.Float('dropout', min_value=.1, max_value=.9, step=.01)
        hp_l2 = hp.Float('l2', min_value=0.0001, max_value=0.001, step=0.0001)
        model = SpamDectionModel(self.vectorize_layer,
                                 self.vocab_size, self.embedding_dim, self.max_length, hp_units, hp_dropout, hp_l2)
        # Adam was best so far
        # tf.keras.optimizers.Nadam() has similar results to Adam but a bit worse. second best
        hp_learning_rate = hp.Choice('learning_rate', values=[
            1e-2, 1e-3, 1e-4, 1e-5, ])
        opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
        # opt = tf.keras.optimizers.Nadam()
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=opt, metrics=['accuracy'])
        # print(model.summary())

        return model


def remove_stopwords(input_text):
    '''
    Function to remove English stopwords from a Pandas Series.

    Parameters:
        input_text : text to clean
    Output:
        cleaned Pandas Series
    '''
    stopwords_list = stopwords.words('english')
    # Some words which might indicate a certain sentiment are kept via a whitelist
    whitelist = ["n't", "not", "no"]
    words = input_text.split()
    clean_words = [word for word in words if (
        word not in stopwords_list or word in whitelist) and len(word) > 1]
    return " ".join(clean_words)


@tf.function
def change_labels(x): return 1 if x == "spam" else 0


def tokenize_data(data):
    # tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    # tokenizer.fit_on_texts(training_sentences)

    # sequences = tokenizer.texts_to_sequences(training_sentences)
    # padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type,
    #                       truncating=trunc_type)

    # testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    # testing_padded = pad_sequences(testing_sequences, maxlen=max_length,
    #                               padding=padding_type, truncating=trunc_type)

    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=max_length)

    # Now that the vocab layer has been created, call `adapt` on the text-only
    # dataset to create the vocabulary. You don't have to batch, but for large
    # datasets this means we're not keeping spare copies of the dataset.
    vectorize_layer.adapt(data)

    # Create the model that uses the vectorize text layer
    # model = tf.keras.models.Sequential()
    # Start by creating an explicit input layer. It needs to have a shape of
    # (1,) (because we need to guarantee that there is exactly one string
    # input per batch), and the dtype needs to be 'string'.
    # model.add(tf.keras.Input(shape=(1,), dtype=tf.string))

    # The first layer in our model is the vectorization layer. After this
    # layer, we have a tensor of shape (batch_size, max_len) containing vocab
    # indices.
    # model.add(vectorize_layer)

    # Now, the model can map strings to integers, and you can add an embedding
    # layer to map these integers to learned embeddings.
    # padded = model.predict(training_sentences)
    # testing_padded = model.predict(testing_sentences)

    return vectorize_layer  # , tokenizer


def load_data():
    data = pd.read_csv('./input/MatrixData', sep='\t')

    # Remove unknown
    data.dropna(inplace=True)
    data['label'] = data['label'].apply(change_labels)

    # Remove stopwords
    data['message'] = data['message'].apply(remove_stopwords)

    # Shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    # Split data into messages and label sets
    sentences = data['message'].tolist()
    labels = data['label'].tolist()

    # Make labels into numpy arrays for use with the network later
    labels_final = np.array(labels)
    sentences_final = np.array(sentences)
    vectorize_layer = tokenize_data(sentences)
    return vectorize_layer, sentences_final, labels_final


def train_hyperparamters(data, labels_final, tuner):
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5)
    tuner.search(data, labels_final, epochs=5, verbose=1, validation_split=0.3,
                 callbacks=[hypertuner_tensorboard_callback, stop_early, progress_bar])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
    The optimal dropout rate is {best_hps.get('dropout')} and the optimal l2 rate is {best_hps.get('l2')}.
    """)

    return best_hps


def train_model(data, labels_final, best_hps, tuner):
    num_epochs = 200
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(data,
                        labels_final,
                        epochs=num_epochs,
                        verbose=1, validation_split=0.3,
                        callbacks=[tensorboard_callback,
                                   progress_bar],)
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 5
    print('Best epoch: %d' % (best_epoch,))
    print("Average train loss: ", np.average(history.history['loss']))
    print("Average test loss: ", np.average(history.history['val_loss']))

    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel_history = hypermodel.fit(data, labels_final, verbose=1,
                                        epochs=best_epoch, validation_split=0.3,
                                        callbacks=[hypermodel_tensorboard_callback,
                                                   tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                                                      save_weights_only=True), progress_bar
                                                   # es_callback
                                                   ]
                                        )

    print("Average train loss(hypermodel_history): ",
          np.average(hypermodel_history.history['loss']))
    print("Average test loss(hypermodel_history): ",
          np.average(hypermodel_history.history['val_loss']))

    return hypermodel


def test_model(vectorize_layer, model):
    # Use the model to predict whether a message is spam
    text_messages = ['Greg, can you call me back once you get this?',
                     'Congrats on your new iPhone! Click here to claim your prize...',
                     'Really like that new photo of you',
                     'Did you hear the news today? Terrible what has happened...',
                     'Attend this free COVID webinar today: Book your session now...',
                     'Are you coming to the party tonight?',
                     'Your parcel has gone missing',
                     'Do not forget to bring friends!',
                     'You have won a million dollars! Fill out your bank details here...',
                     'Looking forward to seeing you again',
                     'oh wow https://github.com/MGCodesandStats/tensorflow-nlp/blob/master/spam%20detection%20tensorflow%20v2.ipynb works really good on spam detection. Guess I go with that as the base model then lol :D',
                     'ayo',
                     'Almost all my spam is coming to my non-gmail address actually',
                     'Oh neat I think I found the sizing sweetspot for my data :D',
                     'would never click on buttons in gmail :D always expecting there to be a bug in gmail that allows js to grab your google credentials :D XSS via email lol. I am too scared for touching spam in gmail',
                     'back to cacophony ',
                     'Room version 11 when',
                     'skip 11 and go straight to 12',
                     '100 events should clear out any events that might be causing a request to fail lol']

    # print(text_messages)

    # Create the sequences
    padding_type = 'post'
    # sample_sequences = tokenizer.texts_to_sequences(text_messages)
    # fakes_padded = pad_sequences(
    #    sample_sequences, padding=padding_type, maxlen=max_length)
    classes = model.predict(text_messages)

    # The closer the class is to 1, the more likely that the message is spam
    for x in range(len(text_messages)):
        print(f"Message: \"{text_messages[x]}\"")
        print(f"Likeliness of spam in percentage: {classes[x][0]:.5f}")
        print('\n')


def main():
    print("TensorFlow version:", tf.__version__)
    print("[Step 1/6] Loading data")
    vectorize_layer, data, labels_final = load_data()
    model = SpamDectionHyperModel(
        vectorize_layer, vocab_size, embedding_dim, max_length)
    # print("[Step 2/6] Plotting model")
    # tf.keras.utils.plot_model(model, rankdir="LR", show_shapes=True)
    tuner = kt.Hyperband(model, hyperband_iterations=2,
                         objective='val_accuracy',
                         max_epochs=200,
                         directory='hyper_tuning',
                         project_name='spam-keras')
    print("[Step 3/6] Tuning hypervalues")
    best_hps = train_hyperparamters(data, labels_final, tuner)
    print("[Step 4/6] Training model")
    model = train_model(data, labels_final, best_hps, tuner)

    print("[Step 5/6] Saving model")
    export_path = f"./models/spam_keras_{time.time()}"
    print('Exporting trained model to', export_path)

    model.save(export_path)

    print("[Step 6/6] Testing model")
    test_model(vectorize_layer, model)


if __name__ == "__main__":
    main()
