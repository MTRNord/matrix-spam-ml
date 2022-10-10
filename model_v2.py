import csv
import os
import re
import time
from datetime import datetime

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
from nltk.corpus import stopwords
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

vocab_size = 1000
# embedding_dim = 16
# embedding_dim = 32
# embedding_dim = 64


logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

hypermodel_logdir = (
    "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_hypermodel"
)
hypermodel_tensorboard_callback = keras.callbacks.TensorBoard(log_dir=hypermodel_logdir)

hypertuner_logdir = "hypertuner_logs/scalars/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S"
)
hypertuner_tensorboard_callback = keras.callbacks.TensorBoard(log_dir=hypertuner_logdir)
# Define the checkpoint directory to store the checkpoints.
checkpoint_dir = "./training_checkpoints"
# Define the name of the checkpoint files.
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

progress_bar = tf.keras.callbacks.ProgbarLogger()


class SpamDectionModel(tf.keras.Model):
    def __init__(self, vectorize_layer, hp_embedding_dim, hp_dense1, hp_dropout, hp_l2):
        super(SpamDectionModel, self).__init__()
        # self.vectorize_layer = vectorize_layer
        # self.embedding = tf.keras.layers.Embedding(
        #    input_dim=len(vectorize_layer.get_vocabulary()) + 1,
        #    output_dim=hp_embedding_dim,
        #    name="text_input",
        #    embeddings_initializer="uniform",
        #    # mask_zero=True,
        # )
        self.dropout = tf.keras.layers.Dropout(
            hp_dropout,
        )
        self.dense1 = tf.keras.layers.Dense(
            hp_dense1,
            activation="relu",
            #   kernel_regularizer=tf.keras.regularizers.l2(hp_l2),
        )
        self.dense2 = tf.keras.layers.Dense(
            1, activation="sigmoid", name="score_output"
        )
        # self.glob_average_pooling_1d = tf.keras.layers.GlobalAveragePooling1D()
        self.use_layer = hub.KerasLayer(
            "https://tfhub.dev/google/universal-sentence-encoder/4",
            trainable=False,
            input_shape=[],
            dtype=tf.string,
            name="USE",
        )

    @tf.function
    def call(self, x, training=False):
        # x = self.vectorize_layer(x)
        # x = self.embedding(x)
        # x = self.glob_average_pooling_1d(x)
        x = self.use_layer(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense1(x)
        # if training:
        #    x = self.dropout(x, training=training)
        return self.dense2(x)


class SpamDectionHyperModel(kt.HyperModel):
    def __init__(self, vectorize_layer):
        super(SpamDectionHyperModel, self).__init__()
        self.vectorize_layer = vectorize_layer

    def build(self, hp):
        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 6-512
        hp_dense1 = hp.Int("dense1", min_value=6, max_value=512, step=12)
        hp_embedding_dim = hp.Int(
            "embedding_dim", min_value=300, max_value=512, step=16
        )
        hp_dropout = hp.Float("dropout", min_value=0.1, max_value=0.9, step=0.1)
        hp_l2 = hp.Float("l2", min_value=0.0001, max_value=0.001, step=0.0001)
        model = SpamDectionModel(
            self.vectorize_layer,
            hp_embedding_dim + 1,
            hp_dense1,
            hp_dropout,
            hp_l2,
        )
        # Adam was best so far
        # tf.keras.optimizers.Nadam() has similar results to Adam but a bit worse. second best
        hp_learning_rate = hp.Choice(
            "learning_rate",
            values=[
                1e-2,
                1e-3,
                1e-4,
                1e-5,
            ],
        )
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
            metrics=["accuracy"],
        )

        return model


def remove_stopwords(input_text):
    """
    Function to remove English stopwords from a Pandas Series.

    Parameters:
        input_text : text to clean
    Output:
        cleaned Pandas Series
    """
    stopwords_list = stopwords.words("english")
    # Some words which might indicate a certain sentiment are kept via a whitelist
    whitelist = ["n't", "not", "no"]
    words = input_text.split()
    clean_words = [
        word
        for word in words
        if (word not in stopwords_list or word in whitelist) and len(word) > 1
    ]
    return " ".join(clean_words)


def change_labels(x):
    return 1 if x == "spam" else 0


def tokenize_data(data):
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        standardize="lower_and_strip_punctuation",
    )

    # Now that the vocab layer has been created, call `adapt` on the text-only
    # dataset to create the vocabulary. You don't have to batch, but for large
    # datasets this means we're not keeping spare copies of the dataset.
    vectorize_layer.adapt(data)

    # vocab = np.array(vectorize_layer.get_vocabulary())
    # print(f"First 20 of Vocab: {vocab[:20]}")

    return vectorize_layer


def load_data():
    data = pd.read_csv(
        "./input/MatrixData.tsv", sep="\t", quoting=csv.QUOTE_NONE, encoding="utf-8"
    )

    # Minimum length
    data = data[df["message"].str.split().str.len().gt(18)]
    # Remove unknown
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    data["label"] = data["label"].apply(change_labels)

    # Remove stopwords
    data["message"] = data["message"].apply(remove_stopwords)

    # Shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    # Split data into messages and label sets
    sentences = data["message"].tolist()
    labels = data["label"].tolist()

    # Separate out the sentences and labels into training and test sets
    # training_size = int(len(sentences) * 0.8)
    training_size = int(len(sentences) * 0.7)
    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    # Make labels into numpy arrays for use with the network later
    testing_labels_final = np.array(testing_labels)
    training_labels_final = np.array(training_labels)
    training_sentences_final = np.array(training_sentences)
    testing_sentences_final = np.array(testing_sentences)
    vectorize_layer = tokenize_data(sentences)
    return (
        vectorize_layer,
        training_sentences_final,
        testing_sentences_final,
        training_labels_final,
        testing_labels_final,
    )


def train_hyperparamters(
    training_sentences_final,
    testing_sentences_final,
    training_labels_final,
    testing_labels_final,
    tuner,
):
    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    tuner.search(
        training_sentences_final,
        training_labels_final,
        epochs=5,
        verbose=1,
        validation_data=(testing_sentences_final, testing_labels_final),
        callbacks=[hypertuner_tensorboard_callback, stop_early, progress_bar],
    )

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('dense1')} and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
    The optimal dropout rate is {best_hps.get('dropout')} and the optimal l2 rate is {best_hps.get('l2')}.
    The optimal embedding_dim is {best_hps.get('embedding_dim')}.
    """
    )

    return best_hps


def train_model(
    vectorize_layer,
    training_sentences_final,
    testing_sentences_final,
    training_labels_final,
    testing_labels_final,
    # best_hps,
    # tuner,
):
    model = SpamDectionModel(
        vectorize_layer,
        0,
        64,
        0.2,
        0,
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    num_epochs = 200
    # model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        training_sentences_final,
        training_labels_final,
        epochs=num_epochs,
        verbose=1,
        validation_data=(testing_sentences_final, testing_labels_final),
        callbacks=[tensorboard_callback, progress_bar],
    )
    val_acc_per_epoch = history.history["val_accuracy"]
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print("Best epoch: %d" % (best_epoch,))
    print("Average train loss: ", np.average(history.history["loss"]))
    print("Average test loss: ", np.average(history.history["val_loss"]))

    model = SpamDectionModel(
        vectorize_layer,
        0,
        64,
        0.2,
        0,
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    # hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel_history = model.fit(
        training_sentences_final,
        training_labels_final,
        verbose=1,
        epochs=best_epoch,
        validation_data=(testing_sentences_final, testing_labels_final),
        callbacks=[
            hypermodel_tensorboard_callback,
            # tf.keras.callbacks.ModelCheckpoint(
            #    filepath=checkpoint_prefix, save_weights_only=True
            # ),
            progress_bar
            # es_callback
        ],
    )

    print(
        "Average train loss(hypermodel_history): ",
        np.average(hypermodel_history.history["loss"]),
    )
    print(
        "Average test loss(hypermodel_history): ",
        np.average(hypermodel_history.history["val_loss"]),
    )

    return model


def test_model(vectorize_layer, model):
    # Use the model to predict whether a message is spam
    text_messages = [
        "Greg, can you call me back once you get this?",
        "Congrats on your new iPhone! Click here to claim your prize...",
        "Really like that new photo of you",
        "Did you hear the news today? Terrible what has happened...",
        "Attend this free COVID webinar today: Book your session now...",
        "Are you coming to the party tonight?",
        "Your parcel has gone missing",
        "Do not forget to bring friends!",
        "You have won a million dollars! Fill out your bank details here...",
        "Looking forward to seeing you again",
        "oh wow https://github.com/MGCodesandStats/tensorflow-nlp/blob/master/spam%20detection%20tensorflow%20v2.ipynb works really good on spam detection. Guess I go with that as the base model then lol :D",
        "ayo",
        "Almost all my spam is coming to my non-gmail address actually",
        "Oh neat I think I found the sizing sweetspot for my data :D",
        "would never click on buttons in gmail :D always expecting there to be a bug in gmail that allows js to grab your google credentials :D XSS via email lol. I am too scared for touching spam in gmail",
        "back to cacophony ",
        "Room version 11 when",
        "skip 11 and go straight to 12",
        "100 events should clear out any events that might be causing a request to fail lol",
        "I'll help anyone interested on how to invest and earn $30k, $50k, $100k, $200k or more in just 72hours from the crypto market.But you will have to pay me my commission! when you receive your profit! if interested send me a direct message let's get started or via WhatsApp +1 (605) 953‑6801",
    ]

    spam_no_spam = [
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]

    # print(text_messages)

    # Create the sequences
    classes = model.predict(np.array(text_messages))

    # The closer the class is to 1, the more likely that the message is spam
    for x in range(len(text_messages)):
        print(f'Message: "{text_messages[x]}"')
        print(f"Likeliness of spam in percentage: {classes[x][0]:.5f}")
        spam = classes[x][0] >= 0.8
        if spam:
            print("Vote by AI: Spam")
        else:
            print("Vote by AI: Not Spam")
        if spam_no_spam[x] != spam:
            print("Model failed to predict correctly")
        else:
            print("Model predicted correctly")
        print("\n")


def main():
    tf.get_logger().setLevel("ERROR")
    print("TensorFlow version:", tf.__version__)
    print("[Step 1/6] Loading data")
    (
        vectorize_layer,
        training_sentences_final,
        testing_sentences_final,
        training_labels_final,
        testing_labels_final,
    ) = load_data()

    # model = SpamDectionHyperModel(vectorize_layer)
    # tuner = kt.Hyperband(
    #    model,
    #    objective="val_accuracy",
    #    max_epochs=100,
    #    directory="hyper_tuning",
    #    project_name="spam-keras",
    # )
    # print("[Step 3/6] Tuning hypervalues")
    # best_hps = train_hyperparamters(
    #    training_sentences_final,
    #    testing_sentences_final,
    #    training_labels_final,
    #    testing_labels_final,
    #    tuner,
    # )

    print("[Step 4/6] Training model")
    model = train_model(
        vectorize_layer,
        training_sentences_final,
        testing_sentences_final,
        training_labels_final,
        testing_labels_final,
        # best_hps,
        # tuner,
    )

    print("[Step 5/6] Saving model")
    export_path = f"./models/spam_keras_{time.time()}"
    print("Exporting trained model to", export_path)

    model.save(export_path)

    print("[Step 6/6] Testing model")
    test_model(vectorize_layer, model)


if __name__ == "__main__":
    main()
