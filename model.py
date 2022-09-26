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

print("[Step 1/9] Loading data")
# Read data
data = pd.read_csv('./input/MatrixData', sep='\t')


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


# Remve unknown
data.dropna(inplace=True)

# Convert label to something useful


def change_labels(x): return 1 if x == "spam" else 0


data['label'] = data['label'].apply(change_labels)

# Count by label
spam = 0
ham = 0


def count_labels(x):
    if x == 1:
        global spam
        spam += 1
    else:
        global ham
        ham += 1
    return x
# .apply(count_labels)
#print("Spam: ", spam)
#print("Ham: ", ham)


# Remove stopwords
data['message'] = data['message'].apply(
    remove_stopwords)

# Print unbalanced
print(data.groupby('label').describe().T)


#ham_msg = data[data.label == 0]
#spam_msg = data[data.label == 1]

# randomly taking data from ham_msg
#ham_msg = ham_msg.sample(n=len(spam_msg)*2, random_state=42)

#data = pd.concat([ham_msg, spam_msg]).reset_index(drop=True)

# Balanced
print(data.groupby('label').describe().T)

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Split data into messages and label sets
sentences = data['message'].tolist()
labels = data['label'].tolist()

# Separate out the sentences and labels into training and test sets
#training_size = int(len(sentences) * 0.8)
training_size = int(len(sentences) * 0.7)
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Make labels into numpy arrays for use with the network later
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

print("[Step 2/9] Tokenizing data")
vocab_size = 1000
embedding_dim = 16
#embedding_dim = 32
#max_length = 120
max_length = None
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type,
                       truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length,
                               padding=padding_type, truncating=trunc_type)


print("[Step 3/9] Prepare callbacks")
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

#es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

print("[Step 4/9] Creating model")


class SpamDectionModel(kt.HyperModel):
    def build(self, hp):
        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 6-512
        hp_units = hp.Int('units', min_value=6, max_value=512, step=12)
        hp_dropout = hp.Float('dropout', min_value=.1, max_value=.9, step=.01)
        hp_l2 = hp.Float('l2', min_value=0.0001, max_value=0.001, step=0.0001)
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(hp_dropout,),
            tf.keras.layers.Dense(units=hp_units, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(hp_l2)),
            # tf.keras.layers.Dense(6, activation='relu',
            #                      kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.Dropout(hp_dropout,),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
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


print("[Step 5/9] Tuning hypervalues")
tuner = kt.Hyperband(SpamDectionModel(),
                     objective='val_accuracy',
                     max_epochs=750,
                     factor=3,
                     directory='hyper_tuning',
                     project_name='spam-keras')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(padded, training_labels_final, epochs=800, verbose=0,
             validation_data=(testing_padded, testing_labels_final), callbacks=[hypertuner_tensorboard_callback, stop_early])
# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
The optimal dropout rate is {best_hps.get('dropout')} and the optimal l2 rate is {best_hps.get('l2')}.
""")


print("[Step 6/9] Fitting initial model")
num_epochs = 200
model = tuner.hypermodel.build(best_hps)
history = model.fit(padded,
                    training_labels_final,
                    epochs=num_epochs,
                    verbose=0,
                    callbacks=[tensorboard_callback, ],
                    validation_data=(testing_padded, testing_labels_final))


val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
print("Average train loss: ", np.average(history.history['loss']))
print("Average test loss: ", np.average(history.history['val_loss']))

print("[Step 7/9] Building final model")
hypermodel = tuner.hypermodel.build(best_hps)
hypermodel_history = hypermodel.fit(padded, training_labels_final, verbose=0,
                                    epochs=best_epoch,
                                    callbacks=[hypermodel_tensorboard_callback,
                                               tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                                                  save_weights_only=True),
                                               # es_callback
                                               ], validation_data=(testing_padded, testing_labels_final)
                                    )

print("Average train loss(hypermodel_history): ",
      np.average(hypermodel_history.history['loss']))
print("Average test loss(hypermodel_history): ",
      np.average(hypermodel_history.history['val_loss']))


print("[Step 8/9] Saving final model")
# Save model
hypermodel.save(f"./models/spam_keras_{time.time()}")

print("[Step 9/9] Testing final model")
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
sample_sequences = tokenizer.texts_to_sequences(text_messages)
fakes_padded = pad_sequences(
    sample_sequences, padding=padding_type, maxlen=max_length)

classes = hypermodel.predict(fakes_padded)

# The closer the class is to 1, the more likely that the message is spam
for x in range(len(text_messages)):
    print(f"Message: \"{text_messages[x]}\"")
    print(f"Likeliness of spam in percentage: {classes[x][0]:.5f}")
    print('\n')


#tf.keras.utils.plot_model(model, rankdir="LR", show_shapes=True)
