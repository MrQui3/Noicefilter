import os
from matplotlib import pyplot as plt
import tensorflow as tf
import scipy.signal as signal
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout

import numpy as np

def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal -> the result has a shorter resolution
    #wav_length = tf.shape(wav)[0]
    #wav_length = tf.cast(wav_length, tf.int64)  # Set dtype to int64
    #wav = tf.py_function(signal.resample,
     #                    [wav, tf.math.multiply(wav_length, tf.constant(32000, dtype=tf.int64) // sample_rate)],
     #                    tf.float32)
    return wav

MUS = '../Data/trainingdata/musicdata'
NOT_MUS = '../Data/trainingdata/rauschdata'
print(MUS)
print('Path is loaded')
mus = tf.data.Dataset.list_files(MUS+'/*.wav')
not_mus = tf.data.Dataset.list_files(NOT_MUS+'/*.wav')

musics = tf.data.Dataset.zip((mus, tf.data.Dataset.from_tensor_slices(tf.ones(len(list(mus))))))
not_musics = tf.data.Dataset.zip((not_mus, tf.data.Dataset.from_tensor_slices(tf.zeros(len(list(not_mus))))))

data = musics.concatenate(not_musics)
print('Data is loaded')

def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)
    # 48000 is 3s of audio
    wav = wav[:16000 * 3]
    # if the audio is less than 3s, it will be padded with zeros
    zero_padding = tf.zeros([16000 * 3] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    # at that point will the spectrogram created
    # frame_length is the number of samples in each frame of the spectrogram
    # i do this, because the audio is 3s long, and i want to have 100 frames
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    # the spectrogram is a complex number, so i take the absolute value
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


filepath, label = data.shuffle(buffer_size=1000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)
plt.figure(figsize=(50, 40))
plt.imshow(spectrogram[..., 0])
plt.show()
print("Label: ", label)

print('Data is splitted')
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

train = data.take(36)
test = data.skip(36).take(15)

samples, labels = train.as_numpy_iterator().next()
print(samples.shape)

print('Build the model')
#model creation
model = Sequential()
#input shape is the shape of the spectrogram, see bei
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257,1)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
print(model.summary())

#train the model
hist = model.fit(train, epochs=4, validation_data=test)

print('Model is trained')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open('model_recognizerPh1.tflite', 'wb') as f:
    f.write(tflite_model)

model.save('model_recognizerPh1.h5', hist)
print('Model is saved')

# plot the results of the training of the model
plt.title('Loss')
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'b')
plt.show()

plt.title('Precision')
plt.plot(hist.history['precision'], 'r')
plt.plot(hist.history['val_precision'], 'b')
plt.show()

plt.title('Recall')
plt.plot(hist.history['recall'], 'r')
plt.plot(hist.history['val_recall'], 'b')
plt.show()


X_test, y_test = test.as_numpy_iterator().next()
yhat = model.predict(X_test)
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
print(yhat)
print(y_test)