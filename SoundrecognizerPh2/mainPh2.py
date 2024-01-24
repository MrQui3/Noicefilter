import os
from matplotlib import pyplot as plt
import tensorflow as tf
import scipy.signal as signal
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten,  Dropout
import numpy as np

def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    # if the frequency is lower the result has a shorter resolution. So it is not possible for good results to use a lower frequency
    #wav_length = tf.shape(wav)[0]
    #wav_length = tf.cast(wav_length, tf.int64)  # Set dtype to int64
    #wav = tf.py_function(signal.resample,
    #                 [wav, tf.math.multiply(wav_length, tf.constant(16000, dtype=tf.int64) // sample_rate)],
    #                 tf.float32)
    return wav

MUS = '../Data/trainingdata/musicdata'
NOT_MUS = '../Data/trainingdata/rauschdata'
MUSwithNoise = '../Data/trainingdata/ueberlagerungvonrauschen'
print('Path is set')

mus = tf.data.Dataset.list_files(MUS+'/*.wav')
not_mus = tf.data.Dataset.list_files(NOT_MUS+'/*.wav')
muswithnoise = tf.data.Dataset.list_files(MUSwithNoise+'/*.wav')
print('Files are loaded')

# Create labels in the form of tensors for the model actionfunktion 'CategoricalCrossentropy'
# Labels are 0 for music, 1 for noise and 2 for music with noise
labels_mus = tf.data.Dataset.from_tensor_slices(tf.zeros(len(mus), dtype=tf.int32))
labels_not_mus = tf.data.Dataset.from_tensor_slices(tf.ones(len(not_mus), dtype=tf.int32))
labels_muswithnoise = tf.data.Dataset.from_tensor_slices(tf.fill([len(muswithnoise)], 2))
print('Labels are loaded')

# Combine the datasets
musics = tf.data.Dataset.zip((mus, labels_mus))
not_musics = tf.data.Dataset.zip((not_mus, labels_not_mus))
musicswithnoise = tf.data.Dataset.zip((muswithnoise, labels_muswithnoise))

data = musics.concatenate(not_musics).concatenate(musicswithnoise)
print('Data is loaded')

def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)
    # 48000 is 2s of audio
    wav = wav[:16000 * 2]
    # if the audio is less than 2s, it will be padded with zeros
    zero_padding = tf.zeros([16000 * 2] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    # create the spectrogram
    # frame_length is the number of samples in each frame of the spectrogram
    # I'm doing this to have 100 frames for the 2s audio
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    # the spectrogram is a complex number, so I take the absolute value
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    # Normalize the spectrogram
    spectrogram -= tf.reduce_mean(spectrogram)
    spectrogram /= tf.math.reduce_std(spectrogram)
    label = tf.one_hot(label, depth=3)
    return spectrogram, label


filepath, label = data.shuffle(buffer_size=1000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)
plt.figure(figsize = (50, 40))
plt.imshow(tf.transpose(spectrogram)[0])
plt.show()
print("Label: ", label)

#prepare the data for training
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

train = data.take(360)
test = data.skip(60).take(15)

samples, labels = train.as_numpy_iterator().next()
print(samples.shape[0])

#model creation
model = Sequential()
#input shape is the shape of the spectrogram, get with this methode samples.shape
model.add(Conv2D(16, (3,3), activation='relu', input_shape=samples.shape))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
#with the activation function softmax, the output is between 0 and 1 and all dense values are summed up to 1 -> probability
model.add(Dense(3, activation='softmax'))

model.compile('Adam', loss='CategoricalCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
print(model.summary())

#train the model
hist = model.fit(train, epochs=40, validation_data=test)

#save the model as h5
model.save('model_recognizerPh2.h5', hist)
print('Model saved as h5')

#Save the model as tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open('model_recognizerPh2.tflite', 'wb') as f:
    f.write(tflite_model)
print('Model saved as tflite')


#plot the results of the training of the model
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
yhat_classes = np.argmax(yhat, axis=1)  # It chooses the class with the highest probability
print(yhat_classes)
print(y_test)

print('Done')