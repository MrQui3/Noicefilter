import tensorflow as tf
from fontTools.merge import cmap
from matplotlib import pyplot as plt
import numpy as np
import os
import scipy.signal as signal

#This part of the code is used to load the data from the folders
#the Method get_dataset() works

#private method
def get_file_names(folder_path):
    file_names = []
    # Gehe durch jede Datei im angegebenen Ordner
    for file_name in os.listdir(folder_path):
        # Überprüfe, ob das Element eine Datei ist
        if os.path.isfile(os.path.join(folder_path, file_name)) and file_name.endswith('.wav'):
            file_names.append(folder_path+"/"+file_name)
    return file_names

#private method
def load_wav_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    #if the frequency is lower the result has a shorter resolution. So it is not possible for good results to use a lower frequency
    #wav_length = tf.shape(wav)[0]
    #wav_length = tf.cast(wav_length, tf.int64)  # Set dtype to int64
    # wav = tf.py_function(signal.resample,
    #               [wav, tf.math.multiply(wav_length, tf.constant(16000, dtype=tf.int64) // sample_rate)],
     #              tf.float32)
    return wav

#private method
def preprocess(file_path):
    wav = load_wav_mono(file_path)
    # Here the audio is cut to 1s after the first 10s
    wav = wav[16000 * 5 : 16000 * 6]
    # if the audio is less than 2s, it will be padded with zeros
    zero_padding = tf.zeros([16000 * 1] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    # create the spectrogram
    # frame_length is the number of samples in each frame of the spectrogram
    # I'm doing this to have 100 frames for the1s audio
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    # the spectrogram is a complex number, so I take the absolute value
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    # Normalize the spectrogram
    spectrogram -= tf.reduce_mean(spectrogram)
    spectrogram /= tf.math.reduce_std(spectrogram)
    return spectrogram

#private method
def get_spectrograms(file_names):
    spectrograms = []
    nr = 0
    for file_name in file_names:
        spectrograms.append(preprocess(file_name))
        #nr+= 1
        #if nr >= 20:
         #   break
    return spectrograms

def get_dataset():
    input_shape = (991, 257,1)
    mus_path = '../Data/trainingdata2/musicdata'
    noise_mus_path = '../Data/trainingdata2/music_with_noise_data'
    print('Path is loaded')

    mus = get_file_names(mus_path)
    noise_mus = get_file_names(noise_mus_path)
    print('Files are loaded')

    print('Music file: ', mus[100])
    print('Music with noise file: ', noise_mus[100])
    mus_spec = get_spectrograms(mus)
    noise_mus_spec = get_spectrograms(noise_mus)
    print('Spectrograms are loaded')

    input_shape = mus_spec[0].shape

    return mus_spec, noise_mus_spec, input_shape
    # Create a dataset of the music data

#test the dataset
print('Test the dataset')
mus_spec, noise_mus_spec, input_shape = get_dataset()
print('Input shape: ', input_shape)
print('Music spectrogram shape: ', mus_spec[0].shape)
print('Noise spectrogram shape: ', noise_mus_spec[0].shape)
print('Number of music spectrograms: ', len(mus_spec))
print('Number of noise spectrograms: ', len(noise_mus_spec))

print('Spectorgram of music:')
spectrogram = mus_spec[100]
plt.imshow(spectrogram[..., 0])
plt.show()

print('Spectorgram of music with noise:')
spectrogram = noise_mus_spec[100]
plt.figure(figsize=(50, 40))
plt.imshow(spectrogram[..., 0])
plt.show()

print('Dataset is tested')