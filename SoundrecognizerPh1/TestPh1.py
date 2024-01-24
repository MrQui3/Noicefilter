import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import load_model

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

def get_file_names(folder_path):
    file_names = []
    # Gehe durch jede Datei im angegebenen Ordner
    for file_name in os.listdir(folder_path):
        # Überprüfe, ob das Element eine Datei ist
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)
    return file_names

# Create a spectrogram from a wav file
def preprocess(file_path):
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
    return spectrogram

# this function is used to load the data from the given path
# and return a list of spectrograms in a numpy array
def settrainingsdata(path):
    training_data = []
    file_names = get_file_names(path)
    for filename in file_names:
        if filename.endswith(".wav"):
            spectrogram = preprocess(f"{path}\\{filename}")
            training_data.append(spectrogram)
    return training_data

interpreter = tf.lite.Interpreter(model_path='model_recognizerPh1.tflite')
interpreter.allocate_tensors()

# Hole die Eingabe- und Ausgabetensoren des Modells
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Definiere eine Funktion zum Ausführen von Vorhersagen
def run_inference(input_data):
    input_data = np.array(input_data, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0][0]

Mus = '../Data/testdata/musicdata'
Not_Mus = '../Data/testdata/rauschdata'
print(Mus)
print('Path is loaded')

training_data_mus = []
training_data_mus = settrainingsdata(Mus)
training_data_not_mus = []
training_data_not_mus = settrainingsdata(Not_Mus)
print('Data is loaded')

print('---------Start prediction----------')
print('---------Recognition of music-----------')
predictions = []
music_it_is = 0
for single_data in training_data_mus:
    prediction = run_inference(single_data)
    answer = [1 if prediction > 0.5 else 0]
    if answer == 1:
        music_it_is += 1
        print('To Music the prediction is: music')
    else:
        print('To Music the prediction is: not music')
    predictions.append([1, answer])

accuracy = music_it_is / len(training_data_mus)
print('The accuracy of the music recognition is: ', accuracy)

print('---------Recognition of noise-----------')
noise_it_is = 0
for single_data in training_data_not_mus:
    prediction = run_inference(single_data)
    answer = [1 if prediction > 0.5 else 0]
    if answer == 0:
        noise_it_is += 1
        print('To Noise the prediction is: noise')
    else:
        print('To Noise the prediction is: not noise')
    predictions.append([0, answer])

accuracy = noise_it_is / len(training_data_not_mus)
print('The accuracy of the noise recognition is: ', accuracy)

print('------------All predictions:------------')
print(predictions)
print('------------End of predictions------------')

