import tensorflow as tf

# Lade das gespeicherte Modell
model = tf.keras.models.load_model('../Models/model_recognizerPh1.h5')

# Konvertiere das Modell zu TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Speichere das konvertierte Modell
with open('../Models/model_recognizerPh1.tflite', 'wb') as f:
    f.write(tflite_model)

print('Model is converted to TensorFlow Lite and saved.')