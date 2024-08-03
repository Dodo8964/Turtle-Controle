import tensorflow as tf
import numpy as np

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def preprocess_audio(audio):
    spectrogram = get_spectrogram(audio)
    return spectrogram

def predict(model, audio):
    spectrogram = preprocess_audio(audio)
    spectrogram = tf.expand_dims(spectrogram, 0)  # Add batch dimension
    prediction = model(spectrogram)
    predicted_label = tf.argmax(prediction, axis=-1)
    return predicted_label.numpy()[0]

def decode_prediction(predicted_label, label_names):
    return label_names[predicted_label]
