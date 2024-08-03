import tensorflow as tf
import numpy as np

LABEL_NAMES = np.array(['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes'])

def load_audio_file(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_spectrogram(waveform):
    waveform = tf.cast(waveform, tf.float32)
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

def decode_prediction(predicted_label, label_names=LABEL_NAMES):
    return label_names[predicted_label]
