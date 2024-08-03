import time
import numpy as np
import pyaudio
from audio_processing import predict, decode_prediction, LABEL_NAMES
from turtle_control import setup_turtle, move_turtle
import tensorflow as tf

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()

def record_audio():
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    frames = []
    seconds = 1
    for _ in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    
    return np.frombuffer(b''.join(frames), dtype=np.int16)

def terminate():
    p.terminate()

def record_and_move(model, t):
    while True:
        audio = record_audio()
        predicted_label = predict(model, audio)
        predicted_command = decode_prediction(predicted_label, LABEL_NAMES)
        print(f"Predicted command: {predicted_command}")
        
        if not move_turtle(t, predicted_command):
            break
        time.sleep(0.5)  # Add a small delay to avoid processing too rapidly

    terminate()

if __name__ == "__main__":
    model = tf.keras.models.load_model('models/my_model.keras')  # Adjust the path to your model
    t, wn = setup_turtle()
    record_and_move(model, t)
    wn.mainloop()
