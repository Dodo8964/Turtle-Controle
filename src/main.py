import pyaudio
import numpy as np
from audio_processing import load_model, predict, decode_prediction
from turtle_control import setup_turtle, move_turtle

# Constants
MODEL_PATH = 'models/my_model.keras'
LABEL_NAMES = ["move_forward", "move_backward", "turn_left", "turn_right", "stop"]

# Initialize model
model = load_model(MODEL_PATH)

# Initialize Turtle
t = setup_turtle()

def record_audio(duration=1, fs=16000):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        frames_per_buffer=3200
    )

    frames = []
    for _ in range(0, int(fs / 3200 * duration)):
        data = stream.read(3200)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    return np.frombuffer(b''.join(frames), dtype=np.int16)

def record_and_move():
    audio = record_audio(duration=1, fs=16000)  # Record 1 second of audio
    predicted_label = predict(model, audio)
    predicted_command = decode_prediction(predicted_label, LABEL_NAMES)
    print(f"Predicted command: {predicted_command}")
    move_turtle(t, predicted_command)

if __name__ == "__main__":
    record_and_move()
