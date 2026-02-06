import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise'
}


def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

X = []
y = []

dataset_path = "dataset"

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            emotion = emotion_map[emotion_code]
            file_path = os.path.join(root, file)
            features = extract_features(file_path)
            X.append(features)
            y.append(emotion)

X = np.array(X)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X = X.reshape(X.shape[0], X.shape[1], 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = Sequential()

model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(40,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(LSTM(64))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(np.unique(y)), activation='softmax'))


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

model.save("speech_emotion_model.keras")






##frontend
import PySimpleGUI as sg
import sounddevice as sd
import soundfile as sf
import tempfile
import numpy as np

# Function to record audio from microphone
def record_audio(duration=3, fs=22050):
    sg.popup_quick_message(f"Recording for {duration} seconds...", auto_close=True, non_blocking=True)
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    audio = np.squeeze(audio)
    
    # Save to temporary WAV file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio, fs)
    return temp_file.name






layout = [
    [sg.Text("Speech Emotion Recognition", font=("Helvetica", 16))],
    [sg.Text("OR select a WAV file:"), sg.Input(key="-FILE-"), sg.FileBrowse(file_types=(("WAV Files", "*.wav"),))],
    [sg.Button("Predict Emotion from File"), sg.Button("Record & Predict Emotion"), sg.Button("Exit")],
    [sg.Text("Predicted Emotion: ", key="-OUTPUT-", font=("Helvetica", 14), text_color="blue")]
]

window = sg.Window("Emotion Recognition", layout)


def predict_emotion(file_path):
    features = extract_features(file_path)
    features = features.reshape(1, 40, 1)
    prediction = model.predict(features)
    emotion = label_encoder.inverse_transform([np.argmax(prediction)])
    return emotion[0]




while True:
    event, values = window.read()
    
    if event == sg.WINDOW_CLOSED or event == "Exit":
        break

    # Predict from selected file
    if event == "Predict Emotion from File":
        file_path = values["-FILE-"]
        if file_path:
            try:
                emotion = predict_emotion(file_path)  # your existing function
                window["-OUTPUT-"].update(f"Predicted Emotion: {emotion.upper()}")
            except Exception as e:
                sg.popup_error(f"Error: {e}")
        else:
            sg.popup("Please select a WAV file!")

    # Record audio from mic and predict
    if event == "Record & Predict Emotion":
        try:
            temp_file = record_audio(duration=3)  # 3-second recording
            emotion = predict_emotion(temp_file)  # your existing function
            window["-OUTPUT-"].update(f"Predicted Emotion: {emotion.upper()}")
        except Exception as e:
            sg.popup_error(f"Error: {e}")






window.close()

