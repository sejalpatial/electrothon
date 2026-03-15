import os
import json
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def create_temp_spectrogram(wav_path, temp_img_path, sr=22050, n_mels=128, hop_length=512):
    y, sr = librosa.load(wav_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def predict_from_audio(wav_path, model_path='model/engine_sound_model.h5'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = load_model(model_path)
    temp_img = 'temp_prediction.png'
    create_temp_spectrogram(wav_path, temp_img)

    img = load_img(temp_img, target_size=(128, 128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_idx = np.argmax(preds, axis=1)[0]

    label_map_path = os.path.join('model', 'label_map.json')
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r', encoding='utf-8') as f:
            class_labels = json.load(f)
    else:
        class_labels = ['normal_engine', 'engine_knocking', 'loose_belt', 'bearing_noise']

    predicted_label = class_labels[class_idx]

    if os.path.exists(temp_img):
        os.remove(temp_img)

    return predicted_label


if __name__ == '__main__':
    audio_file = 'audio_input/engine_sound.wav'
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Input audio not found: {audio_file}")

    print('Predicting engine sound fault...')
    label = predict_from_audio(audio_file)
    print('Detected Fault:', label)
