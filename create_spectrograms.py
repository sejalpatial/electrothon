import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Converts .wav audio to a mel-spectrogram image
# - input_folder: path to folders with class subfolders containing wav files
# - output_folder: where spectrogram png images will be saved

def generate_spectrogram(wav_path, output_path, sr=22050, n_mels=128, hop_length=512):
    y, sr = librosa.load(wav_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def crawl_and_save(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    classes = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]

    for label in classes:
        src_dir = os.path.join(input_root, label)
        dest_dir = os.path.join(output_root, label)
        os.makedirs(dest_dir, exist_ok=True)

        for f in os.listdir(src_dir):
            if f.lower().endswith('.wav'):
                wav_path = os.path.join(src_dir, f)
                out_name = os.path.splitext(f)[0] + '.png'
                output_path = os.path.join(dest_dir, out_name)
                print(f"Generating spectrogram: {wav_path} -> {output_path}")
                generate_spectrogram(wav_path, output_path)


if __name__ == '__main__':
    dataset_folder = os.path.abspath('dataset')
    output_folder = os.path.abspath('preprocessing/spectrograms')
    crawl_and_save(dataset_folder, output_folder)
    print('Spectrogram generation complete.')
