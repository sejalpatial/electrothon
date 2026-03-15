import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint


def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_spectrogram_model(spectrogram_dir='preprocessing/spectrograms', model_path='model/engine_sound_model.h5'):
    img_size = (128, 128)
    batch_size = 16

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        spectrogram_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        spectrogram_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    model = build_model((img_size[0], img_size[1], 3), train_generator.num_classes)

    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)

    model.fit(
        train_generator,
        epochs=15,
        validation_data=val_generator,
        callbacks=[checkpoint]
    )

    model.save(model_path)
    # save label ordering from train_generator for inference
    label_map_path = os.path.join(os.path.dirname(model_path), 'label_map.json')
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(list(train_generator.class_indices.keys()), f)

    print(f"Model saved to {model_path}")
    print(f"Class labels saved to {label_map_path}")


if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    train_spectrogram_model()
