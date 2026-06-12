# Battlefield Audio Classification System

## Overview

This project presents an AI-based audio classification system designed to identify battlefield-related sounds from audio recordings. The system classifies sounds such as gunshots, military vehicles, helicopters, and environmental noise using deep learning techniques.

Audio signals are transformed into Mel spectrogram representations and processed by a Convolutional Neural Network (CNN) for feature learning and classification. The trained model achieved approximately 92% validation accuracy on the evaluation dataset.

---

## Problem Statement

In military and surveillance environments, rapid identification of critical audio events can support situational awareness and decision-making. Manual monitoring of large volumes of audio data is time-consuming and prone to error.

This project aims to automate the detection and classification of battlefield sounds using machine learning and audio signal processing techniques.

---

## Objectives

- Develop an automated audio classification system for battlefield environments.
- Extract meaningful features from audio signals using Mel spectrograms.
- Train a Convolutional Neural Network (CNN) to classify different sound categories.
- Evaluate model performance on unseen audio samples.

---

## Methodology

### Data Preprocessing

The audio dataset was processed to ensure consistency in format and sampling rate. Audio clips were converted into Mel spectrograms, which provide a visual representation of sound frequencies over time.

### Feature Extraction

Mel spectrograms were used as input features because they effectively capture frequency characteristics relevant to sound classification tasks.

### Model Development

A Convolutional Neural Network (CNN) was designed and trained on the generated spectrograms. The network learns spatial patterns within the spectrogram images that correspond to different sound categories.

### Model Evaluation

The trained model was evaluated using validation data to measure classification performance and generalization capability.

---

## Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

---

## Sound Categories

The system is capable of classifying the following categories:

- Gunshots
- Military Vehicles
- Helicopters
- Environmental Noise

Additional sound categories can be incorporated by extending the training dataset.

---

## Results

| Metric               | Value                              |
| -------------------- | ---------------------------------- |
| Model Architecture   | Convolutional Neural Network (CNN) |
| Input Representation | Mel Spectrogram                    |
| Validation Accuracy  | ~92%                               |
| Classification Type  | Multi-Class Audio Classification   |

---

## Project Structure

```text
electrothon/
│
├── data/
├── models/
├── notebooks/
├── src/
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/sejalpatial/electrothon.git
cd electrothon
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

If the project uses Streamlit:

```bash
streamlit run app.py
```

Alternatively:

```bash
python app.py
```

---

## Future Enhancements

- Real-time battlefield audio monitoring.
- Integration with edge devices for field deployment.
- Expansion of sound categories and training dataset.
- Cloud-based deployment for remote monitoring.
- Improved robustness in noisy environments.

---

## Applications

- Defense and military surveillance
- Border security monitoring
- Automated threat detection systems
- Remote acoustic monitoring
- Smart battlefield awareness systems

---

## Authors

Developed as part of the Electrothon project by **Sejal Patial and Team**.
