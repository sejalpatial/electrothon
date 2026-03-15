import os
from predict_fault import predict_from_audio
from llm.explain_fault import explain_fault
from text_to_speech.speak_answer import text_to_speech


def run_full_pipeline():
    audio_file = 'audio_input/engine_sound.wav'

    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found at '{audio_file}'")

    print('1. Predicting engine fault...')
    detected_fault = predict_from_audio(audio_file)
    print(f'Detected Fault: {detected_fault}')

    print('2. Asking LLM to explain detected fault...')
    explanation_text = explain_fault(detected_fault)
    print('Explanation:')
    print(explanation_text)

    print('3. Converting explanation to speech...')
    os.makedirs('text_to_speech', exist_ok=True)
    text_to_speech(explanation_text, output_path='response.mp3')
    print('Audio response generated: response.mp3')


if __name__ == '__main__':
    run_full_pipeline()
