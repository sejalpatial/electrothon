import os
from gtts import gTTS


def text_to_speech(explanation_text, output_path='response.mp3', lang='en'):
    if not explanation_text.strip():
        raise ValueError('Explanation text is empty')

    tts = gTTS(text=explanation_text, lang=lang)
    tts.save(output_path)
    print(f"Audio response saved to {output_path}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python speak_answer.py "<explanation text>"')
        sys.exit(1)

    explanation = sys.argv[1]
    text_to_speech(explanation)
