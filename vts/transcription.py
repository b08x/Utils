import os
import sys
import progressbar
from deepgram import DeepgramClient, PrerecordedOptions, FileSource, DeepgramError
from groq import Groq

parent_directory = os.path.abspath('..')

sys.path.append(parent_directory)


from vts import (
    audio_processing,
    metadata_generation,
    segment_analysis,
    topic_modeling,
    transcription,
    utils,
)

def transcribe_file_deepgram(client, file_path, options, max_retries=3, retry_delay=5):
    print("Transcribing audio using Deepgram...")
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as audio:
                buffer_data = audio.read()
                payload: FileSource = {"buffer": buffer_data, "mimetype": "audio/mp4"}
                response = client.listen.rest.v("1").transcribe_file(payload, options)
            print("Transcription complete.")
            return json.loads(response.to_json())
        except DeepgramError as e:
            if attempt < max_retries - 1:
                print(
                    f"API call failed. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_delay)
            else:
                print(f"Transcription failed after {max_retries} attempts: {str(e)}")
                raise
        except Exception as e:
            print(f"Unexpected error during transcription: {str(e)}")
            raise


def transcribe_file_groq(
    client, file_path, model="whisper-large-v3", language="en", prompt=None
):
    print("Transcribing audio using Groq...")
    try:
        with open(file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(file_path, file.read()),
                model=model,
                prompt=prompt,
                response_format="verbose_json",
                language=language,
                temperature=0.2
            )
        print("Transcription complete.")
        return json.loads(transcription.text)
    except Exception as e:
        print(f"Error during Groq transcription: {str(e)}")
        raise