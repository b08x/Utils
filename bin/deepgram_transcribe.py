import os
import argparse
import time
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource, DeepgramError

load_dotenv()

def transcribe_file(client, file_path, options, max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as audio:
                buffer_data = audio.read()
                payload: FileSource = {
                    "buffer": buffer_data,
                    "mimetype": "audio/opus"  # Specify the correct mimetype for .opus files
                }
                response = client.listen.rest.v("1").transcribe_file(payload, options)
            return response
        except DeepgramError as e:
            if attempt < max_retries - 1:
                print(f"API call failed. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise e
        except Exception as e:
            raise e

def main():
    parser = argparse.ArgumentParser(description="Transcribe .opus files in a folder using Deepgram API.")
    parser.add_argument("folder_path", help="Path to the folder containing .opus files")
    args = parser.parse_args()

    try:
        # Retrieve API key from environment variable
        API_KEY = os.getenv("DG_API_KEY")
        if not API_KEY:
            raise ValueError("DG_API_KEY environment variable is not set")

        deepgram = DeepgramClient(API_KEY)

        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            language="en",
            punctuate=True,
            utterances=True,
            diarize=True,
        )

        for filename in os.listdir(args.folder_path):
            if filename.endswith(".opus"):
                file_path = os.path.join(args.folder_path, filename)
                print(f"Processing {filename}...")
                
                try:
                    response = transcribe_file(deepgram, file_path, options)
                    
                    # Save the transcription to a text file
                    output_filename = os.path.splitext(filename)[0] + "_transcription.txt"
                    output_path = os.path.join(args.folder_path, output_filename)
                    with open(output_path, "w") as output_file:
                        output_file.write(response.to_json(indent=4))
                    
                    print(f"Transcription saved to {output_filename}")
                
                except DeepgramError as e:
                    print(f"API Error while processing {filename}: {str(e)}")
                except IOError as e:
                    print(f"File I/O error while processing {filename}: {str(e)}")
                except Exception as e:
                    print(f"Unexpected error while processing {filename}: {str(e)}")

    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()
