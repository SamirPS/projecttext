from moviepy import * # type: ignore
from pydub import AudioSegment # type: ignore
import os
import requests # type: ignore
import whisper # type: ignore
from datetime import timedelta
import ssl
import certifi # type: ignore
import json
import subprocess

def extract_audio_from_video(video_path, audio_output_path):
    # Extract audio from video and save as MP3
    video = VideoFileClip(video_path) # type: ignore
    audio = video.audio
    audio.write_audiofile(audio_output_path, codec='mp3')
    video.close()

def convert_mp3_to_wav(mp3_path, wav_path):
    # Convert MP3 to WAV format
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

def transcribe_audio_to_text_with_timestamps(audio_path, model_size='medium', verify_ssl=False):
    try:
        # Load Whisper model
        # Set SSL context if verification is an issue
        if not verify_ssl:
            # Create unverified context
            ssl._create_default_https_context = ssl._create_unverified_context
        else:
            # Use the default verified context with certifi
            ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
        
        # Load Whisper model
        model = whisper.load_model(model_size)
        
        # Transcribe audio
        result = model.transcribe(audio_path)
        
        # Format results to match your existing structure
        transcript_with_timestamps = []
        
        for segment in result['segments']:
            transcript_with_timestamps.append(
                (segment['text'], (segment['start'], segment['end']))
            )
        
        return transcript_with_timestamps
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return [(f"Error during transcription: {str(e)}", (0, 0))], []

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    return str(timedelta(seconds=int(seconds))).zfill(8)

def deformat_timestamp(timestamp):
    """Convert HH:MM:SS format to seconds"""
    h, m, s = map(int, timestamp.split(':'))
    return h * 3600 + m * 60 + s

def get_viral_moments_from_mistral(transcript_with_timestamps):
    """
    Analyze transcript using local Mistral model via Ollama to identify viral moments
    """
    # Formater les transcriptions pour le prompt
    formatted_transcript = ""
    for text, (start, end) in transcript_with_timestamps:
        start_time = format_timestamp(start)
        end_time = format_timestamp(end)    
        formatted_transcript += f"[{start_time} - {end_time}] {text}\n"
    
    prompt = f"""Tu es un expert en analyse de contenu viral sur les réseaux sociaux. Voici la transcription d'une vidéo avec horodatages :

    {formatted_transcript}

    Identifie les moments ayant le plus fort potentiel viral selon ces critères :
    - Émotions fortes (rire, surprise, tension)
    - Phrases accrocheuses ou citations marquantes
    - Points culminants ou révélations
    - Conflits ou résolutions
    - Changements soudains de ton ou d'ambiance
    - Humour ou moments inattendus

    Important :
    1. Les segments doivent durer entre 10 secondes et 2 minutes 40 secondes.
    2. Privilégie la qualité à la quantité
    3. Les moments doivent susciter des réactions fortes sur les réseaux sociaux.
    4. Tu ne dois couper des phrases ou mots.
    5. La fin de la vidéo doit être un end time du timestamp fourni.
    6. CRITIQUE : Les moments viraux DOIVENT commencer et finir exactement aux timestamps de la transcription fournie. 
       Tu ne peux PAS choisir des timestamps qui coupent au milieu d'un segment de transcription.
       Utilise UNIQUEMENT les timestamps de début et de fin qui sont fournis dans la transcription.
       Aussi ne pas couper des phrases ou mots, la phrase doit finir sur un point de suspension ou un point.

    Retourne STRICTEMENT une liste de timestamps au format suivant, un par ligne :
    START_TIME|END_TIME

    Exemple :
    00:00:25|00:00:35
    00:01:05|00:01:20

    N'ajoute AUCUN texte explicatif, UNIQUEMENT les timestamps."""


    try:
        response = requests.post('http://localhost:11434/api/generate', 
                               json={
                                   "model": "mixtral",
                                   "prompt": prompt,
                                   "stream": False
                               })
        
        if response.status_code == 200:
            response_text = response.json()['response'].strip()
            viral_moments = []
            
            for line in response_text.split('\n'):
                if '|' in line:
                    start_time, end_time = line.strip().split('|')
                    end_time = end_time.split(" ")[0]
                    if start_time and end_time:
                        viral_moments.append((start_time, end_time))
            
            return viral_moments if viral_moments else [("00:00:10", "00:00:20")]
        else:
            print(f"Erreur lors de l'appel à Ollama: {response.status_code}")
            return [("00:00:10", "00:00:20")]

    except Exception as e:
        print(f"Erreur lors de l'appel à Ollama: {str(e)}")
        return [("00:00:10", "00:00:20")]


def validate_timestamps(start_time, end_time, video_duration):
    """Validate and adjust timestamps to ensure they don't exceed video duration"""
    start_seconds = deformat_timestamp(start_time)
    end_seconds = deformat_timestamp(end_time)
    
    # Ensure timestamps don't exceed video duration
    if start_seconds >= video_duration:
        start_seconds = max(0, video_duration - 30)  # Default to last 30 seconds if start is beyond duration
    if end_seconds > video_duration:
        end_seconds = video_duration
    
    # Ensure start is before end
    if start_seconds >= end_seconds:
        start_seconds = max(0, end_seconds - 30)  # Default to 30 second clip
    
    return format_timestamp(start_seconds), format_timestamp(end_seconds)

def extract_viral_moment(video_path, start_timecode, end_timecode, output_path):
    # Load video to get duration
    video = VideoFileClip(video_path) # type: ignore
    
    # Validate and adjust timestamps
    start_timecode, end_timecode = validate_timestamps(start_timecode, end_timecode, video.duration)
    
    # Extract video segment
    viral_clip = video.subclipped(start_timecode, end_timecode)
    
    
    # Save extracted clip with subtitles
    viral_clip.write_videofile(output_path, codec="libx264")
    
    # Clean up
    viral_clip.close()
    video.close()

def main(video_path, output_folder, model_size='medium'):
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Define paths for temporary audio files
        mp3_path = os.path.join(output_folder, "audio_output.mp3")
        wav_path = os.path.join(output_folder, "audio_output.wav")
        
        # Extract and convert audio
        print("Extracting audio from video...")
        extract_audio_from_video(video_path, mp3_path)
        
        print("Converting audio to WAV format...")
        convert_mp3_to_wav(mp3_path, wav_path)
        
        print("Transcribing audio with Whisper...")
        transcript_with_timestamps = transcribe_audio_to_text_with_timestamps(wav_path, model_size)
        
        print("\nTranscription with timestamps:")
            
        print("\nWord-level timestamps:")

        print("\nIdentifying viral moments using local Mistral model...")
        viral_moments = get_viral_moments_from_mistral(transcript_with_timestamps)
        

        for index, (start_timecode, end_timecode) in enumerate(viral_moments):
            output_video_path = os.path.join(output_folder, f"viral_moment.mp4")
            extract_viral_moment(video_path, start_timecode, end_timecode, output_video_path)
            print(f"Viral video {index + 1} saved to {output_video_path}")
            try:
                command = f"cd test && node sub.mjs && npx remotion render src/index.ts CaptionedVideo ../output/video{index + 1}.mp4 && rm -rf public/viral_moment.json && rm -rf public/viral_moment.mp4"
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    shell=True
                )
                print("Subprocess output:", result.stdout)
                if result.stderr:
                    print("Subprocess errors:", result.stderr)
            except Exception as e:
                print(f"Error running subprocess: {str(e)}")

        # Clean up temporary audio files
        os.remove(mp3_path)
        os.remove(wav_path)
        
        print("\nProcess completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    video_path = "test.mp4"
    output_folder = "test/public/"
    main(video_path, output_folder, model_size='medium')