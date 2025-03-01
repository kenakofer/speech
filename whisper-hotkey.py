#!/usr/bin/env python3
import sounddevice as sd
import numpy as np
import subprocess
import tempfile
import os
import sys
import threading
import time
import signal
import argparse
from pynput import keyboard
from pathlib import Path

# Global variables
TRIGGER_KEY = keyboard.KeyCode.from_char('z')  # Use lowercase 'z' as the default key
recording = False
frames = []
stream = None
model_name = "small"
sample_rate = 16000
use_faster_whisper = False  # Whether to use faster_whisper instead of standard whisper

def notify(title, message):
    subprocess.run(['notify-send', title, message])
    print(f"{title}: {message}")

def insert_text(text):
    """Simulate keyboard typing to insert text at cursor position"""
    # Copy to clipboard first
    process = subprocess.Popen(['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE)
    process.communicate(input=text.encode())
    
    # Then simulate Ctrl+V to paste
    subprocess.run(['xdotool', 'key', 'ctrl+v'])

def load_whisper():
    """Load whisper in a separate thread to avoid lag on first use"""
    global model, use_faster_whisper
    
    if use_faster_whisper:
        try:
            from faster_whisper import WhisperModel
            print("Loading Faster Whisper model:", model_name)
            notify("Whisper STT", f"Loading {model_name} model (faster-whisper)...")
            # Use CPU with 2 threads for better performance
            model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=2)
            print("Model loaded successfully")
            notify("Whisper STT", "Model loaded successfully")
        except ImportError:
            print("faster-whisper not installed, falling back to standard whisper")
            use_faster_whisper = False
            import whisper
            print("Loading Whisper model:", model_name)
            notify("Whisper STT", f"Loading {model_name} model...")
            model = whisper.load_model(model_name)
            print("Model loaded successfully")
            notify("Whisper STT", "Model loaded successfully")
    else:
        import whisper
        print("Loading Whisper model:", model_name)
        notify("Whisper STT", f"Loading {model_name} model...")
        model = whisper.load_model(model_name)
        print("Model loaded successfully")
        notify("Whisper STT", "Model loaded successfully")

def callback(indata, frame_count, time, status):
    """This is called for each audio block"""
    global frames
    if status:
        print(f"Audio callback status: {status}")
    if recording:
        try:
            frames.append(indata.copy())
        except Exception as e:
            print(f"Error in audio callback: {e}")
            print(f"Type of frames: {type(frames)}, indata: {type(indata)}")

def start_recording():
    global recording, stream, frames
    
    # Reset frames
    frames = []
    
    # Start recording
    recording = True
    notify("Whisper STT", "Recording... (release key to process)")
    
    # Print available devices for debugging
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    
    try:
        # Start the audio stream with default input device
        stream = sd.InputStream(
            samplerate=sample_rate, 
            channels=1, 
            callback=callback,
            blocksize=1024,
            dtype='float32',
            latency='low'
        )
        stream.start()
        print("Recording started successfully")
    except Exception as e:
        print(f"Error starting audio stream: {e}")
        recording = False

def stop_recording_and_transcribe():
    global recording, stream, frames, model, use_faster_whisper
    
    if not recording:
        return
    
    # Stop recording
    recording = False
    
    if stream:
        stream.stop()
        stream.close()
    
    if not frames:
        notify("Whisper STT", "No audio recorded")
        return
    
    notify("Whisper STT", "Processing audio...")
    
    # Convert frames to numpy array
    try:
        if len(frames) == 0:
            notify("Whisper STT", "No audio frames captured")
            return
            
        print(f"Number of audio frames captured: {len(frames)}")
        audio_data = np.concatenate(frames, axis=0)
        print(f"Audio data shape: {audio_data.shape}")
        
        # Check if audio data contains silence
        audio_mean = np.abs(audio_data).mean()
        print(f"Audio mean amplitude: {audio_mean}")
        if audio_mean < 0.001:
            print("WARNING: Audio appears to be mostly silence")
        
        # Create direct path for WAV file
        debug_wav = os.path.expanduser("~/speech/last_recording.wav")
        
        # First try using the soundfile library which is more reliable
        try:
            import soundfile as sf
            sf.write(debug_wav, audio_data, sample_rate)
            print(f"Saved WAV using soundfile: {debug_wav}")
            wav_filename = debug_wav
        except ImportError:
            try:
                # Fallback to scipy
                from scipy.io import wavfile
                
                # Ensure audio_data is in the right format (normalized to int16)
                normalized_audio = np.int16(audio_data * 32767)
                wavfile.write(debug_wav, sample_rate, normalized_audio)
                print(f"Saved WAV using scipy: {debug_wav}")
                wav_filename = debug_wav
                
            except ImportError:
                print("Using ffmpeg for WAV conversion")
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp_file:
                    temp_filename = tmp_file.name
                    audio_data.tofile(temp_filename)
                    print(f"Saved raw audio data to {temp_filename}")
                
                # Convert raw file to wav using ffmpeg
                result = subprocess.run([
                    'ffmpeg', '-y', 
                    '-f', 'f32le', 
                    '-ar', str(sample_rate), 
                    '-ac', '1', 
                    '-i', temp_filename, 
                    debug_wav
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"Error converting to WAV: {result.stderr}")
                    return
                
                print(f"Converted to WAV using ffmpeg: {debug_wav}")
                wav_filename = debug_wav
                
        # Install soundfile for better reliability
        subprocess.run(['pip', 'install', '--user', 'soundfile'], capture_output=True)
                
        # Try playing back the audio to verify it worked
        try:
            print("\nPlaying back the recorded audio for verification...")
            subprocess.run(["aplay", wav_filename])
        except Exception as e:
            print(f"Could not play audio: {e}")
    except Exception as e:
        print(f"Error processing audio data: {e}")
        return
    
    try:
        # Validate the WAV file before proceeding
        import wave
        try:
            with wave.open(wav_filename, 'rb') as wf:
                print(f"WAV file info: {wf.getparams()}")
                frames = wf.getnframes()
                if frames == 0:
                    print("WARNING: WAV file has 0 frames")
                    notify("Whisper STT", "Invalid audio recording. Please try again.")
                    return
        except Exception as e:
            print(f"Error validating WAV file: {e}")
            # Continue anyway, maybe it will work
        
        # WAV file has already been saved at debug_wav location
        print(f"Using WAV file at: {wav_filename}")
        
        # Let's use sox to check the audio
        try:
            result = subprocess.run(['soxi', wav_filename], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Audio file details:\n{result.stdout}")
            else:
                print("Sox not available or error checking audio file")
        except:
            pass  # Sox might not be installed
        
        # Transcribe
        print("Starting transcription...")
        
        if use_faster_whisper:
            # faster-whisper has a different API
            try:
                # Try with different parameters to increase chances of success
                segments, info = model.transcribe(
                    debug_wav,
                    language="en", 
                    beam_size=5,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
                print(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
                
                # Collect all segments
                transcribed_text = ""
                segments_list = list(segments)  # Convert generator to list
                print(f"Number of segments: {len(segments_list)}")
                
                for segment in segments_list:
                    print(f"Segment {segment.id}: '{segment.text}' ({segment.start:.2f}s - {segment.end:.2f}s)")
                    transcribed_text += segment.text
                
                transcribed_text = transcribed_text.strip()
            except Exception as e:
                print(f"Error with faster-whisper: {e}")
                # Fall back to standard whisper
                use_faster_whisper = False
                import whisper
                model = whisper.load_model(model_name)
                print("Falling back to standard whisper")
        
        if not use_faster_whisper:
            # Standard whisper
            # Try with different parameters to increase chances of success
            result = model.transcribe(
                debug_wav, 
                language="en", 
                task="transcribe", 
                fp16=False
            )
            transcribed_text = result["text"].strip()
            
        print("Transcription complete")
        
        print(f"Raw transcription: '{transcribed_text}'")
        
        if not transcribed_text:
            print("WARNING: Transcription returned empty string")
            notify("Whisper STT", "No speech detected. Try speaking louder or check mic.")
            
            # Try playing back the audio to check if it's audible
            try:
                print("Attempting to play back the audio for verification...")
                subprocess.run(["aplay", debug_wav])
            except Exception as e:
                print(f"Could not play audio: {e}")
                
            return
            
        # Insert the text
        insert_text(transcribed_text)
        
        # Show notification
        notify("Whisper STT", f"Inserted: {transcribed_text[:50]}{'...' if len(transcribed_text) > 50 else ''}")
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        notify("Whisper STT", f"Error: {str(e)}")
        return
    
    # Clean up (but keep the debug copy)
    try:
        # Clean up temp files, but not the wav_filename which is our debug copy
        if 'temp_filename' in locals():
            os.remove(temp_filename)
    except Exception as e:
        print(f"Error cleaning up: {e}")

def on_press(key):
    global recording, TRIGGER_KEY
    try:
        # Check if it matches our trigger key
        if key == TRIGGER_KEY:
            if not recording:
                start_recording()
    except AttributeError:
        pass

def on_release(key):
    global TRIGGER_KEY
    try:
        # Check if it matches our trigger key
        if key == TRIGGER_KEY:
            if recording:
                stop_recording_and_transcribe()
    except AttributeError:
        pass

def test_microphone(duration=3):
    """Test microphone by recording a short clip and printing stats"""
    print(f"\nTesting microphone for {duration} seconds...")
    try:
        # Record audio
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        print("Recording...")
        sd.wait()
        print("Recording complete.")
        
        # Print stats
        if recording.size > 0:
            mean_amplitude = np.abs(recording).mean()
            max_amplitude = np.abs(recording).max()
            print(f"Recording shape: {recording.shape}")
            print(f"Mean amplitude: {mean_amplitude:.6f}")
            print(f"Max amplitude: {max_amplitude:.6f}")
            
            if mean_amplitude < 0.001:
                print("\033[91mWARNING: Audio levels very low. Microphone may not be working correctly.\033[0m")
            else:
                print("\033[92mMicrophone seems to be working.\033[0m")
        else:
            print("\033[91mERROR: No audio data captured.\033[0m")
    except Exception as e:
        print(f"\033[91mError testing microphone: {e}\033[0m")

def main():
    global model_name, TRIGGER_KEY, use_faster_whisper
    
    parser = argparse.ArgumentParser(description="Hold-to-record speech recognition with Whisper")
    parser.add_argument('--model', default='small', help='Model size to use (tiny, base, small, medium, large)')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate for recording')
    parser.add_argument('--key', default='z', help='Key to hold for recording (single character, e.g., z, a, s)')
    parser.add_argument('--test-mic', action='store_true', help='Test microphone before starting')
    parser.add_argument('--faster', action='store_true', help='Use faster-whisper implementation')
    parser.add_argument('--debug', action='store_true', help='Enable additional debug output')
    args = parser.parse_args()
    
    # Set global variables
    model_name = args.model
    use_faster_whisper = args.faster
    
    # Set the trigger key if specified
    if args.key:
        if len(args.key) == 1:
            TRIGGER_KEY = keyboard.KeyCode.from_char(args.key)
        else:
            try:
                # For function keys like f13, etc.
                TRIGGER_KEY = getattr(keyboard.Key, args.key)
            except AttributeError:
                print(f"Warning: Could not set key to {args.key}, using 'z' instead")
    
    # Make sure required packages are installed
    try:
        subprocess.run(['which', 'xclip', 'xdotool'], check=True)
    except subprocess.CalledProcessError:
        print("Error: Required packages not installed.")
        print("Please install with: sudo apt install xclip xdotool")
        return 1
    
    # Test microphone if requested
    if args.test_mic:
        test_microphone()
        print("\nContinuing with normal operation...\n")
    
    # Load Whisper model in a separate thread
    thread = threading.Thread(target=load_whisper)
    thread.daemon = True
    thread.start()
    
    key_display = args.key if len(args.key) == 1 else args.key.upper()
    print(f"Press and hold '{key_display}' key to record, release to transcribe")
    notify("Whisper STT", f"Press and hold '{key_display}' key to record, release to transcribe")
    
    # Keyboard listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            print("\nExiting...")

if __name__ == "__main__":
    main()