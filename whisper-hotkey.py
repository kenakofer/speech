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
recording = False
frames = []
stream = None
model_name = "small"
sample_rate = 16000

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
    global whisper, model
    import whisper
    print("Loading Whisper model:", model_name)
    notify("Whisper STT", f"Loading {model_name} model...")
    model = whisper.load_model(model_name)
    print("Model loaded successfully")
    notify("Whisper STT", "Model loaded successfully")

def callback(indata, frames, time, status):
    """This is called for each audio block"""
    if recording:
        frames.append(indata.copy())

def start_recording():
    global recording, stream, frames
    
    # Reset frames
    frames = []
    
    # Start recording
    recording = True
    notify("Whisper STT", "Recording... (release key to process)")
    
    # Start the audio stream
    stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=callback)
    stream.start()

def stop_recording_and_transcribe():
    global recording, stream, frames, model
    
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
    audio_data = np.concatenate(frames, axis=0)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        temp_filename = tmp_file.name
        np.save(temp_filename, audio_data)
    
    # Convert numpy file to wav using ffmpeg
    wav_filename = temp_filename + '.wav'
    subprocess.run(['ffmpeg', '-y', '-f', 'f32le', '-ar', str(sample_rate), 
                  '-ac', '1', '-i', temp_filename, wav_filename], 
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    try:
        # Transcribe
        result = model.transcribe(wav_filename)
        transcribed_text = result["text"].strip()
        
        # Insert the text
        insert_text(transcribed_text)
        
        # Show notification
        notify("Whisper STT", f"Inserted: {transcribed_text[:50]}{'...' if len(transcribed_text) > 50 else ''}")
    except Exception as e:
        notify("Whisper STT", f"Error: {str(e)}")
    
    # Clean up
    os.remove(temp_filename)
    os.remove(wav_filename)

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

def main():
    global model_name, TRIGGER_KEY
    
    parser = argparse.ArgumentParser(description="Hold-to-record speech recognition with Whisper")
    parser.add_argument('--model', default='small', help='Model size to use (tiny, base, small, medium, large)')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate for recording')
    parser.add_argument('--key', default='z', help='Key to hold for recording (single character, e.g., z, a, s)')
    args = parser.parse_args()
    
    # Set global variables
    model_name = args.model
    
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