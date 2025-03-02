#!/usr/bin/env python3
"""Minimal Whisper Hotkey - Press F13 to record, release to transcribe"""
import os
import subprocess
import time
import threading
from pynput import keyboard
import sounddevice as sd
import numpy as np

class SimpleWhisperHotkey:
    def __init__(self):
        self.sample_rate = 16000
        self.recording = False
        self.frames = []
        self.model = None
        self.trigger_key = keyboard.KeyCode(vk=269025153)  # F13 key
        self.processing_lock = threading.Lock()

        # Load model in background
        print("Loading Whisper model (small)...")
        threading.Thread(target=self.load_model, daemon=True).start()
        
        print("Whisper Hotkey ready: Press and hold F13 to record")
        
    def load_model(self):
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=2)
            print("Model loaded!")
        except ImportError:
            print("Error: faster-whisper not installed")
            print("Install with: pip install faster-whisper")
            exit(1)
    
    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.frames.append(indata.copy())
    
    def start_recording(self):
        if self.processing_lock.locked():
            return
            
        self.frames = []
        self.recording = True
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                dtype='float32'
            )
            self.stream.start()
            print("Recording...")
        except Exception as e:
            print(f"Error starting audio: {e}")
            self.recording = False
    
    def process_audio(self):
        with self.processing_lock:
            try:
                # Process audio data
                audio_data = np.concatenate(self.frames, axis=0)
                
                # Save to temporary WAV
                import soundfile as sf
                os.makedirs(os.path.expanduser("~/speech"), exist_ok=True)
                wav_path = os.path.expanduser("~/speech/recording.wav")
                sf.write(wav_path, audio_data, self.sample_rate)
                
                # Wait for model to load if necessary
                while self.model is None:
                    print("Waiting for model to load...")
                    time.sleep(0.5)
                
                # Transcribe
                segments, info = self.model.transcribe(
                    wav_path,
                    language="en",
                    beam_size=5,
                    vad_filter=True
                )
                
                # Collect text from segments
                text = ""
                for segment in segments:
                    text += segment.text
                
                text = text.strip()
                
                if text:
                    print(f"Transcribed: {text}")
                    subprocess.run(['xdotool', 'type', text], check=True)
                else:
                    print("No speech detected")
                    
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    def stop_recording(self):
        if not self.recording:
            return
            
        self.recording = False
        if hasattr(self, 'stream') and self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
        if not self.frames:
            print("No audio recorded")
            return
            
        print("Processing...")
        
        # Process in separate thread to keep UI responsive
        threading.Thread(target=self.process_audio, daemon=True).start()
    
    def on_press(self, key):
        if key == self.trigger_key and not self.recording and not self.processing_lock.locked():
            self.start_recording()
    
    def on_release(self, key):
        if key == self.trigger_key and self.recording:
            self.stop_recording()
    
    def run(self):
        # Check for required tools
        try:
            subprocess.run(['which', 'xdotool'], check=True)
        except subprocess.CalledProcessError:
            print("Error: xdotool not installed")
            print("Install with: sudo apt install xdotool")
            return
        
        # Start non-blocking keyboard listener
        listener = keyboard.Listener(
            on_press=self.on_press, 
            on_release=self.on_release
        )
        listener.start()
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Exiting...")

if __name__ == "__main__":
    app = SimpleWhisperHotkey()
    app.run()