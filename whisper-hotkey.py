#!/usr/bin/env python3
"""
Whisper Hold-to-Speak: A speech-to-text tool using OpenAI's Whisper.
Press and hold a key to record, release to insert the transcribed text.
"""
import sounddevice as sd
import numpy as np
import subprocess
import tempfile
import os
import sys
import threading
import time
import argparse
import logging
from pynput import keyboard
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union, Callable


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("whisper_hotkey")


class AudioRecorder:
    """Handles audio recording and processing."""
    
    def __init__(self, sample_rate: int = 16000):
        """Initialize the recorder with specified sample rate."""
        self.sample_rate = sample_rate
        self.recording = False
        self.frames: List[np.ndarray] = []
        self.stream = None
    
    def audio_callback(self, indata, frame_count, time_info, status):
        """Process audio data from the input stream."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        if self.recording:
            try:
                self.frames.append(indata.copy())
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")
                logger.debug(f"Type of frames: {type(self.frames)}, indata: {type(indata)}")
    
    def start(self) -> bool:
        """Start recording audio."""
        self.frames = []
        self.recording = True
        
        # Print available devices for debugging
        logger.debug("Available audio devices:")
        logger.debug(sd.query_devices())
        
        try:
            # Start the audio stream with default input device
            self.stream = sd.InputStream(
                samplerate=self.sample_rate, 
                channels=1, 
                callback=self.audio_callback,
                blocksize=1024,
                dtype='float32',
                latency='low'
            )
            self.stream.start()
            logger.info("Recording started successfully")
            return True
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            self.recording = False
            return False
    
    def stop(self) -> Optional[np.ndarray]:
        """Stop recording and return the audio data."""
        if not self.recording:
            return None
        
        # Stop recording
        self.recording = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        if not self.frames:
            logger.warning("No audio frames captured")
            return None
        
        # Process the recorded frames
        try:
            logger.info(f"Number of audio frames captured: {len(self.frames)}")
            audio_data = np.concatenate(self.frames, axis=0)
            logger.info(f"Audio data shape: {audio_data.shape}")
            
            # Check if audio data contains silence
            audio_mean = np.abs(audio_data).mean()
            logger.info(f"Audio mean amplitude: {audio_mean}")
            if audio_mean < 0.001:
                logger.warning("Audio appears to be mostly silence")
            
            return audio_data
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            return None
    
    def test_microphone(self, duration: int = 3) -> bool:
        """Test microphone by recording for a few seconds."""
        logger.info(f"Testing microphone for {duration} seconds...")
        try:
            # Record audio
            recording = sd.rec(
                int(duration * self.sample_rate), 
                samplerate=self.sample_rate, 
                channels=1, 
                dtype='float32'
            )
            logger.info("Recording test sample...")
            sd.wait()
            logger.info("Recording complete.")
            
            # Print stats
            if recording.size > 0:
                mean_amplitude = np.abs(recording).mean()
                max_amplitude = np.abs(recording).max()
                logger.info(f"Recording shape: {recording.shape}")
                logger.info(f"Mean amplitude: {mean_amplitude:.6f}")
                logger.info(f"Max amplitude: {max_amplitude:.6f}")
                
                if mean_amplitude < 0.001:
                    logger.warning("Audio levels very low. Microphone may not be working correctly.")
                    return False
                else:
                    logger.info("Microphone seems to be working.")
                    return True
            else:
                logger.error("No audio data captured.")
                return False
        except Exception as e:
            logger.error(f"Error testing microphone: {e}")
            return False


class AudioProcessor:
    """Handles audio file processing and conversion."""
    
    def __init__(self, debug_dir: str = None):
        """Initialize the processor with optional debug directory."""
        self.debug_dir = debug_dir or os.path.expanduser("~/speech")
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def save_to_wav(self, audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
        """Save audio data to WAV file using available methods."""
        debug_wav = os.path.join(self.debug_dir, "last_recording.wav")
        
        # First try using the soundfile library which is more reliable
        try:
            import soundfile as sf
            sf.write(debug_wav, audio_data, sample_rate)
            logger.info(f"Saved WAV using soundfile: {debug_wav}")
            return debug_wav
        except ImportError:
            try:
                # Fallback to scipy
                from scipy.io import wavfile
                
                # Ensure audio_data is in the right format (normalized to int16)
                normalized_audio = np.int16(audio_data * 32767)
                wavfile.write(debug_wav, sample_rate, normalized_audio)
                logger.info(f"Saved WAV using scipy: {debug_wav}")
                return debug_wav
                
            except ImportError:
                logger.info("Using ffmpeg for WAV conversion")
                try:
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp_file:
                        temp_filename = tmp_file.name
                        audio_data.tofile(temp_filename)
                        logger.debug(f"Saved raw audio data to {temp_filename}")
                    
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
                        logger.error(f"Error converting to WAV: {result.stderr}")
                        return None
                    
                    logger.info(f"Converted to WAV using ffmpeg: {debug_wav}")
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_filename)
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file: {e}")
                    
                    return debug_wav
                except Exception as e:
                    logger.error(f"Failed to convert audio to WAV: {e}")
                    return None
                
        # Install soundfile for better reliability on next run
        try:
            subprocess.run(['pip', 'install', '--user', 'soundfile'], capture_output=True)
        except Exception:
            pass
    
    def validate_wav_file(self, wav_path: str) -> bool:
        """Validate that a WAV file is properly formatted."""
        try:
            import wave
            with wave.open(wav_path, 'rb') as wf:
                params = wf.getparams()
                logger.debug(f"WAV file info: {params}")
                frames = wf.getnframes()
                if frames == 0:
                    logger.warning("WAV file has 0 frames")
                    return False
                return True
        except Exception as e:
            logger.error(f"Error validating WAV file: {e}")
            return False
    
    def play_audio(self, wav_path: str) -> bool:
        """Play back the audio file for verification."""
        try:
            logger.info(f"Playing back audio file: {wav_path}")
            subprocess.run(["aplay", wav_path])
            return True
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            return False


class NotificationManager:
    """Handles system notifications and feedback."""
    
    def notify(self, title: str, message: str) -> None:
        """Send a desktop notification and log the message."""
        subprocess.run(['notify-send', title, message])
        logger.info(f"{title}: {message}")
    
    def insert_text(self, text: str) -> bool:
        """Insert text at cursor position via clipboard and keyboard shortcut."""
        try:
            # Copy to clipboard first
            process = subprocess.Popen(['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE)
            process.communicate(input=text.encode())
            
            # Then simulate Ctrl+V to paste
            subprocess.run(['xdotool', 'key', 'ctrl+v'])
            return True
        except Exception as e:
            logger.error(f"Failed to insert text: {e}")
            return False


class TranscriptionEngine:
    """Handles speech-to-text transcription using various backends."""
    
    def __init__(self, model_name: str = "small", use_faster_whisper: bool = False):
        """Initialize the transcription engine."""
        self.model_name = model_name
        self.use_faster_whisper = use_faster_whisper
        self.model = None
    
    def load_model(self) -> bool:
        """Load the transcription model in a separate thread."""
        try:
            if self.use_faster_whisper:
                try:
                    from faster_whisper import WhisperModel
                    logger.info(f"Loading Faster Whisper model: {self.model_name}")
                    # Use CPU with 2 threads for better performance
                    self.model = WhisperModel(
                        self.model_name, 
                        device="cpu", 
                        compute_type="int8", 
                        cpu_threads=2
                    )
                    logger.info("Model loaded successfully")
                    return True
                except ImportError:
                    logger.warning("faster-whisper not installed, falling back to standard whisper")
                    self.use_faster_whisper = False
            
            # Standard whisper
            import whisper
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def transcribe(self, wav_path: str) -> Optional[str]:
        """Transcribe audio file to text."""
        if not self.model:
            logger.error("Model not loaded")
            return None
        
        try:
            logger.info("Starting transcription...")
            
            if self.use_faster_whisper:
                # faster-whisper has a different API
                try:
                    # Try with different parameters to increase chances of success
                    segments, info = self.model.transcribe(
                        wav_path,
                        language="en", 
                        beam_size=5,
                        word_timestamps=True,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                    )
                    logger.info(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
                    
                    # Collect all segments
                    transcribed_text = ""
                    segments_list = list(segments)  # Convert generator to list
                    logger.debug(f"Number of segments: {len(segments_list)}")
                    
                    for segment in segments_list:
                        logger.debug(f"Segment {segment.id}: '{segment.text}' ({segment.start:.2f}s - {segment.end:.2f}s)")
                        transcribed_text += segment.text
                    
                    transcribed_text = transcribed_text.strip()
                    return transcribed_text
                except Exception as e:
                    logger.error(f"Error with faster-whisper: {e}")
                    # Fall back to standard whisper
                    self.use_faster_whisper = False
                    import whisper
                    self.model = whisper.load_model(self.model_name)
                    logger.info("Falling back to standard whisper")
            
            if not self.use_faster_whisper:
                # Standard whisper
                # Try with different parameters to increase chances of success
                result = self.model.transcribe(
                    wav_path, 
                    language="en", 
                    task="transcribe", 
                    fp16=False
                )
                transcribed_text = result["text"].strip()
                return transcribed_text
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None


class WhisperHotkey:
    """Main application class coordinating all components."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the application with command line arguments."""
        self.args = args
        
        # Set up logging level
        if args.debug:
            logger.setLevel(logging.DEBUG)
        
        # Initialize components
        self.recorder = AudioRecorder(sample_rate=args.sample_rate)
        self.processor = AudioProcessor()
        self.notifier = NotificationManager()
        self.transcriber = TranscriptionEngine(
            model_name=args.model,
            use_faster_whisper=args.faster
        )
        
        # Set trigger key
        if len(args.key) == 1:
            self.trigger_key = keyboard.KeyCode.from_char(args.key)
        else:
            try:
                # For function keys like f13, etc.
                self.trigger_key = getattr(keyboard.Key, args.key)
            except AttributeError:
                logger.warning(f"Could not set key to {args.key}, using 'z' instead")
                self.trigger_key = keyboard.KeyCode.from_char('z')
    
    def setup(self) -> bool:
        """Set up the application and check requirements."""
        # Make sure required packages are installed
        try:
            subprocess.run(['which', 'xclip', 'xdotool'], check=True)
        except subprocess.CalledProcessError:
            logger.error("Required packages not installed.")
            logger.error("Please install with: sudo apt install xclip xdotool")
            return False
        
        # Test microphone if requested
        if self.args.test_mic:
            if not self.recorder.test_microphone():
                logger.warning("Microphone test failed. Continuing anyway...")
            logger.info("Continuing with normal operation...\n")
        
        # Load Whisper model in a separate thread
        thread = threading.Thread(target=self.transcriber.load_model)
        thread.daemon = True
        thread.start()
        
        # Display key info
        key_display = self.args.key if len(self.args.key) == 1 else self.args.key.upper()
        logger.info(f"Press and hold '{key_display}' key to record, release to transcribe")
        self.notifier.notify(
            "Whisper STT", 
            f"Press and hold '{key_display}' key to record, release to transcribe"
        )
        
        return True
    
    def on_press(self, key) -> None:
        """Handle key press events."""
        try:
            if key == self.trigger_key and not self.recorder.recording:
                self.start_recording()
        except AttributeError:
            pass
    
    def on_release(self, key) -> None:
        """Handle key release events."""
        try:
            if key == self.trigger_key and self.recorder.recording:
                self.stop_and_transcribe()
        except AttributeError:
            pass
    
    def start_recording(self) -> None:
        """Start recording audio."""
        if self.recorder.start():
            self.notifier.notify("Whisper STT", "Recording... (release key to process)")
    
    def stop_and_transcribe(self) -> None:
        """Stop recording and transcribe the audio."""
        audio_data = self.recorder.stop()
        if audio_data is None:
            return
        
        self.notifier.notify("Whisper STT", "Processing audio...")
        
        # Save audio to WAV
        wav_path = self.processor.save_to_wav(audio_data, self.recorder.sample_rate)
        if not wav_path:
            self.notifier.notify("Whisper STT", "Failed to save audio file")
            return
        
        # Validate WAV file
        if not self.processor.validate_wav_file(wav_path):
            self.notifier.notify("Whisper STT", "Invalid audio recording. Please try again.")
            return
        
        # Transcribe
        transcribed_text = self.transcriber.transcribe(wav_path)
        
        # Process result
        if not transcribed_text:
            logger.warning("Transcription returned empty string")
            self.notifier.notify("Whisper STT", "No speech detected. Try speaking louder or check mic.")
            
            # Try playing back the audio for verification
            # self.processor.play_audio(wav_path)
            return
        
        logger.info(f"Raw transcription: '{transcribed_text}'")
        
        # Insert the text
        if self.notifier.insert_text(transcribed_text):
            self.notifier.notify(
                "Whisper STT", 
                f"Inserted: {transcribed_text[:50]}{'...' if len(transcribed_text) > 50 else ''}"
            )
    
    def run(self) -> None:
        """Run the application, listening for key events."""
        if not self.setup():
            return
        
        # Keyboard listener
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                logger.info("Exiting...")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hold-to-record speech recognition with Whisper")
    parser.add_argument('--model', default='small', 
                        help='Model size to use (tiny, base, small, medium, large)')
    parser.add_argument('--sample_rate', type=int, default=16000, 
                        help='Sample rate for recording')
    parser.add_argument('--key', default='z', 
                        help='Key to hold for recording (single character, e.g., z, a, s)')
    parser.add_argument('--test-mic', action='store_true', 
                        help='Test microphone before starting')
    parser.add_argument('--faster', action='store_true', 
                        help='Use faster-whisper implementation')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable additional debug output')
    return parser.parse_args()


def main() -> int:
    """Main entry point for the application."""
    args = parse_arguments()
    app = WhisperHotkey(args)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())