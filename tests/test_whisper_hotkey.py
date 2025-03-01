"""
Tests for the WhisperHotkey class.
"""
import unittest
from unittest.mock import patch, MagicMock, call
import sys
import os
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import the main module
sys.path.insert(0, str(Path(__file__).parent.parent))

from whisper_module import WhisperHotkey, AudioRecorder, AudioProcessor, NotificationManager, TranscriptionEngine


class TestWhisperHotkey(unittest.TestCase):
    """Test the WhisperHotkey class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock arguments
        self.args = argparse.Namespace(
            model='tiny',
            sample_rate=16000,
            key='z',
            test_mic=False,
            faster=False,
            debug=False,
            paste_method='both'
        )
        
        # Create patcher for the keyboard module
        self.keyboard_patcher = patch('whisper_hotkey.keyboard')
        self.mock_keyboard = self.keyboard_patcher.start()
        
        # Create a mock KeyCode
        self.mock_keycode = MagicMock()
        self.mock_keyboard.KeyCode.from_char.return_value = self.mock_keycode
        
        # Create the WhisperHotkey instance
        self.app = WhisperHotkey(self.args)
        
        # Create patchers for the component classes
        self.recorder_patcher = patch.object(self.app, 'recorder', autospec=True)
        self.processor_patcher = patch.object(self.app, 'processor', autospec=True)
        self.notifier_patcher = patch.object(self.app, 'notifier', autospec=True)
        self.transcriber_patcher = patch.object(self.app, 'transcriber', autospec=True)
        
        # Start the patchers
        self.mock_recorder = self.recorder_patcher.start()
        self.mock_processor = self.processor_patcher.start()
        self.mock_notifier = self.notifier_patcher.start()
        self.mock_transcriber = self.transcriber_patcher.start()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop the patchers
        self.keyboard_patcher.stop()
        self.recorder_patcher.stop()
        self.processor_patcher.stop()
        self.notifier_patcher.stop()
        self.transcriber_patcher.stop()
    
    def test_init(self):
        """Test initialization of WhisperHotkey."""
        # Reset the app
        self.app = WhisperHotkey(self.args)
        
        # Check that the components were initialized correctly
        self.assertIsInstance(self.app.recorder, AudioRecorder)
        self.assertIsInstance(self.app.processor, AudioProcessor)
        self.assertIsInstance(self.app.notifier, NotificationManager)
        self.assertIsInstance(self.app.transcriber, TranscriptionEngine)
        
        # Check that the trigger key was set correctly
        self.mock_keyboard.KeyCode.from_char.assert_called_with('z')
        self.assertEqual(self.app.trigger_key, self.mock_keycode)
    
    def test_init_with_function_key(self):
        """Test initialization with a function key."""
        # Set the key to a function key
        self.args.key = 'f13'
        
        # Mock the getattr function to return a mock Key
        mock_key = MagicMock()
        self.mock_keyboard.Key = MagicMock()
        setattr(self.mock_keyboard.Key, 'f13', mock_key)
        
        # Create a new app
        app = WhisperHotkey(self.args)
        
        # Check that the trigger key was set correctly
        self.assertEqual(app.trigger_key, mock_key)
    
    def test_init_with_invalid_key(self):
        """Test initialization with an invalid key."""
        # Set the key to an invalid function key
        self.args.key = 'invalid'
        
        # Make getattr raise AttributeError
        self.mock_keyboard.Key = MagicMock()
        
        # Create a new app
        with patch('whisper_hotkey.logger') as mock_logger:
            app = WhisperHotkey(self.args)
            
            # Check that a warning was logged
            mock_logger.warning.assert_called_once()
            
            # Check that the trigger key was set to default 'z'
            self.mock_keyboard.KeyCode.from_char.assert_called_with('z')
    
    @patch('whisper_hotkey.subprocess')
    @patch('whisper_hotkey.threading')
    @patch('whisper_hotkey.logger')
    def test_setup_success(self, mock_logger, mock_threading, mock_subprocess):
        """Test setup with successful checks."""
        # Mock subprocess.run to succeed
        mock_subprocess.run.return_value = MagicMock()
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError
        
        # Call setup
        result = self.app.setup()
        
        # Check that setup succeeded
        self.assertTrue(result)
        
        # Check that required packages were checked
        mock_subprocess.run.assert_called_with(['which', 'xclip', 'xdotool'], check=True)
        
        # Check that the model was loaded in a thread
        mock_threading.Thread.assert_called_with(target=self.mock_transcriber.load_model)
        mock_thread = mock_threading.Thread.return_value
        mock_thread.start.assert_called_once()
        
        # Check that notification was sent
        self.mock_notifier.notify.assert_called_once()
    
    @patch('whisper_hotkey.subprocess')
    @patch('whisper_hotkey.logger')
    def test_setup_missing_packages(self, mock_logger, mock_subprocess):
        """Test setup with missing packages."""
        # Mock subprocess.run to fail
        mock_subprocess.run.side_effect = subprocess.CalledProcessError(1, ['which'])
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError
        
        # Call setup
        result = self.app.setup()
        
        # Check that setup failed
        self.assertFalse(result)
        
        # Check that errors were logged
        self.assertEqual(mock_logger.error.call_count, 2)
    
    @patch('whisper_hotkey.subprocess')
    @patch('whisper_hotkey.threading')
    @patch('whisper_hotkey.logger')
    def test_setup_with_mic_test(self, mock_logger, mock_threading, mock_subprocess):
        """Test setup with microphone test."""
        # Set test_mic to True
        self.args.test_mic = True
        self.app = WhisperHotkey(self.args)
        
        # Mock recorder and subprocess
        mock_recorder = MagicMock()
        self.app.recorder = mock_recorder
        mock_recorder.test_microphone.return_value = True
        mock_subprocess.run.return_value = MagicMock()
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError
        
        # Call setup
        result = self.app.setup()
        
        # Check that setup succeeded
        self.assertTrue(result)
        
        # Check that test_microphone was called
        mock_recorder.test_microphone.assert_called_once()
    
    def test_on_press_recording_key(self):
        """Test key press event with the recording key."""
        # Mock the key
        key = self.mock_keycode
        
        # Set up recorder to not be recording
        self.mock_recorder.recording = False
        
        # Mock start_recording
        self.app.start_recording = MagicMock()
        
        # Call on_press
        self.app.on_press(key)
        
        # Check that start_recording was called
        self.app.start_recording.assert_called_once()
    
    def test_on_press_already_recording(self):
        """Test key press event when already recording."""
        # Mock the key
        key = self.mock_keycode
        
        # Set up recorder to already be recording
        self.mock_recorder.recording = True
        
        # Mock start_recording
        self.app.start_recording = MagicMock()
        
        # Call on_press
        self.app.on_press(key)
        
        # Check that start_recording was not called
        self.app.start_recording.assert_not_called()
    
    def test_on_press_different_key(self):
        """Test key press event with a different key."""
        # Mock a different key
        different_key = MagicMock()
        
        # Mock start_recording
        self.app.start_recording = MagicMock()
        
        # Call on_press
        self.app.on_press(different_key)
        
        # Check that start_recording was not called
        self.app.start_recording.assert_not_called()
    
    def test_on_press_attribute_error(self):
        """Test key press event with an attribute error."""
        # Mock a key that raises AttributeError on comparison
        key = MagicMock()
        key.__eq__ = MagicMock(side_effect=AttributeError("No __eq__"))
        
        # Mock start_recording
        self.app.start_recording = MagicMock()
        
        # Call on_press (should not raise an exception)
        self.app.on_press(key)
        
        # Check that start_recording was not called
        self.app.start_recording.assert_not_called()
    
    def test_on_release_recording_key(self):
        """Test key release event with the recording key."""
        # Mock the key
        key = self.mock_keycode
        
        # Set up recorder to be recording
        self.mock_recorder.recording = True
        
        # Mock stop_and_transcribe
        self.app.stop_and_transcribe = MagicMock()
        
        # Call on_release
        self.app.on_release(key)
        
        # Check that stop_and_transcribe was called
        self.app.stop_and_transcribe.assert_called_once()
    
    def test_on_release_not_recording(self):
        """Test key release event when not recording."""
        # Mock the key
        key = self.mock_keycode
        
        # Set up recorder to not be recording
        self.mock_recorder.recording = False
        
        # Mock stop_and_transcribe
        self.app.stop_and_transcribe = MagicMock()
        
        # Call on_release
        self.app.on_release(key)
        
        # Check that stop_and_transcribe was not called
        self.app.stop_and_transcribe.assert_not_called()
    
    def test_on_release_different_key(self):
        """Test key release event with a different key."""
        # Mock a different key
        different_key = MagicMock()
        
        # Mock stop_and_transcribe
        self.app.stop_and_transcribe = MagicMock()
        
        # Call on_release
        self.app.on_release(different_key)
        
        # Check that stop_and_transcribe was not called
        self.app.stop_and_transcribe.assert_not_called()
    
    def test_on_release_attribute_error(self):
        """Test key release event with an attribute error."""
        # Mock a key that raises AttributeError on comparison
        key = MagicMock()
        key.__eq__ = MagicMock(side_effect=AttributeError("No __eq__"))
        
        # Mock stop_and_transcribe
        self.app.stop_and_transcribe = MagicMock()
        
        # Call on_release (should not raise an exception)
        self.app.on_release(key)
        
        # Check that stop_and_transcribe was not called
        self.app.stop_and_transcribe.assert_not_called()
    
    def test_start_recording_success(self):
        """Test starting recording successfully."""
        # Mock recorder.start to return True
        self.mock_recorder.start.return_value = True
        
        # Call start_recording
        self.app.start_recording()
        
        # Check that recorder.start was called
        self.mock_recorder.start.assert_called_once()
        
        # Check that a notification was sent
        self.mock_notifier.notify.assert_called_once()
    
    def test_start_recording_failure(self):
        """Test starting recording with a failure."""
        # Mock recorder.start to return False
        self.mock_recorder.start.return_value = False
        
        # Call start_recording
        self.app.start_recording()
        
        # Check that recorder.start was called
        self.mock_recorder.start.assert_called_once()
        
        # Check that no notification was sent
        self.mock_notifier.notify.assert_not_called()
    
    def test_stop_and_transcribe_no_audio(self):
        """Test stopping and transcribing with no audio."""
        # Mock recorder.stop to return None
        self.mock_recorder.stop.return_value = None
        
        # Call stop_and_transcribe
        self.app.stop_and_transcribe()
        
        # Check that recorder.stop was called
        self.mock_recorder.stop.assert_called_once()
        
        # Check that no further processing was done
        self.mock_processor.save_to_wav.assert_not_called()
        self.mock_transcriber.transcribe.assert_not_called()
    
    def test_stop_and_transcribe_successful_transcription(self):
        """Test stopping and transcribing with successful transcription."""
        # Mock recorder.stop to return audio data
        audio_data = MagicMock()
        self.mock_recorder.stop.return_value = audio_data
        
        # Mock processor.save_to_wav to return a path
        wav_path = "/path/to/audio.wav"
        self.mock_processor.save_to_wav.return_value = wav_path
        
        # Mock processor.validate_wav_file to return True
        self.mock_processor.validate_wav_file.return_value = True
        
        # Mock transcriber.transcribe to return text
        transcribed_text = "This is a test transcription."
        self.mock_transcriber.transcribe.return_value = transcribed_text
        
        # Mock notifier.insert_text to return True
        self.mock_notifier.insert_text.return_value = True
        
        # Call stop_and_transcribe
        self.app.stop_and_transcribe()
        
        # Check the flow
        self.mock_recorder.stop.assert_called_once()
        self.mock_notifier.notify.assert_called_with("Whisper STT", "Processing audio...")
        self.mock_processor.save_to_wav.assert_called_with(audio_data, self.app.recorder.sample_rate)
        self.mock_processor.validate_wav_file.assert_called_with(wav_path)
        self.mock_transcriber.transcribe.assert_called_with(wav_path)
        self.mock_notifier.insert_text.assert_called_with(transcribed_text)
        
        # Check the final notification
        self.mock_notifier.notify.assert_called_with(
            "Whisper STT", 
            f"Inserted: {transcribed_text[:50]}{'...' if len(transcribed_text) > 50 else ''}"
        )
    
    def test_stop_and_transcribe_save_error(self):
        """Test stopping and transcribing with an error saving the audio."""
        # Mock recorder.stop to return audio data
        audio_data = MagicMock()
        self.mock_recorder.stop.return_value = audio_data
        
        # Mock processor.save_to_wav to return None (error)
        self.mock_processor.save_to_wav.return_value = None
        
        # Call stop_and_transcribe
        self.app.stop_and_transcribe()
        
        # Check that an error notification was sent
        self.mock_notifier.notify.assert_called_with("Whisper STT", "Failed to save audio file")
        
        # Check that no further processing was done
        self.mock_processor.validate_wav_file.assert_not_called()
        self.mock_transcriber.transcribe.assert_not_called()
    
    def test_stop_and_transcribe_invalid_wav(self):
        """Test stopping and transcribing with an invalid WAV file."""
        # Mock recorder.stop to return audio data
        audio_data = MagicMock()
        self.mock_recorder.stop.return_value = audio_data
        
        # Mock processor.save_to_wav to return a path
        wav_path = "/path/to/audio.wav"
        self.mock_processor.save_to_wav.return_value = wav_path
        
        # Mock processor.validate_wav_file to return False (invalid)
        self.mock_processor.validate_wav_file.return_value = False
        
        # Call stop_and_transcribe
        self.app.stop_and_transcribe()
        
        # Check that an error notification was sent
        self.mock_notifier.notify.assert_called_with("Whisper STT", "Invalid audio recording. Please try again.")
        
        # Check that no further processing was done
        self.mock_transcriber.transcribe.assert_not_called()
    
    def test_stop_and_transcribe_empty_transcription(self):
        """Test stopping and transcribing with empty transcription."""
        # Mock recorder.stop to return audio data
        audio_data = MagicMock()
        self.mock_recorder.stop.return_value = audio_data
        
        # Mock processor.save_to_wav to return a path
        wav_path = "/path/to/audio.wav"
        self.mock_processor.save_to_wav.return_value = wav_path
        
        # Mock processor.validate_wav_file to return True
        self.mock_processor.validate_wav_file.return_value = True
        
        # Mock transcriber.transcribe to return empty string
        self.mock_transcriber.transcribe.return_value = ""
        
        # Call stop_and_transcribe
        with patch('whisper_hotkey.logger') as mock_logger:
            self.app.stop_and_transcribe()
            
            # Check that a warning was logged
            mock_logger.warning.assert_called_once()
        
        # Check that an error notification was sent
        self.mock_notifier.notify.assert_called_with(
            "Whisper STT", 
            "No speech detected. Try speaking louder or check mic."
        )
        
        # Check that text insertion was not attempted
        self.mock_notifier.insert_text.assert_not_called()
    
    def test_stop_and_transcribe_insert_error(self):
        """Test stopping and transcribing with an error inserting text."""
        # Mock recorder.stop to return audio data
        audio_data = MagicMock()
        self.mock_recorder.stop.return_value = audio_data
        
        # Mock processor.save_to_wav to return a path
        wav_path = "/path/to/audio.wav"
        self.mock_processor.save_to_wav.return_value = wav_path
        
        # Mock processor.validate_wav_file to return True
        self.mock_processor.validate_wav_file.return_value = True
        
        # Mock transcriber.transcribe to return text
        transcribed_text = "This is a test transcription."
        self.mock_transcriber.transcribe.return_value = transcribed_text
        
        # Mock notifier.insert_text to return False (error)
        self.mock_notifier.insert_text.return_value = False
        
        # Call stop_and_transcribe
        self.app.stop_and_transcribe()
        
        # Check that text insertion was attempted but failed
        self.mock_notifier.insert_text.assert_called_with(transcribed_text)
        
        # Check that no success notification was sent
        self.assertEqual(self.mock_notifier.notify.call_count, 1)  # Only the "Processing audio..." notification
    
    @patch('whisper_hotkey.keyboard')
    def test_run(self, mock_keyboard):
        """Test the run method."""
        # Mock setup to return True
        self.app.setup = MagicMock(return_value=True)
        
        # Mock the keyboard.Listener
        mock_listener = MagicMock()
        mock_keyboard.Listener.return_value.__enter__.return_value = mock_listener
        
        # Call run
        self.app.run()
        
        # Check that setup was called
        self.app.setup.assert_called_once()
        
        # Check that keyboard.Listener was created and joined
        mock_keyboard.Listener.assert_called_with(
            on_press=self.app.on_press, 
            on_release=self.app.on_release
        )
        mock_listener.join.assert_called_once()
    
    @patch('whisper_hotkey.keyboard')
    def test_run_setup_failure(self, mock_keyboard):
        """Test the run method with setup failure."""
        # Mock setup to return False
        self.app.setup = MagicMock(return_value=False)
        
        # Call run
        self.app.run()
        
        # Check that setup was called
        self.app.setup.assert_called_once()
        
        # Check that keyboard.Listener was not created
        mock_keyboard.Listener.assert_not_called()
    
    @patch('whisper_hotkey.keyboard')
    @patch('whisper_hotkey.logger')
    def test_run_keyboard_interrupt(self, mock_logger, mock_keyboard):
        """Test the run method with a keyboard interrupt."""
        # Mock setup to return True
        self.app.setup = MagicMock(return_value=True)
        
        # Mock the keyboard.Listener to raise KeyboardInterrupt when joined
        mock_listener = MagicMock()
        mock_listener.join.side_effect = KeyboardInterrupt
        mock_keyboard.Listener.return_value.__enter__.return_value = mock_listener
        
        # Call run
        self.app.run()
        
        # Check that the error was logged
        mock_logger.info.assert_called_with("Exiting...")


if __name__ == '__main__':
    unittest.main()