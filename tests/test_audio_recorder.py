"""
Tests for the AudioRecorder class.
"""
import unittest
from unittest.mock import patch, MagicMock, call
import numpy as np
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import the main module
sys.path.insert(0, str(Path(__file__).parent.parent))

from whisper_module import AudioRecorder


class TestAudioRecorder(unittest.TestCase):
    """Test the AudioRecorder class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a patcher for the sounddevice module
        self.sd_patcher = patch('whisper_module.sd')
        self.mock_sd = self.sd_patcher.start()
        
        # Create a recorder instance
        self.recorder = AudioRecorder(sample_rate=16000)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.sd_patcher.stop()
    
    def test_init(self):
        """Test the initialization of AudioRecorder."""
        self.assertEqual(self.recorder.sample_rate, 16000)
        self.assertFalse(self.recorder.recording)
        self.assertEqual(self.recorder.frames, [])
        self.assertIsNone(self.recorder.stream)
    
    def test_audio_callback(self):
        """Test the audio callback function."""
        # Create a mock indata array
        mock_indata = np.array([[0.1], [0.2], [0.3]])
        
        # Set up the recorder for recording
        self.recorder.recording = True
        self.recorder.frames = []
        
        # Call the callback
        self.recorder.audio_callback(mock_indata, 3, None, None)
        
        # Check that the frame was appended
        self.assertEqual(len(self.recorder.frames), 1)
        np.testing.assert_array_equal(self.recorder.frames[0], mock_indata)
    
    def test_audio_callback_not_recording(self):
        """Test the audio callback when not recording."""
        # Create a mock indata array
        mock_indata = np.array([[0.1], [0.2], [0.3]])
        
        # Set up the recorder for not recording
        self.recorder.recording = False
        self.recorder.frames = []
        
        # Call the callback
        self.recorder.audio_callback(mock_indata, 3, None, None)
        
        # Check that no frame was appended
        self.assertEqual(len(self.recorder.frames), 0)
    
    def test_start(self):
        """Test starting recording."""
        # Mock the InputStream
        mock_stream = MagicMock()
        self.mock_sd.InputStream.return_value = mock_stream
        
        # Call start
        result = self.recorder.start()
        
        # Check that recording was started correctly
        self.assertTrue(result)
        self.assertTrue(self.recorder.recording)
        self.assertEqual(self.recorder.frames, [])
        self.assertEqual(self.recorder.stream, mock_stream)
        mock_stream.start.assert_called_once()
        
        # Check the InputStream was created with the right parameters
        self.mock_sd.InputStream.assert_called_with(
            samplerate=16000, 
            channels=1, 
            callback=self.recorder.audio_callback,
            blocksize=1024,
            dtype='float32',
            latency='low'
        )
    
    def test_start_error(self):
        """Test starting recording with an error."""
        # Mock the InputStream to raise an exception
        self.mock_sd.InputStream.side_effect = Exception("Test error")
        
        # Call start
        result = self.recorder.start()
        
        # Check that recording failed correctly
        self.assertFalse(result)
        self.assertFalse(self.recorder.recording)
    
    def test_stop(self):
        """Test stopping recording."""
        # Set up a mock stream
        mock_stream = MagicMock()
        self.recorder.stream = mock_stream
        self.recorder.recording = True
        
        # Create mock frames
        frame1 = np.array([[0.1], [0.2]])
        frame2 = np.array([[0.3], [0.4]])
        self.recorder.frames = [frame1, frame2]
        
        # Call stop
        audio_data = self.recorder.stop()
        
        # Check that recording was stopped correctly
        self.assertFalse(self.recorder.recording)
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        
        # Check the returned audio data
        expected_data = np.concatenate([frame1, frame2], axis=0)
        np.testing.assert_array_equal(audio_data, expected_data)
    
    def test_stop_not_recording(self):
        """Test stopping when not recording."""
        # Set up a not recording state
        self.recorder.recording = False
        
        # Call stop
        audio_data = self.recorder.stop()
        
        # Check that None was returned
        self.assertIsNone(audio_data)
    
    def test_stop_no_frames(self):
        """Test stopping with no frames recorded."""
        # Set up a recording state with no frames
        self.recorder.recording = True
        self.recorder.frames = []
        mock_stream = MagicMock()
        self.recorder.stream = mock_stream
        
        # Call stop
        audio_data = self.recorder.stop()
        
        # Check that the stream was closed but None was returned
        self.assertFalse(self.recorder.recording)
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        self.assertIsNone(audio_data)
    
    @patch('whisper_module.logger')
    def test_test_microphone(self, mock_logger):
        """Test the microphone test function."""
        # Mock the sounddevice rec function
        mock_recording = np.ones((48000, 1)) * 0.1  # Non-silent recording
        self.mock_sd.rec.return_value = mock_recording
        
        # Call test_microphone
        result = self.recorder.test_microphone(duration=3)
        
        # Check that the function called sounddevice correctly
        self.mock_sd.rec.assert_called_with(
            int(3 * 16000),
            samplerate=16000,
            channels=1,
            dtype='float32'
        )
        self.mock_sd.wait.assert_called_once()
        
        # Check the result
        self.assertTrue(result)
    
    @patch('whisper_module.logger')
    def test_test_microphone_silent(self, mock_logger):
        """Test the microphone test function with a silent recording."""
        # Mock the sounddevice rec function with zeros (silent)
        mock_recording = np.zeros((48000, 1))
        self.mock_sd.rec.return_value = mock_recording
        
        # Call test_microphone
        result = self.recorder.test_microphone(duration=3)
        
        # Check the result
        self.assertFalse(result)
    
    @patch('whisper_module.logger')
    def test_test_microphone_error(self, mock_logger):
        """Test the microphone test function with an error."""
        # Mock the sounddevice rec function to raise an exception
        self.mock_sd.rec.side_effect = Exception("Test error")
        
        # Call test_microphone
        result = self.recorder.test_microphone(duration=3)
        
        # Check the result
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()