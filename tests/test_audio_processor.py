"""
Tests for the AudioProcessor class.
"""
import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import numpy as np
import sys
import os
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import the main module
sys.path.insert(0, str(Path(__file__).parent.parent))

from whisper_module import AudioProcessor


class TestAudioProcessor(unittest.TestCase):
    """Test the AudioProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.processor = AudioProcessor(debug_dir=self.temp_dir)
        
        # Create a sample audio array
        self.audio_data = np.random.random((16000, 1)).astype(np.float32)  # 1 second of random audio
        self.sample_rate = 16000
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization of AudioProcessor."""
        # Test with default debug directory
        processor = AudioProcessor()
        self.assertEqual(processor.debug_dir, os.path.expanduser("~/speech"))
        
        # Test with custom debug directory
        self.assertEqual(self.processor.debug_dir, self.temp_dir)
        
        # Check that the directory exists
        self.assertTrue(os.path.exists(self.temp_dir))
    
    @patch('whisper_module.soundfile')
    @patch('whisper_module.logger')
    def test_save_to_wav_with_soundfile(self, mock_logger, mock_sf):
        """Test saving audio to WAV using soundfile."""
        # Call save_to_wav
        result = self.processor.save_to_wav(self.audio_data, self.sample_rate)
        
        # Check that soundfile was called correctly
        expected_path = os.path.join(self.temp_dir, "last_recording.wav")
        mock_sf.write.assert_called_with(expected_path, self.audio_data, self.sample_rate)
        
        # Check that the correct path was returned
        self.assertEqual(result, expected_path)
    
    @patch('whisper_module.soundfile')
    @patch('whisper_module.wavfile')
    @patch('whisper_module.logger')
    def test_save_to_wav_with_scipy(self, mock_logger, mock_wavfile, mock_sf):
        """Test saving audio to WAV using scipy when soundfile is not available."""
        # Make soundfile raise an ImportError
        mock_sf.write.side_effect = ImportError("No soundfile")
        
        # Call save_to_wav
        result = self.processor.save_to_wav(self.audio_data, self.sample_rate)
        
        # Check that wavfile was called correctly
        expected_path = os.path.join(self.temp_dir, "last_recording.wav")
        
        # Check that the normalization happened correctly
        mock_wavfile.write.assert_called_once()
        args = mock_wavfile.write.call_args[0]
        self.assertEqual(args[0], expected_path)
        self.assertEqual(args[1], self.sample_rate)
        # The third argument would be the normalized audio data (int16)
        # We can't easily check the exact array, but we can check its type
        self.assertEqual(args[2].dtype, np.int16)
        
        # Check that the correct path was returned
        self.assertEqual(result, expected_path)
    
    @patch('whisper_module.soundfile')
    @patch('whisper_module.wavfile')
    @patch('whisper_module.subprocess')
    @patch('whisper_module.tempfile')
    @patch('whisper_module.logger')
    def test_save_to_wav_with_ffmpeg(self, mock_logger, mock_tempfile, mock_subprocess, 
                                     mock_wavfile, mock_sf):
        """Test saving audio to WAV using ffmpeg when neither soundfile nor scipy is available."""
        # Make soundfile and wavfile raise ImportError
        mock_sf.write.side_effect = ImportError("No soundfile")
        mock_wavfile.write.side_effect = ImportError("No wavfile")
        
        # Mock tempfile to return a known path
        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/test_audio.raw"
        mock_tempfile.NamedTemporaryFile.return_value.__enter__.return_value = mock_tmp
        
        # Mock successful subprocess run
        mock_subprocess.run.return_value.returncode = 0
        
        # Call save_to_wav
        result = self.processor.save_to_wav(self.audio_data, self.sample_rate)
        
        # Check that ffmpeg was called correctly
        expected_path = os.path.join(self.temp_dir, "last_recording.wav")
        mock_subprocess.run.assert_called_with([
            'ffmpeg', '-y', 
            '-f', 'f32le', 
            '-ar', str(self.sample_rate), 
            '-ac', '1', 
            '-i', mock_tmp.name, 
            expected_path
        ], capture_output=True, text=True)
        
        # Check that the correct path was returned
        self.assertEqual(result, expected_path)
    
    @patch('whisper_module.soundfile')
    @patch('whisper_module.wavfile')
    @patch('whisper_module.subprocess')
    @patch('whisper_module.tempfile')
    @patch('whisper_module.logger')
    def test_save_to_wav_ffmpeg_error(self, mock_logger, mock_tempfile, mock_subprocess, 
                                    mock_wavfile, mock_sf):
        """Test saving audio to WAV when ffmpeg fails."""
        # Make soundfile and wavfile raise ImportError
        mock_sf.write.side_effect = ImportError("No soundfile")
        mock_wavfile.write.side_effect = ImportError("No wavfile")
        
        # Mock tempfile to return a known path
        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/test_audio.raw"
        mock_tempfile.NamedTemporaryFile.return_value.__enter__.return_value = mock_tmp
        
        # Mock failed subprocess run
        mock_subprocess.run.return_value.returncode = 1
        mock_subprocess.run.return_value.stderr = "ffmpeg error"
        
        # Call save_to_wav
        result = self.processor.save_to_wav(self.audio_data, self.sample_rate)
        
        # Check that None was returned
        self.assertIsNone(result)
    
    @patch('whisper_module.wave')
    @patch('whisper_module.logger')
    def test_validate_wav_file_success(self, mock_logger, mock_wave):
        """Test validating a good WAV file."""
        # Mock wave.open to return a valid WAV file
        mock_wf = MagicMock()
        mock_wf.getparams.return_value = "valid params"
        mock_wf.getnframes.return_value = 16000  # 1 second at 16kHz
        mock_wave.open.return_value.__enter__.return_value = mock_wf
        
        # Call validate_wav_file
        result = self.processor.validate_wav_file("/path/to/test.wav")
        
        # Check that the validation succeeded
        self.assertTrue(result)
        mock_wave.open.assert_called_with("/path/to/test.wav", 'rb')
    
    @patch('whisper_module.wave')
    @patch('whisper_module.logger')
    def test_validate_wav_file_empty(self, mock_logger, mock_wave):
        """Test validating an empty WAV file."""
        # Mock wave.open to return an empty WAV file
        mock_wf = MagicMock()
        mock_wf.getparams.return_value = "valid params"
        mock_wf.getnframes.return_value = 0  # No frames
        mock_wave.open.return_value.__enter__.return_value = mock_wf
        
        # Call validate_wav_file
        result = self.processor.validate_wav_file("/path/to/test.wav")
        
        # Check that the validation failed
        self.assertFalse(result)
    
    @patch('whisper_module.wave')
    @patch('whisper_module.logger')
    def test_validate_wav_file_error(self, mock_logger, mock_wave):
        """Test validating a WAV file with an error."""
        # Mock wave.open to raise an exception
        mock_wave.open.side_effect = Exception("Invalid WAV file")
        
        # Call validate_wav_file
        result = self.processor.validate_wav_file("/path/to/test.wav")
        
        # Check that the validation failed
        self.assertFalse(result)
    
    @patch('whisper_module.subprocess')
    @patch('whisper_module.logger')
    def test_play_audio_success(self, mock_logger, mock_subprocess):
        """Test playing audio successfully."""
        # Mock subprocess.run
        mock_subprocess.run.return_value = MagicMock()
        
        # Call play_audio
        result = self.processor.play_audio("/path/to/test.wav")
        
        # Check that aplay was called correctly
        mock_subprocess.run.assert_called_with(["aplay", "/path/to/test.wav"])
        
        # Check that the function returned success
        self.assertTrue(result)
    
    @patch('whisper_module.subprocess')
    @patch('whisper_module.logger')
    def test_play_audio_error(self, mock_logger, mock_subprocess):
        """Test playing audio with an error."""
        # Mock subprocess.run to raise an exception
        mock_subprocess.run.side_effect = Exception("aplay error")
        
        # Call play_audio
        result = self.processor.play_audio("/path/to/test.wav")
        
        # Check that the function returned failure
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()