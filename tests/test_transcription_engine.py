"""
Tests for the TranscriptionEngine class.
"""
import unittest
from unittest.mock import patch, MagicMock, call
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add the parent directory to the path so we can import the main module
sys.path.insert(0, str(Path(__file__).parent.parent))

from whisper_module import TranscriptionEngine


class MockSegment:
    """Mock for a whisper segment."""
    
    def __init__(self, id_, text, start, end):
        self.id = id_
        self.text = text
        self.start = start
        self.end = end


class MockInfo:
    """Mock for whisper info."""
    
    def __init__(self, language, language_probability):
        self.language = language
        self.language_probability = language_probability


class TestTranscriptionEngine(unittest.TestCase):
    """Test the TranscriptionEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = TranscriptionEngine(model_name='tiny', use_faster_whisper=False)
    
    def test_init(self):
        """Test initialization of TranscriptionEngine."""
        # Test with default values
        engine = TranscriptionEngine()
        self.assertEqual(engine.model_name, 'small')
        self.assertFalse(engine.use_faster_whisper)
        self.assertIsNone(engine.model)
        
        # Test with custom values
        engine = TranscriptionEngine(model_name='large', use_faster_whisper=True)
        self.assertEqual(engine.model_name, 'large')
        self.assertTrue(engine.use_faster_whisper)
        self.assertIsNone(engine.model)
    
    @patch('whisper_module.whisper')
    @patch('whisper_module.logger')
    def test_load_model_standard(self, mock_logger, mock_whisper):
        """Test loading standard whisper model."""
        # Mock whisper.load_model
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        
        # Call load_model
        result = self.engine.load_model()
        
        # Check that whisper.load_model was called correctly
        mock_whisper.load_model.assert_called_with('tiny')
        
        # Check that the model was set
        self.assertEqual(self.engine.model, mock_model)
        
        # Check that the function returned success
        self.assertTrue(result)
    
    @patch('whisper_module.WhisperModel')
    @patch('whisper_module.logger')
    def test_load_model_faster(self, mock_logger, mock_whisper_model):
        """Test loading faster whisper model."""
        # Set the engine to use faster whisper
        self.engine.use_faster_whisper = True
        
        # Mock WhisperModel
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model
        
        # Call load_model
        result = self.engine.load_model()
        
        # Check that WhisperModel was called correctly
        mock_whisper_model.assert_called_with(
            'tiny', 
            device="cpu", 
            compute_type="int8", 
            cpu_threads=2
        )
        
        # Check that the model was set
        self.assertEqual(self.engine.model, mock_model)
        
        # Check that the function returned success
        self.assertTrue(result)
    
    @patch('whisper_module.WhisperModel')
    @patch('whisper_module.whisper')
    @patch('whisper_module.logger')
    def test_load_model_faster_fallback(self, mock_logger, mock_whisper, mock_whisper_model):
        """Test loading faster whisper model with fallback to standard."""
        # Set the engine to use faster whisper
        self.engine.use_faster_whisper = True
        
        # Make WhisperModel raise an ImportError
        mock_whisper_model.side_effect = ImportError("No faster_whisper")
        
        # Mock standard whisper
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        
        # Call load_model
        result = self.engine.load_model()
        
        # Check that the engine fell back to standard whisper
        self.assertFalse(self.engine.use_faster_whisper)
        mock_whisper.load_model.assert_called_with('tiny')
        
        # Check that the model was set
        self.assertEqual(self.engine.model, mock_model)
        
        # Check that the function returned success
        self.assertTrue(result)
    
    @patch('whisper_module.whisper')
    @patch('whisper_module.logger')
    def test_load_model_error(self, mock_logger, mock_whisper):
        """Test loading model with an error."""
        # Make whisper.load_model raise an exception
        mock_whisper.load_model.side_effect = Exception("Model error")
        
        # Call load_model
        result = self.engine.load_model()
        
        # Check that the function returned failure
        self.assertFalse(result)
        self.assertIsNone(self.engine.model)
    
    @patch('whisper_module.logger')
    def test_transcribe_no_model(self, mock_logger):
        """Test transcribing without a loaded model."""
        # Call transcribe without loading a model
        result = self.engine.transcribe("/path/to/test.wav")
        
        # Check that the function returned None
        self.assertIsNone(result)
        
        # Check that an error was logged
        mock_logger.error.assert_called_once()
    
    @patch('whisper_module.logger')
    def test_transcribe_standard(self, mock_logger):
        """Test transcribing with standard whisper."""
        # Create a mock model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "This is a test transcription."
        }
        self.engine.model = mock_model
        
        # Call transcribe
        result = self.engine.transcribe("/path/to/test.wav")
        
        # Check that model.transcribe was called correctly
        mock_model.transcribe.assert_called_with(
            "/path/to/test.wav", 
            language="en", 
            task="transcribe", 
            fp16=False
        )
        
        # Check that the function returned the expected text
        self.assertEqual(result, "This is a test transcription.")
    
    @patch('whisper_module.logger')
    def test_transcribe_faster(self, mock_logger):
        """Test transcribing with faster whisper."""
        # Set the engine to use faster whisper
        self.engine.use_faster_whisper = True
        
        # Create mock segments and info
        mock_segments = [
            MockSegment(0, "This is ", 0.0, 1.0),
            MockSegment(1, "a test ", 1.0, 2.0),
            MockSegment(2, "transcription.", 2.0, 3.0)
        ]
        mock_info = MockInfo("en", 0.99)
        
        # Create a mock model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (mock_segments, mock_info)
        self.engine.model = mock_model
        
        # Call transcribe
        result = self.engine.transcribe("/path/to/test.wav")
        
        # Check that model.transcribe was called correctly
        mock_model.transcribe.assert_called_with(
            "/path/to/test.wav",
            language="en", 
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        
        # Check that the function returned the expected text
        self.assertEqual(result, "This is a test transcription.")
    
    @patch('whisper_module.whisper')
    @patch('whisper_module.logger')
    def test_transcribe_faster_error_fallback(self, mock_logger, mock_whisper):
        """Test transcribing with faster whisper that errors and falls back."""
        # Set the engine to use faster whisper
        self.engine.use_faster_whisper = True
        
        # Create a mock model that raises an exception for faster whisper
        mock_faster_model = MagicMock()
        mock_faster_model.transcribe.side_effect = Exception("Faster whisper error")
        
        # Create a mock model for standard whisper fallback
        mock_standard_model = MagicMock()
        mock_standard_model.transcribe.return_value = {
            "text": "This is a fallback transcription."
        }
        mock_whisper.load_model.return_value = mock_standard_model
        
        # Set the initial model
        self.engine.model = mock_faster_model
        
        # Call transcribe
        result = self.engine.transcribe("/path/to/test.wav")
        
        # Check that the engine fell back to standard whisper
        self.assertFalse(self.engine.use_faster_whisper)
        
        # Check that a new standard model was loaded
        mock_whisper.load_model.assert_called_with('tiny')
        
        # Check that the standard model was used for transcription
        mock_standard_model.transcribe.assert_called_with(
            "/path/to/test.wav", 
            language="en", 
            task="transcribe", 
            fp16=False
        )
        
        # Check that the function returned the fallback text
        self.assertEqual(result, "This is a fallback transcription.")
    
    @patch('whisper_module.logger')
    def test_transcribe_error(self, mock_logger):
        """Test transcribing with an error."""
        # Create a mock model that raises an exception
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = Exception("Transcription error")
        self.engine.model = mock_model
        
        # Call transcribe
        result = self.engine.transcribe("/path/to/test.wav")
        
        # Check that the function returned None
        self.assertIsNone(result)
        
        # Check that an error was logged
        mock_logger.error.assert_called_once()


if __name__ == '__main__':
    unittest.main()