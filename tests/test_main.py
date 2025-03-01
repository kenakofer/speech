"""
Tests for the main module functions.
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path
import argparse

# Add the parent directory to the path so we can import the main module
sys.path.insert(0, str(Path(__file__).parent.parent))

from whisper_module import parse_arguments, main
import sys


class TestMain(unittest.TestCase):
    """Test the main module functions."""

    @patch('argparse.ArgumentParser')
    def test_parse_arguments(self, mock_argparse):
        """Test the parse_arguments function."""
        # Mock the ArgumentParser
        mock_parser = MagicMock()
        mock_argparse.return_value = mock_parser
        
        # Mock the parsed arguments
        mock_args = MagicMock()
        mock_parser.parse_args.return_value = mock_args
        
        # Call parse_arguments
        result = parse_arguments()
        
        # Check that ArgumentParser was called
        mock_argparse.assert_called_with(description="Hold-to-record speech recognition with Whisper")
        
        # Check that parse_args was called
        mock_parser.parse_args.assert_called_once()
        
        # Check that the result was the mock args
        self.assertEqual(result, mock_args)
    
    @patch('whisper_module.WhisperHotkey')
    def test_main_success(self, mock_whisper_hotkey):
        """Test the main function with success."""
        # Mock the parse_arguments function
        with patch('whisper_module.parse_arguments') as mock_parse_arguments:
            # Mock the args
            mock_args = MagicMock()
            mock_parse_arguments.return_value = mock_args
            
            # Mock the WhisperHotkey class
            mock_app = MagicMock()
            mock_whisper_hotkey.return_value = mock_app
            
            # Call main
            result = main()
            
            # Check that parse_arguments was called
            mock_parse_arguments.assert_called_once()
            
            # Check that WhisperHotkey was instantiated with the args
            mock_whisper_hotkey.assert_called_with(mock_args)
            
            # Check that the app's run method was called
            mock_app.run.assert_called_once()
            
            # Check that the function returned 0 (success)
            self.assertEqual(result, 0)
    
    def test_main_exception(self):
        """Test the main function with an exception."""
        # Mock the parse_arguments function
        with patch('whisper_module.parse_arguments') as mock_parse_arguments:
            # Mock the args
            mock_args = MagicMock()
            mock_parse_arguments.return_value = mock_args
            
            # Mock the WhisperHotkey class to raise an exception
            with patch('whisper_module.WhisperHotkey') as mock_whisper_hotkey:
                mock_whisper_hotkey.side_effect = Exception("Test error")
                
                # Mock the logger
                with patch('logging.Logger.error') as mock_logger_error:
                    # Call main
                    result = main()
                    
                    # Check that the error was logged
                    mock_logger_error.assert_called_once()
                
                # Check that the function returned 1 (error)
                self.assertEqual(result, 1)


if __name__ == '__main__':
    unittest.main()