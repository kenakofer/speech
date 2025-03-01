"""
Tests for the NotificationManager class.
"""
import unittest
from unittest.mock import patch, MagicMock, call
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import the main module
sys.path.insert(0, str(Path(__file__).parent.parent))

from whisper_module import NotificationManager


class TestNotificationManager(unittest.TestCase):
    """Test the NotificationManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.notifier = NotificationManager(paste_method='both')
    
    def test_init(self):
        """Test initialization of NotificationManager."""
        # Test with default paste method
        notifier = NotificationManager()
        self.assertEqual(notifier.paste_method, 'both')
        
        # Test with custom paste method
        notifier = NotificationManager(paste_method='ctrl+v')
        self.assertEqual(notifier.paste_method, 'ctrl+v')
    
    @patch('whisper_module.subprocess')
    @patch('whisper_module.logger')
    def test_notify(self, mock_logger, mock_subprocess):
        """Test sending notifications."""
        # Call notify
        self.notifier.notify("Test Title", "Test Message")
        
        # Check that notify-send was called correctly
        mock_subprocess.run.assert_called_with(['notify-send', 'Test Title', 'Test Message'])
        
        # Check that the message was logged
        mock_logger.info.assert_called_with('Test Title: Test Message')
    
    @patch('whisper_module.subprocess')
    @patch('whisper_module.time')
    @patch('whisper_module.logger')
    def test_insert_text_both(self, mock_logger, mock_time, mock_subprocess):
        """Test inserting text with 'both' paste method."""
        # Create a mock for Popen
        mock_popen = MagicMock()
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        mock_subprocess.Popen = mock_popen
        
        # Call insert_text
        result = self.notifier.insert_text("Test text")
        
        # Check that xclip was called for both selections
        self.assertEqual(mock_popen.call_count, 2)
        mock_popen.assert_has_calls([
            call(['xclip', '-selection', 'primary'], stdin=mock_subprocess.PIPE),
            call(['xclip', '-selection', 'clipboard'], stdin=mock_subprocess.PIPE)
        ])
        
        # Check that the process.communicate was called with the text
        mock_process.communicate.assert_called_with(input=b'Test text')
        
        # Check that xdotool was called for both paste methods
        self.assertEqual(mock_subprocess.run.call_count, 2)
        mock_subprocess.run.assert_has_calls([
            call(['xdotool', 'click', '2']),  # Middle click
            call(['xdotool', 'key', 'ctrl+v'])  # Ctrl+V
        ])
        
        # Check that we waited between operations
        self.assertEqual(mock_time.sleep.call_count, 2)
        
        # Check that the function returned success
        self.assertTrue(result)
    
    @patch('whisper_module.subprocess')
    @patch('whisper_module.time')
    @patch('whisper_module.logger')
    def test_insert_text_middle(self, mock_logger, mock_time, mock_subprocess):
        """Test inserting text with 'middle' paste method."""
        # Set the paste method to middle
        self.notifier.paste_method = 'middle'
        
        # Create a mock for Popen
        mock_popen = MagicMock()
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        mock_subprocess.Popen = mock_popen
        
        # Call insert_text
        result = self.notifier.insert_text("Test text")
        
        # Check that xclip was called for both selections
        self.assertEqual(mock_popen.call_count, 2)
        
        # Check that xdotool was only called for middle click
        self.assertEqual(mock_subprocess.run.call_count, 1)
        mock_subprocess.run.assert_called_with(['xdotool', 'click', '2'])
        
        # Check that the function returned success
        self.assertTrue(result)
    
    @patch('whisper_module.subprocess')
    @patch('whisper_module.time')
    @patch('whisper_module.logger')
    def test_insert_text_ctrl_v(self, mock_logger, mock_time, mock_subprocess):
        """Test inserting text with 'ctrl+v' paste method."""
        # Set the paste method to ctrl+v
        self.notifier.paste_method = 'ctrl+v'
        
        # Create a mock for Popen
        mock_popen = MagicMock()
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        mock_subprocess.Popen = mock_popen
        
        # Call insert_text
        result = self.notifier.insert_text("Test text")
        
        # Check that xclip was called for both selections
        self.assertEqual(mock_popen.call_count, 2)
        
        # Check that xdotool was only called for ctrl+v
        self.assertEqual(mock_subprocess.run.call_count, 1)
        mock_subprocess.run.assert_called_with(['xdotool', 'key', 'ctrl+v'])
        
        # Check that the function returned success
        self.assertTrue(result)
    
    @patch('whisper_module.subprocess')
    @patch('whisper_module.time')
    @patch('whisper_module.logger')
    def test_insert_text_type(self, mock_logger, mock_time, mock_subprocess):
        """Test inserting text with 'type' paste method."""
        # Set the paste method to type
        self.notifier.paste_method = 'type'
        
        # Create a mock for Popen
        mock_popen = MagicMock()
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        mock_subprocess.Popen = mock_popen
        
        # Call insert_text
        result = self.notifier.insert_text("Test text")
        
        # Check that xclip was called for both selections
        self.assertEqual(mock_popen.call_count, 2)
        
        # Check that xdotool was called for typing
        self.assertEqual(mock_subprocess.run.call_count, 1)
        mock_subprocess.run.assert_called_with(['xdotool', 'type', 'Test text'])
        
        # Check that the function returned success
        self.assertTrue(result)
    
    @patch('whisper_module.subprocess')
    @patch('whisper_module.logger')
    def test_insert_text_error(self, mock_logger, mock_subprocess):
        """Test inserting text with an error."""
        # Make subprocess.Popen raise an exception
        mock_subprocess.Popen.side_effect = Exception("xclip error")
        
        # Call insert_text
        result = self.notifier.insert_text("Test text")
        
        # Check that the function returned failure
        self.assertFalse(result)
        
        # Check that the error was logged
        mock_logger.error.assert_called_once()


if __name__ == '__main__':
    unittest.main()