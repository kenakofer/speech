# Whisper Hotkey Tests

This directory contains tests for the Whisper Hotkey application.

## Running Tests

From the root directory, run:

```bash
./run_tests.sh
```

Or use pytest directly:

```bash
pytest
```

## Test Coverage

These tests use pytest with coverage reporting. The coverage report will show 
which lines of code are tested and which are not.

## Test Organization

- `test_audio_recorder.py`: Tests for the AudioRecorder class
- `test_audio_processor.py`: Tests for the AudioProcessor class
- `test_notification_manager.py`: Tests for the NotificationManager class
- `test_transcription_engine.py`: Tests for the TranscriptionEngine class
- `test_whisper_hotkey.py`: Tests for the WhisperHotkey class
- `test_main.py`: Tests for the main module functions

## Adding New Tests

When adding new features, please add corresponding tests. Follow these guidelines:

1. Use the existing pattern of test classes and methods
2. Mock external dependencies like libraries and hardware
3. Test both success and failure paths
4. Try to achieve high code coverage

## Running Specific Tests

You can run specific test files or test methods with:

```bash
# Run a specific test file
pytest tests/test_audio_recorder.py

# Run a specific test method
pytest tests/test_audio_recorder.py::TestAudioRecorder::test_start
```