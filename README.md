# Whisper Hold-to-Speak

A hold-to-speak speech recognition tool using OpenAI's Whisper. Press and hold a key (default: 'z') to record, and release to insert the transcribed text at the cursor position.

## Features

- Low latency recording with automatic speech-to-text conversion
- Supports various Whisper model sizes (tiny, base, small, medium, large)
- Optional faster-whisper implementation for better performance
- Customizable keyboard trigger (any letter or function key)
- Automatic WAV file creation in multiple formats
- Debug mode for troubleshooting
- Microphone testing capability

## Installation

1. Ensure system dependencies are installed:

```bash
sudo apt install -y python3-pip ffmpeg xclip xdotool
```

2. Clone or download this repository to a location of your choice

3. Make the scripts executable:

```bash
chmod +x whisper-hotkey whisper-hotkey.py
```

## Usage

Run the script from the command line:

```bash
./whisper-hotkey
```

The wrapper script automatically checks for and installs required Python dependencies.

### Command-line Options

```
usage: whisper-hotkey.py [-h] [--model MODEL] [--sample_rate SAMPLE_RATE] [--key KEY] [--test-mic] [--faster] [--debug]

Hold-to-record speech recognition with Whisper

options:
  -h, --help            show this help message and exit
  --model MODEL         Model size to use (tiny, base, small, medium, large)
  --sample_rate SAMPLE_RATE
                        Sample rate for recording
  --key KEY             Key to hold for recording (single character, e.g., z, a, s)
  --test-mic            Test microphone before starting
  --faster              Use faster-whisper implementation
  --debug               Enable additional debug output
  --paste-method {both,middle,ctrl+v,type}
                        Method to use for pasting text: both, middle (mouse click), 
                        ctrl+v, or type (simulates keyboard)
```

### Examples

```bash
# Use tiny model with 's' key for better speed
./whisper-hotkey --model tiny --key s

# Test your microphone first
./whisper-hotkey --test-mic

# Use faster implementation for better performance
./whisper-hotkey --faster --model small

# Enable debug mode for troubleshooting
./whisper-hotkey --debug

# Use direct keyboard typing for pasting (better compatibility with some apps)
./whisper-hotkey --paste-method type
```

### Keyboard Shortcut Setup

To create a system-wide keyboard shortcut:

1. Go to Settings > Keyboard > Keyboard Shortcuts > Custom Shortcuts
2. Add a new shortcut
3. Name: "Speech to Text"
4. Command: `/full/path/to/whisper-hotkey`
5. Set the shortcut key to your desired key (the same key as your `--key` setting)

## How It Works

1. Press and hold the specified key (default: 'z')
2. Speak while holding the key
3. Release the key to stop recording
4. The tool will process your speech and insert the transcribed text at the current cursor position

The first time you run it, it will download the Whisper model (small = ~460MB).

## Architecture

The code is structured in a clean, object-oriented way:

- `AudioRecorder`: Handles microphone recording
- `AudioProcessor`: Manages audio file creation and validation
- `NotificationManager`: Handles desktop notifications and text insertion
- `TranscriptionEngine`: Performs speech-to-text conversion
- `WhisperHotkey`: Main application class coordinating all components

## Troubleshooting

If you encounter issues:

1. Check your microphone is working with `--test-mic`
2. Enable debug mode with `--debug` for more detailed logs
3. Examine the saved audio file at `~/speech/last_recording.wav`
4. Try different model sizes, starting with `tiny` for faster results
5. Consider using `--faster` for improved performance

### Text Insertion Issues

If transcribed text doesn't get inserted into your application:

1. Try a different paste method:
   - `--paste-method middle` for terminal and X11 apps
   - `--paste-method ctrl+v` for most GUI applications
   - `--paste-method type` for direct keyboard typing (slowest but most compatible)

2. Make sure you have proper permissions:
   - Some applications may block programmatic pasting for security reasons
   - You might need to run with elevated permissions for system applications

3. Verify xdotool and xclip are working:
   ```bash
   echo "test" | xclip -selection clipboard
   xdotool key ctrl+v
   ```

## Notes

- The script requires an active internet connection for the initial model download
- All processing happens locally on your computer after the initial download
- The "small" model offers a good balance between accuracy and speed/resource usage
- For more accuracy, use the "medium" or "large" models (but they require more RAM)
- For faster response, use the "tiny" model (less accurate but very fast)