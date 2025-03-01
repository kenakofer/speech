# Whisper Hold-to-Speak

A hold-to-speak speech recognition tool using OpenAI's Whisper. Press and hold a key (default: 'z') to record, and release to insert the transcribed text at the cursor position.

## Installation

1. Copy these scripts to a location of your choice.

2. Install the required dependencies:

```bash
sudo apt update
sudo apt install -y python3-pip ffmpeg python3-pyaudio xclip xdotool
pip install --user openai-whisper sounddevice numpy pynput
```

3. Make the scripts executable:

```bash
chmod +x whisper-hotkey.py whisper-hotkey
```

## Usage

### Command Line

Run the script from the command line:

```bash
./whisper-hotkey
```

Options:
- `--model MODEL`: Choose the Whisper model size (tiny, base, small, medium, large). Default: small
- `--key KEY`: Change the hotkey (e.g., 'z', 'a', 's', or 'f13' for function keys). Default: z
- `--test-mic`: Test microphone before starting
- `--faster`: Use faster-whisper implementation (more efficient, requires extra package)
- `--debug`: Enable additional debug output

Examples:
```bash
# Use tiny model with 's' key for better speed
./whisper-hotkey --model tiny --key s

# Test your microphone first
./whisper-hotkey --test-mic

# Use faster implementation for better performance
./whisper-hotkey --faster --model small
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

## Notes

- The script requires an active internet connection for the initial model download
- All processing happens locally on your computer after the initial download
- The "small" model offers a good balance between accuracy and speed/resource usage