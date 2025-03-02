# SimplifiedWhisper

A minimal speech-to-text utility that transcribes audio with a hotkey.

## Supported Platforms

- Linux (uses xdotool for typing)
- May work on macOS with appropriate modifications
- Not supported on Windows (requires alternative to xdotool)

## Features

- Press F13 to record audio, release to transcribe
- Uses OpenAI's Whisper model (small) via faster-whisper for accurate transcription
- Automatically types transcribed text where your cursor is positioned
- Minimal overhead and simple interface

## Requirements

- Python 3.6+
- Linux with X11 (for xdotool)
- Audio input device

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/simplified-whisper.git
   cd simplified-whisper
   ```

2. Install dependencies:
   ```
   pip install pynput sounddevice numpy soundfile faster-whisper
   sudo apt install xdotool
   ```

## Usage

Run the script:
```
python simplified-whisper.py
```

1. Press and hold the F13 key to start recording
2. Speak clearly into your microphone
3. Release the F13 key to stop recording
4. The transcribed text will be typed at your cursor position

## Customization

To change the hotkey, modify the `trigger_key` value in the script. For example:

```python
# Change from F13 to F12
self.trigger_key = keyboard.KeyCode(vk=269025152)  # F12 key
```

You can also adjust the Whisper model size by changing the model parameter from "small" to "tiny", "base", "medium", or "large".

## License

MIT License - See LICENSE file for details