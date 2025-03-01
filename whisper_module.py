"""
Module to expose classes from whisper-hotkey.py for testing purposes.
This is a helper module to make the classes in whisper-hotkey.py importable 
without having to execute the script.
"""
import sys
import os
from pathlib import Path

# Add the directory containing whisper-hotkey.py to the path
sys.path.insert(0, str(Path(__file__).parent))

# Execute the script but catch the import
with open(os.path.join(Path(__file__).parent, 'whisper-hotkey.py'), 'r') as f:
    code = compile(f.read(), 'whisper-hotkey.py', 'exec')
    namespace = {}
    exec(code, namespace)

# Import all the classes and functions from the script
AudioRecorder = namespace.get('AudioRecorder')
AudioProcessor = namespace.get('AudioProcessor')
NotificationManager = namespace.get('NotificationManager')
TranscriptionEngine = namespace.get('TranscriptionEngine')
WhisperHotkey = namespace.get('WhisperHotkey')
parse_arguments = namespace.get('parse_arguments')
main = namespace.get('main')