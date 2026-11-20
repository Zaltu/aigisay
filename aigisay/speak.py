from omnivoice import OmniVoice, VoiceClonePrompt
from torch import float16
import soundfile as sf

from pathlib import Path

VOICE_DATA_PATH = Path(__file__).parent / "data"
VOICE_MODULES = {
    "aigis": "aigis_voice_sample.pt",

    "default": "aigis_voice_sample.pt"
}

MODEL = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=float16
)

def speak(text, voice=None, path=None, filename=None):
    """
    Generate audio using an existing cloned voice sample, if one exists.
        - If the requested voice does not exist, falls back to Aigis' default voice.
        - If no voice is requested, uses Omnivoice's default settings to generate a random voice.
    If a path is provided, a .wav will be written to that path with an optional filename. Otherwise, the raw
    audio output will be returned.
        - If a path is provided without a filename, the filename will automatically be set to the
        first 10 characters of the text.
    
    :param str text: text to be spoken
    :param str voice: name of the voice to be used (VOICE_MODULES) (optional)
    :param str path: path to folder in which to place generated .wav of audio (optional)
    :param str filename: name of .wav file to create (optional)

    :returns: Path to the generated audio file, or raw audio output
    :rtype: str|list
    """
    voice_sample = VoiceClonePrompt.load(VOICE_DATA_PATH / VOICE_MODULES.get(voice, VOICE_MODULES["default"])) if voice else None
    audio = MODEL.generate(text, voice_clone_prompt=voice_sample)

    if path:
        output_filepath = Path(path) / Path(filename+".wav" if filename else f"{text[:10]}.wav")
        sf.write(output_filepath, audio[0], 24000)
        return output_filepath
    return audio
