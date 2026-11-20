from omnivoice import OmniVoice, VoiceClonePrompt
import soundfile as sf
from torch import float16

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=float16
)

voice_sample = VoiceClonePrompt.load("aigis_voice_sample.pt")
audio = model.generate("You have the potential to grow even stronger.", voice_clone_prompt=voice_sample)
sf.write("out2.wav", audio[0], 24000)