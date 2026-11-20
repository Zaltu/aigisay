from omnivoice import OmniVoice
import soundfile as sf
import torch

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)


###### To save voice sample
aigis_voice_sample = model.create_voice_clone_prompt(
    ref_audio="aigisSample.wav",
    ref_text="I drew a paper fortune and it said \"Great Luck\". I'm sure that means everything will work out for the best."    
)
aigis_voice_sample.save("aigis_voice_sample.pt")


###### To generate audio

# audio = model.generate(
#     text="I'm going to punch your face. [pause] In the face.",
#     ref_audio="aigisSample.wav",
#     ref_text="I drew a paper fortune and it said \"Great Luck\". I'm sure that means everything will work out for the best."
# )

# sf.write("out.wav", audio[0], 24000)
