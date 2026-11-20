# Aigisay
Aigisaywhat?  
What?

## TTS
TTS is powered by OmniVoice. Tools directory contains some excessively simple setup to get a voice cloned and a voice sample file generated.

```python
from aigisay.speak import speak
speak("You have selected Aigis' default voice.", voice="aigis", path="./", filename="sample")
```

## Chatbot
Chatbot functionality locally powered by Ollama. Currently a fairly simple loader with basic history retention. Perhaps more to come, perhaps not.

```python
from aigisay.think import AigisBrain

bot = AigisBrain()

response = bot.chat("What's up gamers?")
print(response.message.content)
```

