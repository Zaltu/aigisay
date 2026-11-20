from ollama import chat
from pprint import pprint as pp

DEFAULT_MODEL = "aigiscord"

class AigisBrain():
    def __init__(self, model=DEFAULT_MODEL, history=[]):
        self.model = model
        self.history = history

    def import_history(self, messages):
        """
        This is a very simple extension of the existing history that expects to be given a properly
        formatted list of Ollama-compliant messages. TODO make this more caller-friendly.
        """
        self.history.extend(messages)
        self._truncate_history()

    def chat(self, text):
        self.history.append({
            "role": "user",
            "content": text
        })
        response = chat(
            model="aigiscord",
            messages=self.history
        )
        self.history.append(response.message)
        self._truncate_history()
        return response

    def _truncate_history(self):
        if len(self.history) > 20:
            # TODO I could just ask it to summarize the prior conversation a this point to boil it down
            # and maybe keep some long-term memory, but for now just cut off after a point.
            self.history = self.history[-20:]  # cut the first message, not the latest
