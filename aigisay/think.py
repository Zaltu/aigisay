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

    def append_user_message(self, text):
        """
        Adds a user message to the end of the history.

        :param str text: message text to append
        """
        self.history.append({
            "role": "user",
            "content": text
        })

    def chat(self):
        """
        Trigger the bot to actually say something. This heavily relies on the assumption that
        something has been added to the message history at some point.

        :returns: chatbot result
        :rtype: ollama.ChatMessage
        """
        if not self.history:
            # This is being called with absolutely no history whatsoever. Just default to an introduction.
            self.history.append({
                "role": "user",
                "content": "Introduce yourself."
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
