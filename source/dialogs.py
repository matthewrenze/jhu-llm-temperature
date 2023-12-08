from logs import Log

class Dialog(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def append(self, text: str):
        super().append(text)

    def __str__(self):
        return "\n".join(self)