from logs.log import Log

class Model:
    def __init__(self, model_name: str, temperature: float, log: Log):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = 1000
        self.log = log