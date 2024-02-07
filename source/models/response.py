class Response():
    def __init__(self):
        self.has_error = False
        self.choices = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0