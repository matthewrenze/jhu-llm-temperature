class ResultsRow:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.model_name = ""
        self.agent_name = ""
        self.exam_name = ""
        self.temperature = 0.0
        self.questions = 0
        self.accuracy = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.tokens_per_question = 0.0
        self.total_cost = 0.0
        self.cost_per_question = 0.0
        self.runtime = 0.0
        self.runtime_per_question = 0.0