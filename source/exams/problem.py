class Problem:
    def __init__(self, problem_json):
        for key, value in problem_json.items():
                setattr(self, key, value)