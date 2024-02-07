import json

class Problem:
    def __init__(self, problem_json):
        for key, value in problem_json.items():
                setattr(self, key, value)

def load_exam(file_path: str)-> list[Problem]:
    exam = []
    with open(file_path, "r", encoding="utf8") as input_file:
        for line in input_file:
            problem = json.loads(line.strip())
            exam.append(Problem(problem))
    return exam

def get_problem_text(problem: Problem)-> str:
    problem_text = ""
    if (problem.topic is not None and problem.topic != ""):
        problem_text += f"Topic: {problem.topic}\n"

    if (problem.context is not None and problem.context != ""):
        problem_text += f"Context: {problem.context}\n"

    problem_text += f"Question: {problem.question}\n"

    choices = problem.choices
    if len(choices) > 0:
        problem_text += f"Choices:\n"
        for choice in choices:
            problem_text += f"  {choice}: {choices[choice]}\n"

    return problem_text
