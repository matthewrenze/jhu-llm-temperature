from agents.agent import Agent

system_prompt = """
You are an expert in {{expertise}}.
Your task is to answer the following multiple-choice questions.
First, you should recite all of the relevant knowledge you have about the question and each option.
Next, you should think step-by-step through the problem to ensure you have the correct answer.
Then, you should critically evaluate your thoughts to identify any flaws in your facts, logic, and reasoning.
Finally, you MUST answer the question using the following format 'Action: Answer("[choice]")'  
The parameter [choice] is the letter or number of the answer you want to select (e.g. "A", "B", "C", or "D")
For example, 'Answer("C")' will select choice "C" as the best answer.
The answer MUST ALWAYS be one of the available choices; it CANNOT be "None of the Above".
If you think the answer is "none of the above", then you MUST select the most likely answer.
"""

example_problem = """
Question: What is the capital of the state where Johns Hopkins University is located?
Choices:
  A: Baltimore
  B: Annapolis
  C: Des Moines
  D: Las Vegas
"""

example_solution = """
Knowledge: 
  Johns Hopkins University is located in Baltimore, Maryland.
  A: Baltimore is a city located in the state of Maryland, but it is not the capital of Maryland.
  B: Annapolis is a the capital of the State of Maryland.
  C: Des Moines is a city located in the State of Iowa, but it is not the capital of Iowa.
  D: Las Vegas is located in the State of Nevada, but it is not the capital of Nevada.
Thought: 
  Johns Hopkins University is located in Baltimore.
  Baltimore is a city located in the state of Maryland.
  The capital of Maryland is Baltimore.
  Therefore, the capital of the state where Johns Hopkins University is located is Baltimore.
  The answer is A: Baltimore.
Criticism: 
  You are correct that Johns Hopkins is located in the State of Baltimore. 
  However, the capital of Maryland is Annapolis, not Baltimore.
  So, the correct answer is actually B: Annapolis.
Action: Answer("B")  
"""

class CompositeAgent(Agent):

    def __init__(self, model, expertise, num_choices, log):
        super().__init__(model, expertise, num_choices, log)
        self.system_prompt_template = system_prompt
        self.example_problem = example_problem
        self.example_solution = example_solution
