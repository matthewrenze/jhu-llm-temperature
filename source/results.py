import os
import pandas as pd
from experiments import Experiment
from details import DetailsTable
from logs import Log

class Result:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.experiment = ""
        self.model = ""
        self.temperature = 0.0
        self.exam = ""
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

def create_result(experiment: Experiment, details: pd.DataFrame, pricing: dict) -> Result:

    # Create the result object
    result = Result()

    # Get the experiment results
    result.start_time = experiment.start_time.strftime("%Y-%m-%d %H:%M:%S")
    result.end_time = experiment.end_time.strftime("%Y-%m-%d %H:%M:%S")
    result.experiment = experiment.name
    result.model = experiment.model
    result.temperature = experiment.temperature
    result.exam = experiment.exam

    # Get the core results
    result.questions = len(details)
    result.accuracy = details.score.sum() / result.questions

    # Get the token results
    result.input_tokens = details.input_tokens.sum()
    result.output_tokens = details.output_tokens.sum()
    result.total_tokens = details.total_tokens.sum()
    result.tokens_per_question = result.total_tokens / result.questions

    # Get the cost results
    price = pricing[experiment.model]
    input_cost = price[0] * result.input_tokens / 1000
    output_cost = price[1] * result.output_tokens / 1000
    result.total_cost = input_cost + output_cost
    result.cost_per_question = result.total_cost / result.questions

    # Get the runtime results
    result.runtime = (experiment.end_time - experiment.start_time).total_seconds()
    result.runtime_per_question = result.runtime / result.questions

    return result

def save_result(results_file_path: str, result: Result):

    # Load or create the results file
    if os.path.exists(results_file_path):
        results = pd.read_csv(results_file_path)
    else:
        results = pd.DataFrame()

    # Create a dictionary from the result object
    result_dict = result.__dict__
    result_dict = {key.replace("_", " ").title(): value for key, value in result_dict.items()}
    result_dict = {key.replace("Per", "per"): value for key, value in result_dict.items()}

    # Append the results to the existing results file
    results = results._append(result_dict, ignore_index=True)

    # Save the results file
    # Note: If the file is locked, then save with a unique name
    try:
        results.to_csv(results_file_path, index=False)
    except:
        date_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file_path = f"{results_file_path}_{date_time}.csv"
        results.to_csv(results_file_path, index=False)

def log_result(log: Log, result: Result) -> None:
    result_text = (
        f'Start Time: {result.start_time}\n'
        f'End Time: {result.end_time}\n'
        f'Experiment: {result.experiment}\n'
        f'Model: {result.model}\n'
        f'Temperature: {result.temperature}\n'
        f'Exam: {result.exam}\n'
        f'Questions: {result.questions}\n'
        f'Accuracy: {result.accuracy:.4f}\n'
        f'Input Tokens: {result.input_tokens}\n'
        f'Output Tokens: {result.output_tokens}\n'
        f'Total Tokens: {result.total_tokens}\n'
        f'Tokens per Question: {result.tokens_per_question}\n'
        f'Total Cost: ${result.total_cost:.2f}\n'
        f'Cost per question: ${result.cost_per_question:.4f}\n'
        f'Total Runtime: {result.runtime:.2f} seconds\n'
        f'Runtime per question: {result.runtime_per_question:.2f} seconds\n')

    log.head(f'### Results ###')
    log.info(result_text)

