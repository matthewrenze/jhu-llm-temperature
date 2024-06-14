import os
import pandas as pd
from experiments.experiment import Experiment
from logs.log import Log
from experiments.results_row import ResultsRow

class ResultsTable:

    def has_match(self, file_path: str, experiment: Experiment) -> bool:

        if not os.path.exists(file_path):
            return False

        table = pd.read_csv(file_path)

        matches = table[
            (table["Model Name"] == experiment.model_name) &
            (table["Agent Name"] == experiment.agent_name) &
            (table["Exam Name"] == experiment.exam_name) &
            (table["Temperature"] == experiment.temperature)]

        return len(matches) > 0


    def create_result(self, experiment: Experiment, details: pd.DataFrame, pricing: dict) -> ResultsRow:

        result = ResultsRow()

        # Get the metadata
        result.start_time = experiment.start_time.strftime("%Y-%m-%d %H:%M:%S")
        result.end_time = experiment.end_time.strftime("%Y-%m-%d %H:%M:%S")
        result.model_name = experiment.model_name
        result.agent_name = experiment.agent_name
        result.exam_name = experiment.exam_name
        result.temperature = experiment.temperature

        # Get the performance metrics
        result.questions = len(details)
        result.num_correct = details.score.sum()
        result.num_incorrect = len(details) - result.num_correct
        result.num_errors = details.error.apply(lambda x: x.strip() != "").sum()
        result.accuracy = details.score.sum() / result.questions

        # Get the number of tokens
        result.input_tokens = details.input_tokens.sum()
        result.output_tokens = details.output_tokens.sum()
        result.total_tokens = details.total_tokens.sum()
        result.tokens_per_question = result.total_tokens / result.questions

        # Get the cost
        price = pricing[experiment.model_name]
        input_cost = price[0] * result.input_tokens / 1000
        output_cost = price[1] * result.output_tokens / 1000
        result.total_cost = input_cost + output_cost
        result.cost_per_question = result.total_cost / result.questions

        # Get the runtime
        result.runtime = (experiment.end_time - experiment.start_time).total_seconds()
        result.runtime_per_question = result.runtime / result.questions

        return result

    def save_result(self, results_file_path: str, result: ResultsRow):

        # Load or create the experiments file
        if os.path.exists(results_file_path):
            results = pd.read_csv(results_file_path)
        else:
            results = pd.DataFrame()

        # Create a dictionary from the result object
        result_dict = result.__dict__
        result_dict = {key.replace("_", " ").title(): value for key, value in result_dict.items()}
        result_dict = {key.replace("Per", "per"): value for key, value in result_dict.items()}

        # Append the experiments to the existing experiments file
        results = results._append(result_dict, ignore_index=True)

        # Sort the results
        results = results.sort_values(by=["Model Name", "Agent Name", "Exam Name", "Temperature"])

        # Save the experiments file
        # Note: If the file is locked, then save with a unique name
        try:
            results.to_csv(results_file_path, index=False)
        except:
            date_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
            results_file_path = f"{results_file_path}_{date_time}.csv"
            results.to_csv(results_file_path, index=False)

    def log_result(self, log: Log, result: ResultsRow) -> None:
        result_text = (
            f'Start Time: {result.start_time}\n'
            f'End Time: {result.end_time}\n'
            f'Model Name: {result.model_name}\n'
            f'Agent Name: {result.agent_name}\n'
            f'Exam Name: {result.exam_name}\n'
            f'Temperature: {result.temperature}\n'
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

