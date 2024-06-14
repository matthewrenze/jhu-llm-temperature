# Import libraries
import os
import pandas as pd
from agents.agent_factory import AgentFactory
from agents.actions import get_answer
from models.model_factory import ModelFactory
from exams.exam import load_exam, get_problem_text
from experiments.experiment import Experiment
from experiments.details_table import DetailsTable
from experiments.results_table import ResultsTable
from experiments.responses_table import ResponsesTable
from experiments.responses_row import ResponsesRow
from logs.log_level import LogLevel
from logs.log import Log

# Set the models
model_names = [
    "gpt-35-turbo",
    # "gpt-4",
    # "gpt-4o",
    # "llama-2-7b-chat",
    # "llama-2-70b-chat",
    # "mistral-large",
    # "cohere-command-r-plus",
    # "gemini-1.0-pro",
    # "gemini-1.5-pro-preview-0409",
    # "claude-3-opus-20240229"
]

# Set the agents
agent_names = [
    # "baseline",
    # "domain_expert",
    # "self_recitation",
    "chain_of_thought",
    # "composite"
]

# Set the exam paths
exam_names = [
    "comprehensive-100",
    # "aqua-rat-100",
    # "arc-challenge-test-100",
    # "hellaswag_val-100",
    # "logiqa-en-100",
    # "lsat-ar-100",
    # "lsat-lr-100",
    # "lsat-rc-100",
    # "medmcqa-dev-100",
    # "sat-en-100",
    # "sat-math-100",
]

# Set pricing (per 1000 tokens)
model_pricing = {
    "gpt-35-turbo": (0.0015, 0.002),
    "gpt-4": (0.03, 0.06),
    "gpt-4o": (0.005, 0.015),
    "llama-2-7b-chat": (0.00052, 0.00067),
    "llama-2-70b-chat": (0.00154, 0.00177),
    "mistral-large": (0.008, 0.024),
    "cohere-command-r-plus": (0.003, 0.015),
    "gemini-1.0-pro": (0.0005, 0.0015),
    "gemini-1.5-pro-preview-0409": (0.007, 0.021),
    "claude-3-opus-20240229": (0.015, 0.075)}

# Loop through each model
for model_name in model_names:

    # Loop through each agent
    for agent_name in agent_names:

        # Loop through each exam
        for exam_name in exam_names:

            # Loop through each temperature
            for i in range(0, 11):
            # for i in range(11, 17):

                # Get the experiment start date/time
                start_time = pd.Timestamp.now()

                # Set the experiment parameters
                experiment = Experiment()
                experiment.start_time = start_time
                experiment.end_time = None
                experiment.model_name = model_name
                experiment.agent_name = agent_name
                experiment.exam_name = exam_name
                experiment.temperature = temperature = i / 10
                experiment.num_choices = 10

                # Set the experiment name based on the temperature
                experiment.name = f"{experiment.model_name} - {experiment.agent_name} - {experiment.exam_name} - temp-{experiment.temperature:.1f}"

                # Set file and folder paths
                exam_file_path = f"../data/exams/{experiment.exam_name}.jsonl"
                log_file_path = f"../data/logs/{experiment.name}.txt"
                details_file_path = f"../data/details/{experiment.name}.csv"
                results_file_path = f"../data/results/results.csv"
                responses_file_path = f"../data/responses/{experiment.name}.csv"

                # Check to see if this experiment has already been run
                results_table = ResultsTable()
                if results_table.has_match(results_file_path, experiment):
                    # Get permissions from the user to continue via the console
                    print(f"Experiment already run: {experiment.name}")
                    print("Do you want to re-run this experiment?")
                    response = input("Enter 'y' to continue or 'n' to skip: ")
                    if response != "y":
                        continue

                # Create the folders
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                os.makedirs(os.path.dirname(details_file_path), exist_ok=True)
                os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

                # Create the details table
                details_table = DetailsTable(details_file_path)
                details_table.create_header()

                # Create the response table
                responses_table = ResponsesTable()

                # Create the log file
                log_level = LogLevel.INFO
                log = Log(log_file_path, log_level)

                # Load the exam
                exam = load_exam(exam_file_path)

                # Loop through each exam problem
                for j, problem in enumerate(exam):

                    # # DEBUG: Only answer the first n questions
                    # if j >= 3:
                    #     break

                    # Log a status update
                    input_file_name = os.path.basename(exam_file_path)
                    log.head(f"### Model: {experiment.model_name} | Agent: {experiment.agent_name} | Exam: {experiment.exam_name} | Temperature: {experiment.temperature:.1f} | Problem {j + 1} of {len(exam)} ###")

                    # Create the details row
                    details_row = details_table.create_row()
                    details_row.date_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    details_row.model_name = experiment.model_name
                    details_row.agent_name = experiment.agent_name
                    details_row.exam_name = experiment.exam_name
                    details_row.temperature = experiment.temperature
                    details_row.problem_id = problem.id
                    details_row.source = problem.source
                    details_row.source_id = problem.source_id
                    details_row.topic = problem.topic
                    details_row.correct_answer = problem.answer
                    details_row.solution = problem.solution

                    # Create the dialog
                    dialog = []

                    # Create the problem text
                    problem_text = get_problem_text(problem)

                    # Add the problem to the dialog
                    dialog.append(problem_text)

                    # Add the problem to the details
                    details_row.problem = problem_text
                    log.info(problem_text)

                    # Create the model
                    model_factory = ModelFactory()
                    model = model_factory.create(
                        experiment.model_name,
                        experiment.temperature,
                        log)

                    # Create the agent
                    agent_factory = AgentFactory()
                    agent = agent_factory.create(
                        model,
                        experiment.agent_name,
                        details_row.topic,
                        experiment.num_choices,
                        log)

                    # Get the agent's response
                    responses = agent.request(dialog)

                    # Update the details row
                    details_table.update_details_row(details_row, responses)

                    # Create a list for the votes
                    votes = []

                    # Loop through each choice to record the vote
                    for k, choice in enumerate(responses.choices):
                        choice_answer = get_answer(choice)
                        setattr(details_row, f"answer_{k + 1}", choice_answer)
                        votes.append(choice_answer)
                        log.info(f"Response {k}:\n{choice}")

                    # If there are no votes:
                    # Then, the final answer to None
                    # Else, perform a majority vote
                    if len(votes) == 0:
                        final_answer = "[None]"
                    else:
                        final_answer = max(set(votes), key=votes.count)

                    # Add the final answer to the details
                    details_row.agent_answer = final_answer

                    # Calculate the score
                    is_correct = details_row.agent_answer == details_row.correct_answer
                    details_row.score = 1 if is_correct else 0

                    # Add the details row to the details table
                    details_table.add_row(details_row)
                    details_table.log_detail_summary(log, details_row)

                    # Save the responses
                    for k, response in enumerate(responses.choices):
                        responses_row = ResponsesRow()
                        responses_row.model_name = experiment.model_name
                        responses_row.agent_name = experiment.agent_name
                        responses_row.exam_name = experiment.exam_name
                        responses_row.temperature = experiment.temperature
                        responses_row.problem_id = problem.id
                        responses_row.response_id = k
                        responses_row.text = response
                        responses_table.add_row(responses_row)

                # End the experiment
                experiment.end_time = pd.Timestamp.now()

                # Save the result
                results_table = ResultsTable()
                results_row = results_table.create_result(experiment, details_table.table, model_pricing)
                results_table.save_result(results_file_path, results_row)
                results_table.log_result(log, results_row)

                # Save the responses
                responses_table.save(responses_file_path)

                # Close the files
                details_table.close()
                log.close()
