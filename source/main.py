# Import 3rd-party packages
import os
import pandas as pd
import openai

# Import local packages/modules
import experiments
import exams
import models
import agents
import actions
import results
import details
import logs

# Set the agents
agent_types = [
    #"baseline",
    #"domain_expert",
    #"self_recitation",
    #"chain_of_thought",
    "composite"
]

# Set the exam paths
exam_paths = [
    ("comprehensive", "comprehensive-100"),
    # ("agi-eval", "aqua-rat-100"),
    # ("agi-eval", "logiqa-en-100"),
    # ("agi-eval", "lsat-ar-100"),
    # ("agi-eval", "lsat-lr-100"),
    # ("agi-eval", "lsat-rc-100"),
    # ("agi-eval", "sat-en-100"),
    # ("agi-eval", "sat-math-100"),
    # ("arc", "arc-challenge-test-100"),  
    # ("hellaswag", "hellaswag_val-100"),
    # ("medmcqa", "medmcqa-dev-100")
]

# Loop through each agent
for agent_type in agent_types:

    # Loop through each exam
    for exam_path in exam_paths:

        # Loop through each temperature
        for i in range(14, 21):

            # Get the experiment start date/time
            start_time = pd.Timestamp.now()

            # Set the experiment parameters
            experiment = experiments.Experiment()
            experiment.start_time = start_time
            experiment.end_time = None
            experiment.agent_type = agent_type
            # experiment.model = "gpt-4"
            experiment.model = "gpt-35-turbo"
            # experiment.model = "llama-2-70b"
            experiment.temperature = temperature = i / 10
            experiment.folder = exam_path[0]
            experiment.exam = exam_path[1]
            experiment.num_choices = 10

            # Set the experiment name based on the temperature
            experiment.name = f"temp-{experiment.temperature:.1f}"

            # Set the Azure AI parameters
            openai.api_type = "azure"
            openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
            openai.api_key = os.getenv("AZURE_OPENAI_KEY")
            openai.api_version = "2023-05-15"

            # Set Azure Open AI pricing (per 1000 tokens)
            pricing = {
                "gpt-35-turbo": (0.0015, 0.002),
                "gpt-4": (0.03, 0.06),
                "llama-2-7b": (0.00, 0.00),
                "llama-2-70b": (0.00, 0.00)}

            # Set file and folder paths
            exam_file_path = f"../exams/{experiment.folder}/{experiment.exam}.jsonl"
            file_name_prefix = start_time.strftime("%Y-%m-%d %H-%M-%S")
            log_file_path = f"../logs/{experiment.agent_type}/{file_name_prefix} - {experiment.name} - {experiment.model} - {experiment.exam}.txt"
            details_file_path = f"../details/{experiment.agent_type}/{file_name_prefix} - {experiment.name} - {experiment.model} - {experiment.exam}.csv"
            results_file_path = f"../results/{experiment.agent_type}/{experiment.exam}.csv"

            # Create the folders
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(details_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

            # Create the details table
            details_table = details.DetailsTable(details_file_path)
            details_table.create_header()

            # Create the log file
            log_level = logs.LogLevel.INFO
            log = logs.Log(log_file_path, log_level)

            # Load the exam
            exam = exams.load_exam(exam_file_path)

            # Loop through each exam problem
            for j, problem in enumerate(exam):

                # # DEBUG: Only answer questions from a specific topic
                # topic = "college biology"
                # if (problem.topic != topic):
                #     continue

                # Log a status update
                input_file_name = os.path.basename(exam_file_path)
                log.head(f"### Exam: {experiment.exam} | Temperature: {experiment.temperature:.1f} | Problem {j + 1} of {len(exam)} ###")

                # Create the details row
                details_row = details_table.create_row()
                details_row.id = problem.id
                details_row.date_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                details_row.exam = os.path.basename(exam_file_path)
                details_row.source = problem.source
                details_row.source_id = problem.source_id
                details_row.topic = problem.topic
                details_row.correct_answer = problem.answer
                details_row.solution = problem.solution

                # Create the dialog
                dialog = []

                # Create the problem text
                problem_text = exams.get_problem_text(problem)

                # Add the problem to the dialog
                dialog.append(problem_text)

                # Add the problem to the details
                details_row.problem = problem_text

                # Log the problem text
                log.info(problem_text)

                # Create the agent
                factory = agents.agent_factory.AgentFactory()
                agent = factory.create_agent(
                    experiment.model,
                    experiment.agent_type,
                    experiment.temperature,
                    details_row.topic,
                    experiment.num_choices,
                    log)

                # Get the agent's response
                responses = agent.request(dialog)

                # Update the details row
                details.update_details_row(details_row, responses)

                # Create a list for the votes
                votes = []

                # Loop through each choice to record the vote
                for k, choice in enumerate(responses.choices):

                    # Get the choice's answer
                    choice_answer = actions.get_answer(choice)

                    # Add the choice to the details
                    setattr(details_row, f"answer_{k + 1}", choice_answer)

                    # Add the choice to the votes
                    votes.append(choice_answer)

                    # Log the choice's resonse
                    log.info(f"Response {k}:\n{choice}")

                # If there are no votes:
                if (len(votes) == 0):

                    # Then, the final answer to None
                    final_answer = "[None]"
                else:

                    # Else, perform a majority vote
                    final_answer = max(set(votes), key=votes.count)

                # Add the final answer to the details
                details_row.agent_answer = final_answer

                # Calculate the score
                is_correct = details_row.agent_answer == details_row.correct_answer
                details_row.score = 1 if is_correct else 0

                # Add the details row to the details table
                details_table.add_row(details_row)

                # Log the details
                details.log_detail_summary(log, details_row)

            # End the experiment
            experiment.end_time = pd.Timestamp.now()

            # Create the result
            result = results.create_result(experiment, details_table.table, pricing)

            # Save the result
            results.save_result(results_file_path, result)

            # Log the results
            results.log_result(log, result)

            # Close the files
            details_table.close()
            log.close()
