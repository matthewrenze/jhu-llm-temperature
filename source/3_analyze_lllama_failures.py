# Import the packages
import pandas as pd

# Load the data
details = pd.read_csv(f"../data/results/all-details.csv")

# Filter models
details = details[details["Model Name"] == "llama-2-7b-chat"]
# details = details[details["Model Name"] == "llama-2-70b-chat"]

# Count the number of problems
total_problems = len(details)

# Compute the total number of answers
total_answers = total_problems * 10

# Project just the columns "Answer 1" through "Answer 10"
answers = details[['Answer 1', 'Answer 2', 'Answer 3', 'Answer 4', 'Answer 5', 'Answer 6', 'Answer 7', 'Answer 8', 'Answer 9', 'Answer 10']]

# Count the number of missing answers
no_answers = answers.isna().sum().sum()

# Calculate the percentage of incorrectly formatted answers
no_answers_percent = (no_answers / total_answers) * 100

# Calculate the percent of correct answers
correct_answers_percent = details['Score'].mean() * 100

# Calculate the percent of incorrect answers
incorrect_answers_percent = 100 - correct_answers_percent

# Calculate the percent of correct format but incorrect answers
correct_format_incorrect_answers_percent = incorrect_answers_percent - no_answers_percent

print(f"Total Problems: {total_problems}")
print(f"Total Answers: {total_answers}")
print(f"No Answers: {no_answers} ({no_answers_percent:.2f}%)")
print(f"Correct Answers: {correct_answers_percent:.2f}%")
print(f"Incorrect Answers: {incorrect_answers_percent:.2f}%")
print(f"Correct Format, Incorrect Answers: {correct_format_incorrect_answers_percent:.2f}%")
