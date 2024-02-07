import pandas as pd
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
details = pd.read_csv(f"../details/all-details.csv")

# Filter models
# details = details[details["Model"] == "llama-2-7b"]
# details = details[details["Model"] == "llama-2-70b"]
details = details[details["Model"].isin(["llama-2-7b", "llama-2-70b"])]

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