import pandas as pd
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import seaborn as sns
import shared

# Hide future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the details
details = pd.read_csv(f"../data/results/all-details.csv")

# Filter the details
details = details[details["Model Name"] == "gpt-35-turbo"]
# details = details[details["Agent Name"] == "chain_of_thought"]
details = details[details["Exam Name"] == "comprehensive-100"]
details = details[details["Temperature"] <= 1.0]

# Process the details
details = shared.set_model_titles(details)
details = shared.set_agent_titles(details)
details = shared.sort_by_agent(details)

# Create a subject ID column
details['Subject_ID'] = details['Model Name'] + "-" + details['Agent Name'] + "-" + details['Exam Name'] + "-" + details['Problem ID'].astype(str)

# Verify the data
# NOTE: There should be 500 subject IDs
# NOTE: (1 models x 5 agent x 1 exam x 100 problems)
print(f"Models: {len(details['Model Name'].unique())}")
print(f"Agents: {len(details['Agent Name'].unique())}")
print(f"Exams: {len(details['Exam Name'].unique())}")
print(f"Temperatures: {len(details['Temperature'].unique())}")
print(f"Problems IDs: {len(details['Problem ID'].unique())}")
print(f"Subject IDs: {len(details['Subject_ID'].unique())}")
print("")

assert len(details['Model Name'].unique()) == 1
assert len(details['Agent Name'].unique()) == 5
assert len(details['Exam Name'].unique()) == 1
assert len(details['Temperature'].unique()) == 11
assert len(details['Problem ID'].unique()) == 100
assert len(details['Subject_ID'].unique()) == 500

# Loop through each agent/prompt
for agent_title in details['Agent Title'].unique():

    # Filter the details
    agent_details = details[details["Agent Title"] == agent_title]

    # Note: The accuracy data are bimodally distributed at 0 and 1 (i.e. not normally distributed)
    # So we can't use Repeated Measures ANOVA

    # Preparing the data for Kruskal-Wallis Test
    grouped_data = [agent_details['Accuracy'][agent_details['Temperature'] == temp] for temp in agent_details['Temperature'].unique()]

    # Perform the Kruskal-Wallis Test
    stat, p = kruskal(*grouped_data)

    # Report the experiments
    print(f"{agent_title}, H = {stat:.3f}, p = {p:.3f}")

