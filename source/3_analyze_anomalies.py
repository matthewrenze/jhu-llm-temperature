import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import shared

# Load the details
details = pd.read_csv("../data/results/all-details.csv")

# Filter the details
details = details[details["Model Name"] == "gpt-35-turbo"]
details = details[details["Agent Name"] == "chain_of_thought"]
details = details[details["Exam Name"] == "sat-math-100"]
details = details[details["Temperature"] <= 1.0]

# Process the details
details = shared.set_model_titles(details)
details = shared.set_agent_titles(details)
details = shared.sort_by_agent(details)

# Verify the data
# NOTE: There should be 1,000 subjects
# NOTE: (i.e., 1 model x 1 agent x 10 exams x 100 problems)
print(f"Models: {len(details['Model Name'].unique())}")
print(f"Agents: {len(details['Agent Name'].unique())}")
print(f"Exams: {len(details['Exam Name'].unique())}")
print(f"Temperatures: {len(details['Temperature'].unique())}")
print(f"Problems IDs: {len(details['Problem ID'].unique())}")
print("")

assert len(details['Model Name'].unique()) == 1
assert len(details['Agent Name'].unique()) == 1
assert len(details['Exam Name'].unique()) == 1
assert len(details['Temperature'].unique()) == 11
assert len(details['Problem ID'].unique()) == 100

# Group the details
by_temperature = details \
    .groupby(["Temperature"]) \
    .agg({"Accuracy": "mean"}) \
    .reset_index()

# Plot the score by temperature
plt.figure(figsize=(7, 5))
sns.barplot(
    x="Temperature",
    y="Accuracy",
    data=by_temperature)
plt.xlabel("Temperature")
plt.ylabel("Accuracy")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

# Pivot the details with the problem ID as the index, the temperature as the columns, and the score as the values
pivot_table = details.pivot_table(
    index=["Problem ID"],
    columns=["Temperature"],
    values="Accuracy",
    aggfunc="mean")

# Create a heatmap with the temperature on the x-axis, the problem ID on the y-axis, and the score as the value
plt.figure(figsize=(5, 10))
sns.heatmap(
    pivot_table,
    cmap="coolwarm",
    cbar_kws={'label': 'Accuracy'})
plt.xlabel("Temperature")
plt.ylabel("Problem ID")
plt.tight_layout()
plt.show()

errors = details[details["Error"].notna()]

# Create a subject ID column
details['Subject_ID'] = details['Model Name'] + "-" + details['Agent Name'] + "-" + details['Exam Name'] + "-" + details['Problem ID'].astype(str)
grouped_data = [details['Accuracy'][details['Temperature'] == temp] for temp in details['Temperature'].unique()]

# Perform the Kruskal-Wallis Test
stat, p = kruskal(*grouped_data)
print(f"H = {stat:.3f}, p={p:.3f}")

# Perform the Dunn's Test
from scikit_posthocs import posthoc_dunn
dunn_results = posthoc_dunn(
    details,
    val_col='Accuracy',
    group_col='Temperature')
print(dunn_results)

# Create a heatmap of the Dunn's Test results
# Set the range of the colors from 0.0 to 1.0
plt.figure(figsize=(5, 5))
sns.heatmap(
    dunn_results,
    cmap="coolwarm",
    cbar_kws={'label': 'p-value'},
    vmin=0.0,
    vmax=1.0)
plt.xlabel("Temperature")
plt.ylabel("Temperature")
plt.tight_layout()
plt.show()



