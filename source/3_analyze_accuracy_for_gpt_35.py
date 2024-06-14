import pandas as pd
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import seaborn as sns

# Load the details
details = pd.read_csv(f"../data/results/all-details.csv")

# Filter the details
details = details[details["Model Name"] == "gpt-35-turbo"]
details = details[details["Agent Name"] == "chain_of_thought"]
details = details[details["Exam Name"] != "comprehensive-100"]
details = details[details["Temperature"] <= 1.0]

# Create a subject ID column
details['Subject_ID'] = details['Model Name'] + "-" + details['Agent Name'] + "-" + details['Exam Name'] + "-" + details['Problem ID'].astype(str)

# Verify the data
# NOTE: There should be 1,000 subjects
# NOTE: (i.e., 1 model x 1 agent x 10 exams x 100 problems)
print(f"Models: {len(details['Model Name'].unique())}")
print(f"Agents: {len(details['Agent Name'].unique())}")
print(f"Exams: {len(details['Exam Name'].unique())}")
print(f"Temperatures: {len(details['Temperature'].unique())}")
print(f"Problems IDs: {len(details['Problem ID'].unique())}")
print(f"Subject IDs: {len(details['Subject_ID'].unique())}")
print("")

assert len(details['Model Name'].unique()) == 1
assert len(details['Agent Name'].unique()) == 1
assert len(details['Exam Name'].unique()) == 10
assert len(details['Temperature'].unique()) == 11
assert len(details['Problem ID'].unique()) == 100
assert len(details['Subject_ID'].unique()) == 1000

# Plot the distribution of data as a density plot
plt.figure(figsize=(7, 5))
sns_plot = sns.kdeplot(
    data=details,
    x="Accuracy",
    hue="Temperature",
    fill=True,
    common_norm=False)
plt.xlabel("Accuracy")
plt.xlim(0.0, 1.0)
plt.ylabel("Density")
plt.tight_layout()
sns_plot.legend_.remove()
plt.show()

# Note: The accuracy data are bimodally distributed / non-normal
# So we can't use Repeated Measures ANOVA

# Preparing the data for Kruskal-Wallis Test
grouped_data = [details['Accuracy'][details['Temperature'] == temp] for temp in details['Temperature'].unique()]

# Perform the Kruskal-Wallis Test
stat, p = kruskal(*grouped_data)

# Report the experiments
print(f"H = {stat:.3f}, p={p:.3f}")

