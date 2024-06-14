# Import the packages
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import shared

# Hide future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the results
details = pd.read_csv(f"../data/results/all-details.csv")

# Filter the results
details = details[details["Model Name"] == "gpt-35-turbo"]
details = details[details["Agent Name"] == "chain_of_thought"]
details = details[details["Exam Name"] == "comprehensive-100"]
# details = details[details["Temperature"] <= 1.0]

# Process the results
details = shared.set_model_titles(details)
details = shared.set_agent_titles(details)
details = shared.set_exam_titles(details)

# Verify the data
print(f"Models: {len(details['Model Name'].unique())}")
print(f"Agents: {len(details['Agent Name'].unique())}")
print(f"Exams: {len(details['Exam Name'].unique())}")
print(f"Temperatures: {len(details['Temperature'].unique())}")
print(f"Problems IDs: {len(details['Problem ID'].unique())}")

assert len(details['Model Name'].unique()) == 1
assert len(details['Agent Name'].unique()) == 1
assert len(details['Exam Name'].unique()) == 1
assert len(details['Temperature'].unique()) == 17
assert len(details['Problem ID'].unique()) == 100

# Group the results
groups = details \
    .groupby(["Model Title", "Agent Title", "Exam Title", "Temperature"]) \
    .agg({"Accuracy": "mean"}) \
    .reset_index()

# Verify the groups
print(f"Groups: {len(groups)}")
assert(len(groups) == 17)

# Set Matplotlib to use Type 1 fonts
mpl.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'

sns.set_theme(
    style="white",
    font="Times New Roman",
    font_scale=1.5)

# Set the plot size
plt.figure(figsize=(7, 5))

# Set font size again to ensure it's applied
plt.rc('font', size=20)

# Plot the responses for different events and regions
sns.lineplot(
    x="Temperature",
    y="Accuracy",
    hue="Model Title",
    style="Model Title",
    data=groups,
    marker="o")

# Add a vertical line at 1.0
plt.axvline(x=1.0, color="black", linestyle="--")

# Hide the legend
plt.legend().remove()

# Set the axis limits
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 1.7)
plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
plt.tight_layout()
plt.savefig(f"../data/plots/gpt-35-accuracy-by-temperature-extended.pdf")
plt.show()

