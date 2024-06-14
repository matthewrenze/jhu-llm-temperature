# Import the packages
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import shared

# Hide future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set the parameters
input_file = f"../data/results/all-details.csv"
output_folder = f"../data/plots"

# Create the output folder
os.makedirs(output_folder, exist_ok=True)

# Load the results
details = pd.read_csv(input_file)

# Filter the data
details = details[details["Model Name"] == "gpt-35-turbo"]
details = details[details["Agent Name"] == "chain_of_thought"]
details = details[details["Exam Name"] != "comprehensive-100"]
details = details[details["Temperature"] <= 1.0]

# Process the results
details = shared.set_model_titles(details)
details = shared.set_agent_titles(details)
details = shared.set_exam_titles(details)
details = shared.sort_by_exam(details)

# Verify the data
print(f"Models: {len(details['Model Name'].unique())}")
print(f"Agents: {len(details['Agent Name'].unique())}")
print(f"Exams: {len(details['Exam Name'].unique())}")
print(f"Temperatures: {len(details['Temperature'].unique())}")
print(f"Problems IDs: {len(details['Problem ID'].unique())}")

assert len(details['Model Name'].unique()) == 1
assert len(details['Agent Name'].unique()) == 1
assert len(details['Exam Name'].unique()) == 10
assert len(details['Temperature'].unique()) == 11
assert len(details['Problem ID'].unique()) == 100

# Group the results
groups = details \
    .groupby(["Model Title", "Agent Title", "Exam Title", "Temperature"]) \
    .agg({"Accuracy": "mean"}) \
    .reset_index()

# Verify the groups
print(f"Groups: {len(groups)}")
assert(len(groups) == 110)

# Set Matplotlib to use Type 1 fonts
mpl.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'

sns.set_theme(
    style="white",
    font="Times New Roman",
    font_scale=1.5)

plt.figure(figsize=(7, 5))
sns.lineplot(
    x="Temperature",
    y="Accuracy",
    hue="Exam Title",
    style="Exam Title",
    data=details,
    markers=True,
    dashes=False,
    errorbar=None)
# plt.title(f"Accuracy by Temperature and Model")
plt.xlabel("Temperature")
plt.ylabel("Accuracy")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 0.80, 1])

# Move the legend to the right margin
plt.legend(
    title="Exam",
    bbox_to_anchor=(1.03, 1.025),
    loc="upper left",
    ncol=1,
    title_fontsize=16,
    fontsize=12,
    handlelength=1,
    handletextpad=0.5,
    columnspacing=0.75)
# plt.subplots_adjust(right=0.77)
# plt.savefig(f"{output_folder}/accuracy-by-temperature-and-model.png")
plt.savefig(f"{output_folder}/gpt-35-accuracy-by-temperature-and-exam.pdf")
plt.show()

