# Import the packages
import os
import re
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Set the parameters
model_type = "gpt-3.5"
prompt_type = None
exam_type = None
input_file = f"../details/all-details.csv"
output_folder = f"../icml"

# Create the output folder
os.makedirs(output_folder, exist_ok=True)

# Load the data
details = pd.read_csv(input_file)

# Filter by model and agent
details = details[details["Model"] == model_type]

# Filter out the comprehensive-100 exam
details = details[details["Exam"] != "comprehensive-100"]

# Filter the temperatures from 0.0 to 1.0
details = details[details["Temperature"] <= 1.0]

# Rename the exam types
details["Exam"] = details["Exam"].replace("aqua-rat-100", "AQUA-RAT")
details["Exam"] = details["Exam"].replace("arc-challenge-test-100", "ARC Challange Test")
details["Exam"] = details["Exam"].replace("hellaswag_val-100", "HellaSwag Val")
details["Exam"] = details["Exam"].replace("logiqa-en-100", "LogiQA (English)")
details["Exam"] = details["Exam"].replace("lsat-ar-100", "LSAT-AR")
details["Exam"] = details["Exam"].replace("lsat-lr-100", "LSAT-LR")
details["Exam"] = details["Exam"].replace("lsat-rc-100", "LSAT-RC")
details["Exam"] = details["Exam"].replace("medmcqa-dev-100", "MedMCQA Valid")
details["Exam"] = details["Exam"].replace("sat-en-100", "SAT-English")
details["Exam"] = details["Exam"].replace("sat-math-100", "SAT-Math")

# Sort the data by agent
details["Prompt"] = pd.Categorical(details["Prompt"], ["Baseline", "Domain-Expert", "Self-Recitation", "Chain-of-Thought", "Composite"])

# Sort data by exam
details["Exam"] = pd.Categorical(details["Exam"], ["AQUA-RAT", "ARC Challange Test", "HellaSwag Val", "LogiQA (English)", "LSAT-AR", "LSAT-LR", "LSAT-RC", "MedMCQA Valid", "SAT-English", "SAT-Math"])

# Convert exam back to string
details["Exam"] = details["Exam"].astype(str)

# Group by prompt and temperature and average the accuracy
results = details \
    .groupby(["Prompt", "Exam", "Temperature"]) \
    .agg({"Accuracy": "mean"}) \
    .reset_index()

# Set Matplotlib to use Type 1 fonts
mpl.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'

sns.set_theme(
    style="white",
    font="Times New Roman",
    font_scale=1.5)

# Create the FacetGrid for the small multiples
g = sns.FacetGrid(
    results,
    col="Exam",
    col_wrap=5,
    height=1,
    aspect=1)

# Define the plotting function for the FacetGrid
def plot_lineplot(*args, **kwargs):
    data = kwargs.pop("data")
    sns.lineplot(
        x="Temperature",
        y="Accuracy",
        hue="Prompt",
        style="Prompt",
        data=data,
        markers=True,
        dashes=False,
        errorbar=None)
    plt.ylim(0, 1)

# Plot the accuracy by temperature and prompt
g.map_dataframe(plot_lineplot)
# g.fig.suptitle('Accuracy of GPT-3.5 by Temperature, Prompt, and Exam', size=11)
g.fig.set_figwidth(10)
g.fig.set_figheight(7)
g.fig.subplots_adjust(top=0.90, wspace=0.2, hspace=0.4, bottom=0.2)
g.set_titles(col_template="{col_name}")
g.set_axis_labels("", "")

# Set a single x-axis label
g.fig.text(
    0.515,
    0.08,
    "Temperature",
    ha="center",
    va="center",
    fontsize=16)

# Set a single y-axis label
g.fig.text(
    0.015,
    0.55,
    "Accuracy",
    ha="center",
    va="center",
    rotation=90,
    fontsize=16)

# Create a small legend for the chart
g.add_legend(
    title="Prompt",
    loc="upper center",
    bbox_to_anchor=(0.80, 0.11),
    frameon=True,
    ncol=5,
    fontsize=12,
    handlelength=1.0,
    handletextpad=0.25,
    columnspacing=0.75)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, left=0.05)
# g.savefig(f"{output_folder}/accuracy-by-temperature-prompt-and-exam.png")
g.savefig(f"{output_folder}/accuracy-by-temperature-prompt-and-exam.pdf")
plt.show()

