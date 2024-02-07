# Import the packages
import os
import re
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Set the parameters
model_type = None
prompt_type = "Composite"
exam_type = "comprehensive-100"
input_file = f"../details/all-details.csv"
output_folder = f"../plots"

# Create the agent name
agent_name = prompt_type.replace("_", "-").title()

# Create the output folder
os.makedirs(output_folder, exist_ok=True)

# Load the data
details = pd.read_csv(input_file)

# Filter by prompt and exam type
details = details[details["Prompt"] == prompt_type]
details = details[details["Exam"] == exam_type]

# Filter the temperatures from 0.0 to 1.0
details = details[details["Temperature"] <= 1.0]

# Rename model names
details["Model"] = details["Model"].str.replace("gpt", "GPT")
details["Model"] = details["Model"].str.replace("llama", "Llama")

# Group by model and temperature and average the accuracy
results = details \
    .groupby(["Model", "Temperature"]) \
    .agg({"Accuracy": "mean"}) \
    .reset_index()

# Plot the accuracy by temperature and model
plt.figure(figsize=(6, 5))

# Set Matplotlib to use Type 1 fonts
mpl.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'

sns.set_theme(
    style="white",
    font="Times New Roman",
    font_scale=1.5)

sns.lineplot(
    x="Temperature",
    y="Accuracy",
    hue="Model",
    style="Model",
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

# Shrink the legend to fit the plot
plt.legend(
    title="Model",
    #bbox_to_anchor=(1, 1),
    loc="lower center",
    ncol=4,
    fontsize=12,
    handlelength=1,
    handletextpad=0.5,
    columnspacing=0.75)
# plt.subplots_adjust(right=0.77)
# plt.savefig(f"{output_folder}/accuracy-by-temperature-and-model.png")
plt.savefig(f"{output_folder}/accuracy-by-temperature-and-model.pdf")
plt.show()
