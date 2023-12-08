# Import the packages
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the parameters
model_type = "gpt-3.5"
prompt_type = None
exam_type = None
input_file = f"../details/all-details.csv"
output_folder = f"../plots"

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

# Sort the data by agent
details["Prompt"] = pd.Categorical(details["Prompt"], ["Baseline", "Domain-Expert", "Self-Recitation", "Chain-of-Thought", "Composite"])

# Group by prompt and temperature and average the accuracy
results = details \
    .groupby(["Prompt", "Temperature"]) \
    .agg({"Accuracy": "mean"}) \
    .reset_index()

# Plot the accuracy by temperature and prompt
plt.figure(figsize=(10, 5))
sns.lineplot(
    x="Temperature",
    y="Accuracy",
    hue=("Prompt"),
    data=details,
    marker="o",
    errorbar=None)
plt.title(f"Accuracy of GPT-3.5 by Temperature and Prompt")
plt.xlabel("Temperature")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
# Adjust the legend
plt.legend(
    title="Prompt",
    bbox_to_anchor=(1, 1),
    loc='upper left')
plt.subplots_adjust(right=0.75)
plt.savefig(f"{output_folder}/accuracy-by-temperature-and-prompt.png")
plt.show()