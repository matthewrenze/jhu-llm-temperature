# Import the packages
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the parameters
model_type = "gpt-3.5"
prompt_type = "Composite"
exam_type = None
input_file = f"../details/all-details.csv"
output_folder = f"../plots"

# Create the agent name
agent_name = prompt_type.replace("_", "-").title()

# Create the output folder
os.makedirs(output_folder, exist_ok=True)

# Load the data
details = pd.read_csv(input_file)

# Filter the data
details = details[details["Model"] == model_type]
details = details[details["Prompt"] == prompt_type]
details = details[details["Exam"] != "comprehensive-100"]

# Group by model and temperature and average the accuracy
results = details \
    .groupby(["Temperature"]) \
    .agg({"Accuracy": "mean"}) \
    .reset_index()

# Plot the accuracy by temperature with 95% confidence intervals
plt.figure(figsize=(10, 5))
sns.lineplot(
    x="Temperature",
    y="Accuracy",
    hue=("Model"),
    data=details,
    marker="o",
    errorbar=("ci", 95))
plt.title(f"Accuracy by Temperature for GPT-3.5")
plt.xlabel("Temperature")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{output_folder}/accuracy-by-temperature.png")
plt.show()