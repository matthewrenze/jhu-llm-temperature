# Import the packages
import os
import re
import pandas as pd
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

# Group by model and temperature and average the accuracy
results = details \
    .groupby(["Model", "Temperature"]) \
    .agg({"Accuracy": "mean"}) \
    .reset_index()

# Plot the accuracy by temperature and model
plt.figure(figsize=(10, 5))
sns.lineplot(
    x="Temperature",
    y="Accuracy",
    hue=("Model"),
    data=details,
    marker="o",
    errorbar=None)
plt.title(f"Accuracy by Temperature and Model")
plt.xlabel("Temperature")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.legend(
    title="Model",
    bbox_to_anchor=(1, 1),
    loc='upper left')
plt.subplots_adjust(right=0.85)
plt.savefig(f"{output_folder}/accuracy-by-temperature-and-model.png")
plt.show()
