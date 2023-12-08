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

# Create the prompt title
prompt_title = prompt_type.replace("_", "-").title()

# Create the output folder
os.makedirs(output_folder, exist_ok=True)

# Load the data
details = pd.read_csv(input_file)


# Filter by model and agent
details = details[details["Model"] == model_type]
details = details[details["Prompt"] == prompt_type]

# Filter out the comprehensive-100 exam
details = details[details["Exam"] != "comprehensive-100"]

# Filter the temperatures from 0.0 to 1.0
details = details[details["Temperature"] <= 1.0]

# Group by exam and temperature and average the accuracy
results = details \
    .groupby(["Exam", "Temperature"]) \
    .agg({"Accuracy": "mean"}) \
    .reset_index()

# Plot the accuracy by temperature and exam
plt.figure(figsize=(10, 5))
sns.lineplot(
    x="Temperature",
    y="Accuracy",
    hue=("Exam"),
    data=details,
    marker="o",
    errorbar=None)
plt.title(f"Accuracy of GPT-3.5 by Temperature and Exam")
plt.xlabel("Temperature")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.legend(
    title="Exam",
    bbox_to_anchor=(1, 1),
    loc='upper left')
plt.subplots_adjust(right=0.78)
plt.savefig(f"{output_folder}/accuracy-by-temperature-and-exam.png")
plt.show()