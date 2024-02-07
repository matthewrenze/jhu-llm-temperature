# Import the packages
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the parameters
input_folder = f"../details"
output_folder = f"../details"

# Create the output folder
os.makedirs(output_folder, exist_ok=True)

# Create a table for the details
details = pd.DataFrame()

# Get the files recursively
file_paths = []
for root, dirs, file_names in os.walk(input_folder):

    # Skip the "_archive" folder
    if "_archive" in root:
        continue

    # Loop through each file
    for file_name in file_names:

        # Skip the all-details file
        if file_name == "all-details.csv":
            continue

        # Get the file name
        file_path = f"{root}/{file_name}"
        file_path = file_path.replace("\\", "/")
        file_paths.append(file_path)

# Include only csv files
file_paths = [file for file in file_paths if file.endswith(".csv")]

# Loop through each file
for file_path in file_paths:

        # Load the file
        details_file = pd.read_csv(file_path)

        # Get the agent from the file path
        prompt_name = re.search(r"details/(.*)/", file_path).group(1)
        prompt_name = prompt_name.replace("_", "-").title()
        prompt_name = prompt_name.replace("-Of-", "-of-")

        # Get the model from the file name
        model_name = re.search(r"temp-\d\.\d - (.*) - (.*)\.csv", file_path).group(1)

        # Rename GPT-3.5
        if model_name == "gpt-35-turbo":
            model_name = "gpt-3.5"

        # Get the temperature from the file name
        temperature = float(re.search(r"temp-(\d\.\d)", file_path).group(1))

        # Add the additional properties to the table
        details_file["Model"] = model_name
        details_file["Prompt"] = prompt_name
        details_file["Temperature"] = temperature

        # Remove file extension from exam
        details_file["Exam"] = details_file["Exam"].str.replace(".jsonl", "")

        # Create the problem ID column
        details_file["Problem ID"] = range(1, 101)

        # If there is no "Source Id" column, then create it
        if "Source Id" not in details_file.columns:
            details_file["Source ID"] = range(1, 101)

        # If there is a source id column, then rename it ("Id" -> "ID")
        if "Source Id" in details_file.columns:
            details_file.rename(columns={"Source Id": "Source ID"}, inplace=True)

        # Add the file to the table
        details = pd.concat([details, details_file])

# Create Accuracy column
details["Accuracy"] = details.apply(
    lambda row: sum([row[f"Answer {i}"] == row["Correct Answer"] for i in range(1, 11)]) / 10,
    axis=1)

# Create title for experiment
details["Title"] = details["Model"] + " - " + details["Prompt"] + " - " + details["Exam"]

# Save the details to a csv file
details.to_csv(f"{output_folder}/all-details.csv", index=False)