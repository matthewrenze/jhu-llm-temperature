# Import the packages
import os
import pandas as pd

# Set the parameters
input_folder = f"../data/details"
output_folder = f"../data/results"

# Create the output folder
os.makedirs(output_folder, exist_ok=True)

# Create a table for the details
details = pd.DataFrame()

# Get the files recursively
file_paths = []
for file_name in os.listdir(input_folder):

    # Skip non-csv files
    if not file_name.endswith(".csv"):
        continue

    # Skip the all-details file
    if file_name == "all-details.csv":
        continue

    # Display a status update
    print(f"Processing {file_name}...")

    # Get the file name
    file_path = f"{input_folder}/{file_name}"
    file_path = file_path.replace("\\", "/")

    # DEBUG: Skip the file if zero bytes
    if os.path.getsize(file_path) == 0:
        continue

    # Load the file
    details_file = pd.read_csv(file_path)

    # # Rename columns
    # details_file.rename(columns={"Problem Id": "Problem ID"}, inplace=True)
    # details_file.rename(columns={"Source Id": "Source ID"}, inplace=True)

    # Add the file to the table
    details = pd.concat([details, details_file])

# Create Accuracy column
details["Accuracy"] = details.apply(
    lambda row: sum([row[f"Answer {i}"] == row["Correct Answer"] for i in range(1, 11)]) / 10,
    axis=1)

# Sort the details
details = details.sort_values(by=["Model Name", "Agent Name", "Exam Name", "Temperature", "Problem ID"])

# Save the details to a csv file
details.to_csv(f"{output_folder}/all-details.csv", index=False)