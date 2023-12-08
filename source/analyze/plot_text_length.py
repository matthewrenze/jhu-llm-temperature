# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the parameters

output_folder = f"../plots"

# Load the data
logs = pd.read_csv("../logs/logs.csv")
details = pd.read_csv("../details/all-details.csv")

# Rename "Problem" to "Problem ID"
logs = logs.rename(columns={"Problem": "Problem ID"})

# Rename agent column to prompt
logs = logs.rename(columns={"Agent": "Prompt"})
logs["Prompt"] = logs["Prompt"].str.replace("Full", "Composite")

# Filter in only the comprehensive-100 exam
# NOTE: Called "composite-100" in the logs
logs["Exam"] = logs["Exam"].str.replace("composite-100", "comprehensive-100")
logs = logs[logs["Exam"] == "comprehensive-100"]

# Filter in only the temperatures from 0.0 to 1.0
logs = logs[logs["Temperature"] <= 1.0]

# Inner Join the data
data = logs.merge(details, on=["Prompt", "Model", "Exam", "Temperature", "Problem ID"])

# Sort the data by agent
data["Prompt"] = pd.Categorical(data["Prompt"], ["Baseline", "Domain-Expert", "Self-Recitation", "Chain-of-Thought", "Composite"])

# Create a line chart of temperature on the x-axis, and token length on the y-axis
plt.figure(figsize=(12, 8))
sns.lineplot(
    data=data,
    x="Temperature",
    y="Token Length",
    hue="Model",
    style="Model",
    markers=True,
    dashes=False,
    palette="colorblind",
    errorbar=None)
plt.title("Token Length by Temperature and Model")
plt.ylim(0, 320)
plt.savefig(f"{output_folder}/token-length-by-temperature-and-model.png")
plt.show()
plt.close()



