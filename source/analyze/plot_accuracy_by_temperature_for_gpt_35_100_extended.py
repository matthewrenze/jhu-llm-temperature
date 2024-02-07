import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
details = pd.read_csv(f"../details/all-details.csv")

# Filter in only GPT-3.5 or GPT-4 model
# details = details[details["Model"] == "gpt-4"]
details = details[details["Model"] == "gpt-3.5"]

# Filter in only the composite prompt
details = details[details["Prompt"] == "Composite"]

# Filter in (or out) only the comprehensive-100 exam
details = details[details["Exam"] == "comprehensive-100"]
# details = details[details["Exam"] != "comprehensive-100"]

# Filter in only the temperatures from 0.0 to 1.0
# details = details[details["Temperature"] <= 1.0]

# Rename the "Problem ID" column
# details = details.rename(columns={"Problem ID": "Problem_ID"})

# Rename gpt to GPT
details["Model"] = details["Model"].str.replace("gpt", "GPT")


# Create a plot of accuracy by temperature with confidence intervals

# Set Matplotlib to use Type 1 fonts
mpl.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'

sns.set_theme(
    style="white",
    font="Times New Roman",
    font_scale=1.5)

# Set the plot size
plt.figure(figsize=(7, 5))

# Set font size again to ensure it's applied
plt.rc('font', size=20)

# Plot the responses for different events and regions
sns.lineplot(
    x="Temperature",
    y="Accuracy",
    hue="Model",
    style="Model",
    data=details,
    errorbar=("ci", 95),
    marker="o")

# Add a vertical line at 1.0
plt.axvline(x=1.0, color="black", linestyle="--")

# Hide the legend
plt.legend().remove()

# Set the axis limits
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 1.4)
plt.tight_layout()
plt.savefig(f"../plots/accuracy-by-temperature-for-gpt-35-100-extended.pdf")
plt.show()

