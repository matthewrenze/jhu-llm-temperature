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
# details = details[details["Prompt"] == "Composite"]

# Filter in (or out) only the comprehensive-100 exam
# details = details[details["Exam"] == "comprehensive-100"]
details = details[details["Exam"] != "comprehensive-100"]

# Filter in only the temperatures from 0.0 to 1.0
details = details[details["Temperature"] <= 1.0]

# Rename the "Problem ID" column
# details = details.rename(columns={"Problem ID": "Problem_ID"})

# Rename gpt to GPT
details["Model"] = details["Model"].str.replace("gpt", "GPT")


# Create a plot of accuracy by temperature with confidence intervals

# Set Matplotlib to use Type 1 fonts
mpl.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'

sns.set_theme(
    style="whitegrid",
    font="Times New Roman",
    font_scale=1.5)

# Set the plot size
plt.figure(figsize=(6, 5))

# Plot the responses for different events and regions
sns.lineplot(
    x="Temperature",
    y="Accuracy",
    hue="Prompt",
    style="Prompt",
    markers=True,
    dashes=False,
    data=details,
    errorbar=None)

# Create a legend
plt.legend(
    title="Prompt",
    loc="lower right",
    handlelength=1,
    handletextpad=0.5,
    fontsize=14)

# Set the axis limits
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 1.0)
plt.tight_layout()
plt.savefig(f"../plots/accuracy-by-temperature-for-gpt-35-by-prompt.pdf")
plt.show()

