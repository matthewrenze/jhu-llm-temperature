# Import libraries
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import shared

# Hide future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set the parameters
input_file = f"../data/results/text-similarity.csv"
output_folder = f"../data/plots"

# Load the text similarity data
table = pd.read_csv(input_file)

# Filter the data
table = table[table["Model Name"] == "gpt-35-turbo"]
table = table[table["Agent Name"] == "chain_of_thought"]
table = table[table["Exam Name"] != "comprehensive-100"]
table = table[table["Temperature"] <= 1.0]

# Convert levenshtein distance to Levenshtein similarity
levenshtein = 1 - table["Levenshtein Distance"]
table["Levenshtein Similarity"] = (levenshtein - levenshtein.min()) / (levenshtein.max() - levenshtein.min())

# Process the data
table = shared.set_model_titles(table)
table = shared.set_agent_titles(table)
table = shared.set_exam_titles(table)

# Verify the data
print(f"Models: {len(table['Model Name'].unique())}")
print(f"Agents: {len(table['Agent Name'].unique())}")
print(f"Exams: {len(table['Exam Name'].unique())}")
print(f"Temperatures: {len(table['Temperature'].unique())}")
print(f"Problems IDs: {len(table['Problem ID'].unique())}")

assert len(table['Model Name'].unique()) == 1
assert len(table['Agent Name'].unique()) == 1
assert len(table['Exam Name'].unique()) == 10
assert len(table['Temperature'].unique()) == 11
assert len(table['Problem ID'].unique()) == 100

# Group by temperature and aggregate the similarity metrics
similarity_by_temperature = table \
    .groupby(["Model Title", "Agent Title", "Exam Title", "Temperature"]) \
    .agg({"TF-IDF Similarity": "mean"}) \
    .reset_index()

# Set Matplotlib to use Type 1 fonts
mpl.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'

sns.set_theme(
    style="white",
    font="Times New Roman",
    font_scale=1.5)

# Plot the similarities by temperature
plt.figure(figsize=(6, 5))
sns.set_theme(
    style="white",
    font="Times New Roman",
    font_scale=1.25)
sns.lineplot(
    x="Temperature",
    y="TF-IDF Similarity",
    hue="Exam Title",
    style="Exam Title",
    data=similarity_by_temperature,
    markers=True,
    dashes=False,
    errorbar=None)
plt.ylim(0, 1)
plt.legend(
    title="Exam",
    handlelength=1.5,
    handletextpad=0.25,
    labelspacing=0.20,)

# plt.title("Text Similarity by Temperature and Exam")
plt.xlabel("Temperature")
plt.ylabel("TF-IDF Similarity")
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 1.0)
plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)

# plt.tight_layout()
plt.savefig(f"{output_folder}/tf-idf-similarity-by-temperature-and-exam.pdf")
plt.show()
