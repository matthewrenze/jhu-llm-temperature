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
table = table[table["Exam Name"] == "comprehensive-100"]
# table = table[table["Temperature"] <= 1.0]

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
assert len(table['Exam Name'].unique()) == 1
assert len(table['Temperature'].unique()) == 17
assert len(table['Problem ID'].unique()) == 99  # Problem 83 is missing

# Group by temperature and aggregate the similarity metrics
similarity_by_temperature = table \
    .groupby(["Model Title", "Agent Title", "Exam Title", "Temperature"]) \
    .agg({
        "Jaccard Similarity": "mean",
        "BoW Similarity": "mean",
        "TF-IDF Similarity": "mean",
        "Levenshtein Similarity": "mean",
        "BLEU Score": "mean",
        "SBERT Similarity": "mean"}) \
    .reset_index()

# Set Matplotlib to use Type 1 fonts
mpl.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'

sns.set_theme(
    style="white",
    font="Times New Roman",
    font_scale=1.5)

# Melt the data
similarity_by_temperature = similarity_by_temperature.melt(
    id_vars=["Exam Title", "Temperature"],
    value_vars=[
        "Jaccard Similarity",
        "BoW Similarity",
        "TF-IDF Similarity",
        "Levenshtein Similarity",
        "BLEU Score",
        "SBERT Similarity"],
    var_name="Metric",
    value_name="Value")

# Plot the similarities by temperature
plt.figure(figsize=(7, 5))
sns.set_theme(
    style="white",
    font="Times New Roman",
    font_scale=1.25)
sns.lineplot(
    x="Temperature",
    y="Value",
    hue="Metric",
    style="Metric",
    data=similarity_by_temperature,
    markers=True,
    dashes=False,
    errorbar=None)
plt.ylim(0, 1)
plt.legend(
    title="Metric",
    handlelength=1.5,
    handletextpad=0.25,
    labelspacing=0.25,)

# plt.title("Text Similarity by Temperature and Metric")
plt.xlabel("Temperature")
plt.ylabel("Similarity")
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 1.7)
plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)

# Add a vertical line at 1.0
plt.axvline(x=1.0, color="black", linestyle="--")

# plt.tight_layout()
# plt.savefig(f"{output_folder}/similarity-by-temperature-and-metric.png")
plt.savefig(f"{output_folder}/text-similarity-by-temperature-and-metric.pdf")
plt.show()
