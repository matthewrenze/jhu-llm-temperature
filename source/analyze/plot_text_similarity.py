# Import libraries
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Set the parameters
output_folder = f"../plots"

# Load the log data
logs = pd.read_csv("../logs/logs.csv")

# Rename agent column to prompt
logs = logs.rename(columns={"Agent": "Prompt"})
logs["Prompt"] = logs["Prompt"].str.replace("Full", "Composite")

# Filter in only GPT-3.5 models
logs = logs[logs["Model"].str.contains("gpt-3.5")]

# Filter out the comprehensive-100 exam
# NOTE: Called "composite-100" in the logs
logs = logs[logs["Exam"] != "composite-100"]

# Filter in only the temperatures from 0.0 to 1.0
logs = logs[logs["Temperature"] <= 1.0]

# Convert levenshtein distance to Levenshtein similarity
levenshtein = 1 - logs["Levenshtein Distance"]
logs["Levenshtein Similarity"] = (levenshtein - levenshtein.min()) / (levenshtein.max() - levenshtein.min())

# Rename the exam types
logs["Exam"] = logs["Exam"].replace("aqua-rat-100", "AQUA-RAT")
logs["Exam"] = logs["Exam"].replace("arc-challenge-test-100", "ARC Challange Test")
logs["Exam"] = logs["Exam"].replace("hellaswag_val-100", "HellaSwag Val")
logs["Exam"] = logs["Exam"].replace("logiqa-en-100", "LogiQA (English)")
logs["Exam"] = logs["Exam"].replace("lsat-ar-100", "LSAT-AR")
logs["Exam"] = logs["Exam"].replace("lsat-lr-100", "LSAT-LR")
logs["Exam"] = logs["Exam"].replace("lsat-rc-100", "LSAT-RC")
logs["Exam"] = logs["Exam"].replace("medmcqa-dev-100", "MedMCQA Valid")
logs["Exam"] = logs["Exam"].replace("sat-en-100", "SAT-English")
logs["Exam"] = logs["Exam"].replace("sat-math-100", "SAT-Math")

# Sort the data by prompt
logs["Prompt"] = pd.Categorical(logs["Prompt"], ["Baseline", "Domain-Expert", "Self-Recitation", "Chain-of-Thought", "Composite"])

# Sort data by exam
logs["Exam"] = pd.Categorical(logs["Exam"], ["AQUA-RAT", "ARC Challange Test", "HellaSwag Val", "LogiQA (English)", "LSAT-AR", "LSAT-LR", "LSAT-RC", "MedMCQA Valid", "SAT-English", "SAT-Math"])

# Convert exam back to string
logs["Exam"] = logs["Exam"].astype(str)

# Group by temperature and aggregate the BoW and TF-IDF similarity (mean)
similarity_by_temperature = logs \
    .groupby(["Prompt", "Exam", "Temperature"]) \
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

# Define a function to plot the text similarity by temperature
def plot_similarities(table, metric):

    # Get the metric file-name part
    metric_file_name = metric.lower().replace(" ", "-")

    # Create the FacetGrid for the small multiples
    g = sns.FacetGrid(
        table,
        col="Exam",
        col_wrap=5,
        height=1,
        aspect=1)

    # Define the plotting function for the FacetGrid
    def plot_lineplot(*args, **kwargs):
        data = kwargs.pop("data")
        sns.lineplot(
            data=data,
            x="Temperature",
            y=metric,
            hue="Prompt",
            style="Prompt",
            markers=True,
            dashes=False,
            errorbar=None)
        plt.ylim(0, 1)
        plt.legend(title="Prompt")

    # Create the FacetGrid
    g.map_dataframe(plot_lineplot)
    g.fig.set_figwidth(10)
    g.fig.set_figheight(7)
    g.fig.subplots_adjust(top=0.90, wspace=0.2, hspace=0.4, bottom=0.2)
    # g.fig.suptitle('Text Similarity by Temperature by Prompt and Exam', size=11)
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("", "")

    # Set a single x-axis label
    g.fig.text(
        0.515,
        0.08,
        "Temperature",
        ha="center",
        va="center",
        fontsize=16)

    # Set a single y-axis label
    g.fig.text(
        0.015,
        0.55,
        metric,
        ha="center",
        va="center",
        rotation=90,
        fontsize=16)
    # Create a small legend for the chart
    g.add_legend(
        title="Prompt",
        loc="upper center",
        bbox_to_anchor=(0.80, 0.11),
        frameon=True,
        ncol=5,
        fontsize=12,
        handlelength=1.0,
        handletextpad=0.25,
        columnspacing=0.75)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.05)
    #g.savefig(f"{output_folder}/{metric_file_name}-by-temperature-and-exam.png")
    g.savefig(f"{output_folder}/{metric_file_name}-by-temperature-and-exam.pdf")
    plt.show()

# Plot the similarities by temperature and exam
plot_similarities(similarity_by_temperature, "TF-IDF Similarity")

# Filter in only the composite prompt
similarity_by_temperature = similarity_by_temperature[similarity_by_temperature["Prompt"] == "Composite"]

# Melt the data
similarity_by_temperature = similarity_by_temperature.melt(
    id_vars=["Exam", "Temperature"],
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
plt.figure(figsize=(6, 5))
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
plt.xlim(0.0, 1.0)
plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)

# plt.tight_layout()
# plt.savefig(f"{output_folder}/similarity-by-temperature-and-metric.png")
plt.savefig(f"{output_folder}/similarity-by-temperature-and-metric.pdf")
plt.show()
