# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.patches as patches

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

# Sort the data by prompt
logs["Prompt"] = pd.Categorical(logs["Prompt"], ["Baseline", "Domain-Expert", "Self-Recitation", "Chain-of-Thought", "Composite"])

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

# Define a function to plot the text similarity by temperature
def plot_similarities(table, metric):

    # Get the metric file-name part
    metric_file_name = metric.lower().replace(" ", "-")

    # Create the FacetGrid for the small multiples
    g = sns.FacetGrid(
        table,
        col="Exam",
        col_wrap=3,
        height=2,
        aspect=1.5)

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
    g.fig.suptitle('Text Similarity by Temperature by Prompt and Exam', size=11)
    g.fig.subplots_adjust(top=0.92)
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Temperature", metric)
    # g.add_legend(title="Prompt", loc="upper right", bbox_to_anchor=(1, 1))
    g.savefig(f"{output_folder}/{metric_file_name}-by-temperature-and-exam.png")
    plt.show()

    # Create a stand-alone legend for the chart
    # NOTE: Because adding a legend above is creating excess whitespace
    line_plot = sns.lineplot(
        data=similarity_by_temperature,
        x="Temperature",
        y=metric,
        hue="Prompt",
        style="Prompt",
        dashes=False)
    handles, labels = line_plot.get_legend_handles_labels()
    plt.close()
    legend_fig = plt.figure(figsize=(2.0, 1.5))
    legend_fig.legend(handles, labels, loc='center', title="Prompt")
    plt.axis('off')
    legend_fig.savefig(f"{output_folder}/text-similarity-legend.png", bbox_inches='tight')
    plt.show()

    # Add the legend to the chart image
    plot_image = Image.open(
        f"{output_folder}/{metric_file_name}-by-temperature-and-exam.png")
    legend_image = Image.open(f"{output_folder}/text-similarity-legend.png")
    new_image = Image.new("RGB", (plot_image.width, plot_image.height))
    new_image.paste(plot_image)
    legend_x = plot_image.width - legend_image.width - 20
    legend_y = plot_image.height - legend_image.height - 20
    new_image.paste(legend_image, (legend_x, legend_y))
    new_image.save(
        f"{output_folder}/{metric_file_name}-by-temperature-and-exam.png")
    plt.imshow(new_image)
    plt.show()
    plot_image.close()
    legend_image.close()
    new_image.close()

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
plt.figure(figsize=(10, 5))
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
plt.legend(title="Metric")
plt.title("Text Similarity by Temperature and Metric")
plt.xlabel("Temperature")
plt.ylabel("Similarity")
plt.tight_layout()
plt.savefig(f"{output_folder}/similarity-by-temperature-and-metric.png")
plt.show()
