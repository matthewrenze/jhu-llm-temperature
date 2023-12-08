# Import the packages
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.patches as patches

# Set the parameters
model_type = "gpt-3.5"
prompt_type = None
exam_type = None
input_file = f"../details/all-details.csv"
output_folder = f"../plots"

# Create the output folder
os.makedirs(output_folder, exist_ok=True)

# Load the data
details = pd.read_csv(input_file)

# Filter by model and agent
details = details[details["Model"] == model_type]

# Filter out the comprehensive-100 exam
details = details[details["Exam"] != "comprehensive-100"]

# Filter the temperatures from 0.0 to 1.0
details = details[details["Temperature"] <= 1.0]

# Sort the data by agent
details["Prompt"] = pd.Categorical(details["Prompt"], ["Baseline", "Domain-Expert", "Self-Recitation", "Chain-of-Thought", "Composite"])

# Group by prompt and temperature and average the accuracy
results = details \
    .groupby(["Prompt", "Exam", "Temperature"]) \
    .agg({"Accuracy": "mean"}) \
    .reset_index()

# Create the FacetGrid for the small multiples
g = sns.FacetGrid(
    results,
    col="Exam",
    col_wrap=3,
    height=2,
    aspect=1.5)

# Define the plotting function for the FacetGrid
def plot_lineplot(*args, **kwargs):
    data = kwargs.pop("data")
    sns.lineplot(
        x="Temperature",
        y="Accuracy",
        hue="Prompt",
        data=data,
        marker="o",
        errorbar=None)
    plt.ylim(0, 1)
    plt.legend(title="Agent")
    # Only add legend to the last subplot
    if kwargs.get('label') == 'last_label':
        ax.legend(
            title="Prompt",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3)

# Plot the accuracy by temperature and prompt
g.map_dataframe(plot_lineplot)
g.fig.suptitle('Accuracy of GPT-3.5 by Temperature, Prompt, and Exam', size=11)
g.fig.subplots_adjust(top=0.92)
g.set_titles(col_template="{col_name}")
g.set_axis_labels("Temperature", "Accuracy")
# g.add_legend(
#     title="Prompt",
#     loc="lower right",
#     bbox_to_anchor=(0.80, 0.05),
#     frameon=True)
g.savefig(f"{output_folder}/accuracy-by-temperature-prompt-and-exam.png")
plt.show()

# Create a stand-alone legend for the chart
# NOTE: Because adding a legend above is creating excess whitespace
line_plot = sns.lineplot(
    data=details,
    x="Temperature",
    y="Accuracy",
    hue="Prompt",
    style="Prompt",
    marker="o",
    dashes=False)
handles, labels = line_plot.get_legend_handles_labels()
plt.close()
legend_fig = plt.figure(figsize=(2.0, 1.5))
legend_fig.legend(handles, labels, loc='center', title="Prompt")
plt.axis('off')
legend_fig.savefig(f"{output_folder}/accuracy-by-temperature-prompt-and-exam_legend.png", bbox_inches='tight')
plt.show()

# Add the legend to the chart image
plot_image = Image.open(f"{output_folder}/accuracy-by-temperature-prompt-and-exam.png")
legend_image = Image.open(f"{output_folder}/accuracy-by-temperature-prompt-and-exam_legend.png")
new_image = Image.new("RGB", (plot_image.width, plot_image.height))
new_image.paste(plot_image)
legend_x = plot_image.width - legend_image.width - 20
legend_y = plot_image.height - legend_image.height - 20
new_image.paste(legend_image, (legend_x, legend_y))
new_image.save(f"{output_folder}/accuracy-by-temperature-prompt-and-exam.png")
# plt.imshow(new_image)
# plt.show()
plot_image.close()
legend_image.close()
new_image.close()

