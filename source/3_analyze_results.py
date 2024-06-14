# Import the packages
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import shared

# Hide future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the results
results = pd.read_csv(f"../data/results/results.csv")

# Filter the results
results = results[results["Model Name"] == "llama-2-70b-chat"]
results = results[results["Agent Name"] == "chain_of_thought"]
results = results[results["Exam Name"] == "comprehensive-100"]
results = results[results["Temperature"] <= 1.0]

# Process the results
results = shared.set_model_titles(results)
results = shared.set_agent_titles(results)
results = shared.sort_by_agent(results)