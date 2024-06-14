import pandas as pd
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import seaborn as sns

# Load the details
details = pd.read_csv(f"../data/results/all-details.csv")

# Filter the details
details = details[details["Model Name"] == "gpt-35-turbo"]
# details = details[details["Agent Name"] == "chain_of_thought"]
details = details[details["Exam Name"] == "comprehensive-100"]
details = details[details["Temperature"] <= 1.0]

# Get errors (where error is not nan)
errors = details[details["Error"].notna()]