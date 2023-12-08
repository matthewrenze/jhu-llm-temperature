import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm

# Load the data
details = pd.read_csv(f"../details/all-details.csv")

# Filter in only GPT-3.5 or GPT-4 model
details = details[details["Model"] == "gpt-4"]
# details = details[details["Model"] == "gpt-3.5"]

# Filter in only the composite prompt
details = details[details["Prompt"] == "Composite"]

# Filter in only the comprehensive-100 exam
details = details[details["Exam"] == "comprehensive-100"]

# Filter in only the temperatures from 0.0 to 1.0
details = details[details["Temperature"] <= 1.0]

# Rename the "Problem ID" column
# details = details.rename(columns={"Problem ID": "Problem_ID"})

# Create a subject ID column
details['Subject_ID'] = details['Model'] + "-" + details['Prompt'] + "-" + details['Exam'] + "-" + details['Problem ID'].astype(str)

# Run the repeated measures ANOVA
rm_anova = AnovaRM(
    data=details,
    depvar='Accuracy',
    subject='Subject_ID',
    within=['Temperature'])

# Fit the model
results = rm_anova.fit()

# Print the ANOVA results
print("\nRepeated Measures ANOVA Results:")
print(results)
