import os
import pandas as pd
from causalimpact import CausalImpact
import matplotlib.pyplot as plt

# Create output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created: {output_dir}")

# Read data
cigar = pd.read_csv(
    "data/cigar.txt",
    header=None,
    sep="\s+",
    names=["state", "year", "price", "pop", "pop16", "cpi", "ndi", "sales", "pimin"],
).dropna()

# List for states to skip
skip_state = [3, 9, 10, 22, 21, 23, 31, 33, 48]
cigar = cigar[(~cigar["state"].isin(skip_state)) & (cigar["year"] >= 70)].reset_index(drop=True)
cigar["area"] = cigar["state"].apply(lambda x: "CA" if x == 5 else "Other states")

# Create a new column for the year
y = cigar[cigar.state == 5][["year", "sales"]].set_index("year")
y.columns = ["y"]

# Create a pivot table for the sales data
X = pd.pivot_table(
    cigar[cigar.state != 5][["year", "state", "sales"]], values="sales", index="year", columns="state"
).add_prefix("X_")

# Set the pre and post period
pre_period = [0, 15]
post_period = [16, 22]

df_final = pd.concat([y, X], axis=1)

# Filtering training data
# df_training = df_final.loc[df_final.index <= pre_period[1] + 70].dropna()

# Calculate correlation coefficient
# threshold = 0.3
# correlation = df_training.corr()["y"]

# Distinct columns over threshold
# columns = correlation[correlation.abs() > threshold].index
# df_final = df_final.drop(columns=["X_30"])

# Drop columns whose correlation between 'y' is less than threshold
# df_final = df_final[columns]

df_final = df_final.reset_index(drop=True)

# Create causalimpact model and fit the model
ci = CausalImpact(df_final, pre_period, post_period, model_args={"fit_method": "hmc"})

plt.figure()

print(ci.summary(output="report"))
ci.plot(show=False)

# Change xticks' labels
plt.xticks(ticks=[0, 5, 10, 15, 20], labels=[70, 75, 80, 85, 90])

plt.savefig(os.path.join(output_dir, "causal_impact_plot_3.png"))
plt.savefig(os.path.join(output_dir, "causal_impact_plot_3.pdf"))
