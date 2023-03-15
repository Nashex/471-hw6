import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

columns = ["model_name", "dev_accurary", "test_accurary"]
df = pd.read_csv("model_accuracies.csv", names=columns)

# Round the accuracies to 2 decimal places
df["dev_accurary"] = df["dev_accurary"].apply(lambda x: round(float(x), 2))
df["test_accurary"] = df["test_accurary"].apply(lambda x: round(float(x), 2))

# Make a grouped bar plot of the accuracies
fig, ax  = plt.subplots(figsize=(10, 8))
width = 0.4
x = np.arange(len(df["model_name"]))
dev = ax.bar(x - width / 2, df["dev_accurary"], width=0.4, label="Dev")
test = ax.bar(x + width / 2, df["test_accurary"], width=0.4, label="Test")

ax.set_xticks(x)
ax.set_xticklabels(df["model_name"])

# Add the y-axis label
ax.set_ylabel("Accuracy")

# Add the legend
ax.legend()

# Add the title
ax.set_title("Model Accuracies")

# Save the plot
fig.savefig("figures/model_accuracies.png")
