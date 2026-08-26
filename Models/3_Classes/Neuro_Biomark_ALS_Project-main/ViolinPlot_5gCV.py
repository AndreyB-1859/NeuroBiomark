import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from config.config import config

# Prepare storage for results
records = []

# Loop through all fold CSV files
for file in os.listdir(config.logs_dir_path):
    if file.startswith("fold_") and file.endswith("_evaluation_results.csv"):
        fold_path = os.path.join(config.logs_dir_path, file)
        fold_num = file.split("_")[1]  # extract fold number from filename
        
        df = pd.read_csv(fold_path)
        df = df[df['Class'] != 'Overall']  # skip the 'Overall' row
        
        for _, row in df.iterrows():
            records.append({
                'Fold': int(fold_num),
                'Class': str(row['Class']),
                'Sensitivity': float(row['Sensitivity']),
                'Specificity': float(row['Specificity'])
            })

# Convert to DataFrame
data = pd.DataFrame(records)

# Optional: Replace numeric class labels with readable names
class_map = {
    '0': 'Control',
    '1': 'ALS-Concordant',
    '2': 'ALS-Discordant'
}
data['Class'] = data['Class'].map(class_map)

# Melt the data for easy plotting (long format)
melted = data.melt(id_vars=['Fold', 'Class'], value_vars=['Sensitivity', 'Specificity'],
                   var_name='Metric', value_name='Score')

# Set up the figure
plt.figure(figsize=(10, 6))
sns.violinplot(x='Class', y='Score', hue='Metric', data=melted, inner='box', cut=0)
plt.title("Distribution of Sensitivity and Specificity Across Folds")
plt.ylabel("Score")
plt.xlabel("Class")
plt.legend(title='Metric')

# Save the figure
save_path = os.path.join(config.logs_dir_path, "violin_plot_metrics.png")
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

print(f"Violin plot saved to: {save_path}")
