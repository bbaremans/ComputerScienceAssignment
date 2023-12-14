
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt


current_directory = os.getcwd()
json_path = '\\CSOutput.csv'
path = Path(current_directory + json_path)
csv = pd.read_csv(path, sep=',|;', engine='python', index_col=0)

# Transpose the DataFrame for easy plotting
df = csv.T

# Add a new row with extrapolated values, such that the plot goes to 1 in fraction of comparisons, otherwise it would
# stop at a fraction of 0.5. To do this, we extrapolated the value at 0.5 to a slightly smaller of bigger value
new_values = {'fraction_comparison': 1.0, 'pq_lsh': 0.001, 'pc_lsh': 0.99, 'f1_star': 0.002, 'pq': 0.01, 'pc': 0.965, 'f1':0.025}
# Adding the new row at the first index
df = pd.concat([pd.DataFrame(new_values, index=[-1]), df])
# Convert the index to numeric type
df.index = pd.to_numeric(df.index, errors='coerce').fillna(0).astype(float)
# Sort the index
df = df.sort_index()

# Plotting pq_lsh and pq together against the fraction of comparisons
plt.figure(figsize=(10, 6))
plt.plot(df['fraction_comparison'], df['pq_lsh'], label='pq_lsh')
plt.plot(df['fraction_comparison'], df['pq'], label='pq')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('Pair Quality')
plt.legend()
plt.show()

# Plotting pc_lsh and pc together against the fraction of comparisons
plt.figure(figsize=(10, 6))
plt.plot(df['fraction_comparison'], df['pc_lsh'], label='pc_lsh')
plt.plot(df['fraction_comparison'], df['pc'], label='pc')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('Pair Completeness')
plt.legend()
plt.show()

# Plotting f1_star and f1 together against the fraction of comparisons
plt.figure(figsize=(10, 6))
plt.plot(df['fraction_comparison'], df['f1_star'], label='f1_star')
plt.plot(df['fraction_comparison'], df['f1'], label='f1')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('F1-measure')
plt.legend()
plt.show()
