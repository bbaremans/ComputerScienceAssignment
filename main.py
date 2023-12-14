import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from Functions import create_equalized_dataframe, compare_kpv, get_f1_score, separate_coordinates_lists, format_time, \
    similarity_matrices, bands_rows, threshold, real_duplicates, create_bin_vector, start_min_hashing, \
    signature_matrix, start_lsh, split_raw_data
import time

# Record the start time
start_time = time.time()

# set a seed such that the draws for a and b are the same each time the code is run
seed_value = 42
np.random.seed(seed_value)

# Absolute path of the current directory
current_directory = os.getcwd()

# Absolute path of the target file or directory
json_path = '\\TVs-all-merged.json'

# Get the relative path
path = Path(current_directory+json_path)

with open(path, 'r') as file:
    # Load the JSON data
    data_json = json.load(file)

NUMBER_OF_KEYS_TO_USE = 5
all_products, unique_brand_list, most_occurring_featureMaps_key = split_raw_data(data_json)

equalized_df, selection_most_used_keys = create_equalized_dataframe(all_products, unique_brand_list,
                                                                    most_occurring_featureMaps_key,
                                                                    NUMBER_OF_KEYS_TO_USE)

NUMBER_OF_BOOTSTRAPS = 5

thresholds = np.arange(0.05, 1.05, 0.05)
thresholds = np.round(thresholds, 2)
row_names = ["fraction_comparison", "pq_lsh", "pc_lsh", "f1_star", "pq", "pc", "f1"]

result_all_bootstraps = pd.DataFrame(data=0, index=row_names, columns=thresholds.astype(str))

for _ in range(NUMBER_OF_BOOTSTRAPS):
    equalized_df = equalized_df.sample(frac=0.60).reset_index(drop=True)
    modelIDs = equalized_df.iloc[:, 0]
    titles = equalized_df.iloc[:, 1]

    real_dup = real_duplicates(modelIDs)

    binary_vector = create_bin_vector(equalized_df)

    hash_values = start_min_hashing(binary_vector, round(len(binary_vector) * 0.5))

    sig_matrix = signature_matrix(binary_vector, hash_values, round(len(binary_vector) * 0.5))

    band_row_pairs = bands_rows(sig_matrix)
    print(band_row_pairs)

    result_per_bootstrap = pd.DataFrame(data=0, index=row_names, columns=thresholds.astype(str))

    for threshold_value in thresholds:
        optimal_band_row = threshold(band_row_pairs, threshold_value)
        print(optimal_band_row)

        candidate_pairs = start_lsh(sig_matrix, optimal_band_row)
        total_comparisons = (len(candidate_pairs) * (len(candidate_pairs) - 1)) / 2
        candidates_lsh = candidate_pairs.sum().sum()
        fraction_comp = candidates_lsh / total_comparisons
        result_per_bootstrap.at["fraction_comparison", str(threshold_value)] = fraction_comp
        print(fraction_comp)

        rows, columns = separate_coordinates_lists(candidate_pairs)

        jaccard_lsh, dissim_lsh = similarity_matrices(titles, rows, columns)

        pq_lsh, pc_lsh, f1_star = get_f1_score(candidate_pairs, modelIDs, dissim_lsh)
        result_per_bootstrap.at["pq_lsh", str(threshold_value)] = pq_lsh
        result_per_bootstrap.at["pc_lsh", str(threshold_value)] = pc_lsh
        result_per_bootstrap.at["f1_star", str(threshold_value)] = f1_star

        pairs_msm, dissimilarity_msm = compare_kpv(candidate_pairs, jaccard_lsh, dissim_lsh, equalized_df,
                                                                                            selection_most_used_keys)
        pq, pc, f1_score = get_f1_score(pairs_msm, equalized_df.iloc[:, 0], dissimilarity_msm)
        result_per_bootstrap.at["pq", str(threshold_value)] = pq
        result_per_bootstrap.at["pc", str(threshold_value)] = pc
        result_per_bootstrap.at["f1", str(threshold_value)] = f1_score

        print("Threshold", threshold_value, "is now completed.")
    result_all_bootstraps += result_per_bootstrap
    print("Bootstrap", _, "is now completed.")
    print(result_per_bootstrap)

result_all_bootstraps = result_all_bootstraps / NUMBER_OF_BOOTSTRAPS
result_all_bootstraps.to_csv('CSOutput.csv', index=True)

# Record the end time
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
# Print the elapsed time in a readable format
print(f"Elapsed time: {format_time(elapsed_time)}")
