# Scalable product duplicate detection using Feature-Based Classification
Over the years, online web-shops like Amazon, Wish, Bol.com and nowadays Temu have been expanding with an uncontrollable speed. As these platforms keep adding new products to their product range the probability to face the exact same product on different web-shops increases. For consumers it could be valuable to know whether they come across the exact same product on different web shops. Therefore this paper investigates how duplicates can be accurately found across web shops. This paper applies preselection in terms of Locality-Sensitive Hashing (LSH) to reduce the number of comparisons made and thus a reduction in computation time. Instead of using a clustering algorithm, which is extensively used in previous work on this topic, we classify possible duplicates to be real duplicates based on the same feature information such as brand, USB port and TV type. We call this novel approach Feature-Based Classification (FBC).

### Code description
Now a brief but clear discussion on the code will be provided. The code consists of the main code, included in 'main.py', from which we observe the results and store them in a csv file. The functions used in this file are imported from 'Function.py'. Lastly, the 'plotting.py' takes the results that were stored in a csv file and converts it to useful and clear plots that are used in the final paper. Concluding, one can first run 'main.py', followed by 'plotting.py' in order to obtain the final graphs. Note that the directories and csv file names need to match. 

#### Function.py
##### -split_raw_data(data_json)
Takes the original data json file and converts it to a more useful format which can be used.

##### -create_equalized_dataframe(all_products, unique_brand_list, most_occurring_featuremaps_key, NUMBER_OF_KEYS_TO_USE)
Uses the three functions below to return an equalized dataframe with all the information.

##### -create_equalized_row(string, unique_brand_list)
Creates equalized titles; units and representations are normalized. 

##### -equalize_feature_information(feature_information)
Creates equalized feature information; units and representations are normalized.

##### -create_equalized_data(title_product, unique_brand_list)
Creates equalized titles; units and representations are normalized.

##### -real_duplicates(vector_modelID)
Counts the number of real duplicates in the dataset based on the modelID's. 

##### -create_bin_vector(equalized_data)
Creates a binary vector per product, where the row indices are the model words. A 1 is given when the model word is in the title of a product and 0 if not. After doing this for all products, we end up with a binary matrix.

##### -hash_functions(a, b, x)
For each entry of a, b and integer x, a hash value is created. 

##### -hash_ab(N)
Creates lists for a and b needed to define the N different hash functions.

##### -start_min_hashing(binary_matrix, N)
Creates and stores hash values for each hash function and each product.

##### -signature_matrix(binary_matrix, hash_values, N)
From the hash values, a signature matrix is created by going over all the hash values per product and updating it if a lower value is seen.

##### -bands_rows(signature_matrix)
Based on the length of the signature matrix (n), define the possible combinations of bands (b) and rows (r) such that n = b * r holds. 

##### -threshold(b_r_values, threshold)
Given a certain threshold, search through the combinations of bands and rows that are closest to the threshold using the approximation of the threshold: t ≅ (1/b)^(1/r).

##### -start_lsh(sig_matrix, optimal_br)
Using the optimal bands and rows found for the threshold are used for Locality-Sensitive Hashing. Gives candidate pairs after hashing products with the same hash code to the same buckets per band. 

##### -find_coordinates(signature_matrix)
Stores the coordinates of the candidate pairs, looking at the candidate pair binary matrix whether a 1 is present at a certain point in the matrix.

##### -separate_coordinates_lists(binary_matrix)
Stores the coordinates found in separate lists/vectors. 

##### -jaccard_similarity(str1, str2, n=3)
Computes the Jaccard similarity between two titles using a sliding window of 3 characters.

##### -similarity_matrices(strings, rows, columns)
Stores all the Jaccard similarities in a matrix, with corresponding coordinates, such that the pairs that are not candidate duplicates are excluded. In addition, it gives the dissimilarity back as well. 

##### -compare_kpv(candidate_pairs, jaccard, dissimilarity, equalized_df, selection_most_used_keys)
Compares the feature information of each candidate pair; if some information is missing, i.e. 'None', we continue. If not the same information is provided, we set the dissimilarity manually to infinity. At the end, we update the candidate pairs matrix if we find any dissimilarities set to infinity. 

##### -get_f1_score(candidate_pairs, equalized_df, dissimilarity)
Computes the pair completeness, pair quality and f1 score. 

##### -format_time(seconds)
Converts the seconds passed in order to complete the code to interpretable time units, such as hours and minutes. 

#### main.py
Includes keeping track of the time and running the bootstraps and different thresholds. Besides, it computes the corresponding fraction of comparison with each threshold, stores the output per threshold and averages the output over the bootstraps. The code inbetween are functions called from the 'Functions.py' file.

#### plotting.py
This file takes the csv-file created by 'main.py' and converts it to a dataframe. After that, we transpose the dataframe in order to more easily get the plots. Finally, we create 3 plots, each including either pair completeness, pair quality or f1-score for both LSH and FBC.
