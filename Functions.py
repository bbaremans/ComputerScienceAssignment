
import re
import numpy as np
import pandas as pd
from nltk import ngrams


def split_raw_data(data_json):
    key_occurrences = {}
    all_products = []
    unique_brand_list = ["insignia", "dynex", "tcl", "elite", "viore", "gpx", "contex", "curtisyoung", "mitsubishi",
                         "hiteker", "avue", "optoma"]
    for items in data_json.keys():

        for single_item in data_json[items]:
            all_products.append(single_item)
            featuremaps = single_item["featuresMap"]
            featuremaps_brand = featuremaps.get("Brand")

            for key in featuremaps:
                key_occurrences[key] = key_occurrences.get(key, 0) + 1

            if featuremaps_brand is not None:
                featuremaps_brand = featuremaps_brand.lower()
                unique_brand_list.append(featuremaps_brand)
                unique_brand_list = list(set(unique_brand_list))
            else:
                single_item["title"] += " featuresmap_has_no_brand_value"

    del key_occurrences['Brand']
    most_occurring_featuremaps_key = sorted(key_occurrences, key=lambda k: key_occurrences[k], reverse=True)

    # Display the key occurrences
    for key in most_occurring_featuremaps_key:
        print(f"The key '{key}' occurs {key_occurrences[key]} times.")

    return all_products, unique_brand_list, most_occurring_featuremaps_key


def create_equalized_dataframe(all_products, unique_brand_list, most_occurring_featuremaps_key, NUMBER_OF_KEYS_TO_USE):
    selection_most_used_keys = most_occurring_featuremaps_key[:NUMBER_OF_KEYS_TO_USE]

    unique_brand_list = sorted(unique_brand_list, key=len, reverse=True)
    equalized_df = pd.DataFrame(columns=[])
    for product in all_products:
        equalized_title, brand = create_equalized_row(product.get("title"), unique_brand_list)

        featuremaps = product["featuresMap"]
        featuremaps_brand = featuremaps.get("Brand")

        if brand is None:
            if featuremaps_brand is not None:
                brand = featuremaps_brand.lower()

        feature_information = {}
        featuremaps = product["featuresMap"]
        for key in selection_most_used_keys:
            feature_item = featuremaps.get(key)
            feature_information.update({key: feature_item})

        # equalized_feature_information = equalize_feature_information(feature_information)
        webshop = product["shop"]

        product_information = {**{'ModelID': product.get("modelID"), 'Title': equalized_title, 'Brand': brand,
                                  "FeaturesMaps": featuremaps_brand, "WebShop": webshop}, **feature_information}
        new_row = pd.DataFrame(product_information, index=[0])
        equalized_df = pd.concat([equalized_df, new_row], ignore_index=True)

    # Replace "Pansonic" with "Panasonic" and "lg" with "lg electronics" regardless of case and
    # delete FeatureMaps since it is incomplete
    equalized_df['Brand'] = equalized_df['Brand'].str.replace('pansonic', 'panasonic', case=False)
    equalized_df['Brand'] = equalized_df['Brand'].str.replace('lg', 'lg electronics', case=False)
    equalized_df = equalized_df.drop('FeaturesMaps', axis=1)
    # selection_most_used_keys.remove('Component Video Inputs')
    selection_most_used_keys.remove('Aspect Ratio')
    selection_most_used_keys.append('WebShop')
    selection_most_used_keys.append('Brand')

    return equalized_df, selection_most_used_keys


def create_equalized_row(string, unique_brand_list):
    value = string.lower()
    brand = None

    if value.find(" featuresmap_has_no_brand_value") != -1:
        for potential_brand in unique_brand_list:
            if value.find(potential_brand) != -1:
                brand = potential_brand

    value = value.replace(" featuresmap_has_no_brand_value", "")

    inches = ["'", '"', "inches", " inches", " inch", "-inch", '”', " '", ' "', ' ”', "-inch"]
    hertzes = ["hertz", " hz", "-hz", " - hz"]
    websites = ["amazon.com", "amazon", "newegg.com", "newegg", "best-buy.com", "best-buy", "best buy",
                "thenerds.net", "thenerds", "the nerds"]

    value = re.sub(r'(\b(?:[0-9]*)[a-z]+\b)|(\b[a-z]+\b)', "", value)

    # Change inch format
    for inch in inches:
        value = value.replace(inch, "inch")

    # Change hertz format
    for hertz in hertzes:
        value = value.replace(hertz, "hz")

    # Erase website
    for website in websites:
        value = value.replace(website, "")

    value = value.replace(" .", "")
    value = value.replace(". ", "")
    value = value.replace('  ', ' ')
    value = re.sub("[^a-zA-Z0-9\s\.]", "", value)

    for character in value:
        if value[0] == " ":
            value = value[1:]
        else:
            break

    return value, brand


def equalize_feature_information(feature_information):
    equalized_feature_information = {}

    for item in feature_information:
        feature_item = feature_information.get(item)

        if feature_item is None:
            continue

        if isinstance(feature_item, str):
            feature_item = feature_item.lower()
        else:
            feature_item = feature_item.str.lower()

        if item == "Maximum Resolution":
            feature_item = feature_item.replace(" ", "")
            feature_item = feature_item.replace("x", "")
            if feature_item == "1,024768(native)":
                # strange value to "" to avoid error later
                feature_item = ""

        elif item == "Aspect Ratio":
            feature_item = feature_item.replace(":", "")
            feature_item = feature_item.replace(",", "")
            feature_item = feature_item.replace("and", "")
            feature_item = feature_item.replace(" ", "")
            feature_item = feature_item.replace("and", "")
            feature_item = feature_item.replace("|", "")

        elif item == "V-Chip" or item == "USB Port":
            if feature_item.find("yes") != -1:
                feature_item = "yes"
            elif feature_item.find("no") != -1:
                feature_item = "no"
            elif feature_item.find("0") != -1:
                feature_item = "no"
            else:
                feature_item = "no"

        elif item == "Screen Size (Measured Diagonally)" or item == "Screen Size Class" or item == "Vertical Resolution":
            # No commas are in this col.
            feature_item = re.compile(r'[^0-9]').sub('', feature_item)

        elif item == "TV Type":
            feature_item = re.compile(r'[^a-zA-Z0-9\s]').sub('', feature_item)

        feature_item = feature_item.replace(" ", "")
        feature_information.update({item: feature_item})

    return equalized_feature_information


def create_equalized_data(title_product, unique_brand_list):
    title_product = title_product.lower()
    brand = None

    # Find brand in title only when not found in featuresMap dict
    if title_product.find(" featuresmap_has_no_brand_value") != -1:
        for potential_brand in unique_brand_list:
            if title_product.find(potential_brand) != -1:
                brand = potential_brand

    inches = ["'", '"', "inches", " inches", " inch", "-inch", '”', " '", ' "', ' ”', "-inch"]
    hertzes = ["hertz", " hz", "-hz", " - hz"]

    # Change inch format
    for inch in inches:
        title_product = title_product.replace(inch, "inch")

    # Change hertz format
    for hertz in hertzes:
        title_product = title_product.replace(hertz, "hz")

    regex = re.compile(
        r'(?:^|(?<=[ \[\(]))([a-zA-Z0-9]*(?:(?:[0-9]+[^0-9\., ()]+)|(?:[^0-9\., ()]+[0-9]+)|(?:([0-9]+\.[0-9]+)[^0-9\., ()]+))[a-zA-Z0-9]*)(?:$|(?=[ \)\]]))')
    modelword = [x for sublist in regex.findall(title_product) for x in sublist if x != ""]
    modelword = ' '.join(modelword)

    return modelword, brand


def real_duplicates(vector_modelID):
    number_of_duplicates = 0

    for i in range(len(vector_modelID)):
        for j in range(i + 1, len(vector_modelID)-1):
            if i == j:
                continue
            if vector_modelID[i] == vector_modelID[j]:
                number_of_duplicates += 1

    return number_of_duplicates


def create_bin_vector(equalized_data):
    model_words = set()

    for title in equalized_data["Title"]:
        model_words.update(title.split())

    model_words_list = sorted(list(model_words))

    binary_matrix = pd.DataFrame(0, index=model_words_list, columns=equalized_data.index)

    for index, product in equalized_data.iterrows():
        for mw in model_words_list:
            if mw in product['Title'].split():
                binary_matrix.at[mw, index] = 1

    binary_matrix.columns = binary_matrix.columns + 1

    return binary_matrix


# needed in start_min_hashing
def hash_functions(a, b, x):
    # based on the input number x, make sure to use the corresponding prime number for the hash functions
    # as if not a prime number is used, the division into the buckets will not be arbitrarily over all the buckets
    prime = x + 2
    for nr in range(2, prime):
        if prime % nr == 0:  # if the number is divisible by nr, it is not prime so update prime by 1
            prime = prime + 1
            nr = 2

    # define the hash functions
    hash = (a * x + b) % prime  # former part defined as y = ax + b (linear function)

    return hash


# needed in start_min_hashing
def hash_ab(N):
    # create lists for possible values for a and b for the hash values
    a_values = []
    b_values = []
    for individual in range(N):
        a_values.append(np.random.randint(1, N))
        b_values.append(np.random.randint(1, N))

    return a_values, b_values


def start_min_hashing(binary_matrix, N):
    rows, columns = np.shape(binary_matrix)

    # hash value for each row and each hash function
    hash_values = np.zeros((N, columns), dtype=int)

    for column in range(columns):
        # load the values of the variables needed for the hash functions; the lists for a and b
        a_values, b_values = hash_ab(N)
        for index in range(len(a_values)):
            hash_outcome = hash_functions(a_values[index], b_values[index], N + 1)
            # Save the hash values
            hash_values[index, column] = hash_outcome

    return hash_values


def signature_matrix(binary_matrix, hash_values, N):
    rows, columns = binary_matrix.shape
    signature_matrix = pd.DataFrame(np.full((N, columns), np.inf),
                                    index=range(N))  # create an initial signature matrix with infinity as elements
    rows_sig, columns_sig = np.shape(signature_matrix)
    # update the signature matrix subsequently following the iterative approach given in the slides
    for i in range(rows):
        for j in range(columns_sig):
            # Find the row containing the first 1
            if binary_matrix.iloc[i, j] == 1:

                for value in range(len(hash_values)):
                    # Check if the hash value is smaller than the value stored in the M matrix,
                    # if true update the M matrix with the smaller number
                    if hash_values[value, i] < signature_matrix.loc[value, j]:
                        signature_matrix.loc[value, j] = hash_values[value, i]
    return signature_matrix


def bands_rows(signature_matrix):
    # create lists for possible number of bands and rows
    b_r_values = []
    N = len(signature_matrix)

    for b in range(1, N + 1):
        r = N // b
        if b * r == N:
            b_r_values.append((b, r))

    return b_r_values


def threshold(b_r_values, threshold):
    t_values = []
    for b,r in b_r_values:
        t = (1 / b) ** (1 / r)          # threshold approximated by this equation
        t_values.append(t)

    # find the b and r that belong to the t value that is closest to the 0.5 threshold set ourselves
    closest_t = min(t_values, key=lambda x: abs(x - threshold))
    optimal_br = b_r_values[t_values.index(closest_t)]

    return optimal_br


def start_lsh(sig_matrix, optimal_br):
    row_sig, col_sig = np.shape(sig_matrix)
    bands, rows = optimal_br

    # find all the hash values per band and make a string to capture these hash values
    band_hash_values = []

    # iterate over the bands
    for i in range(0, bands, rows):
        hash_vals_band_i = []
        end = i + rows
        ith_band = sig_matrix.iloc[i: end, :]
        for col in range(col_sig):
            # convert the hash values to a hash code for each column in a band
            col_values = ith_band.iloc[:, col]
            hash_string = ''.join(map(str, col_values))
            hash_vals_band_i.append(hash_string)
        # store the hash codes of a band in a list which contains this info of all bands
        band_hash_values.append(hash_vals_band_i)

    # now for each band, put columns with the exact same hash code into the same bucket, whereas the others who do not
    # have a hash code buddy get their own bucket
    buckets_per_band = []

    # iterate over all bands
    for band in band_hash_values:
        # create a list to store the buckets per band
        pairs_band = []
        bucket_hash_rows, bucket_hash_cols = np.shape(band_hash_values)
        # iterate over the hash codes of each band
        for hash_code in range(bucket_hash_cols):
            # check whether current hash code has already been assigned to a bucket, i.e. when hash code is 'done'
            # if not, make a new list/bucket for this hash code
            if band[hash_code] == 'done':
                same_hash = []
            elif band[hash_code] != 'done':
                same_hash = [hash_code]
                # now check for all the other columns/hash codes in this band whether they are the same as the current
                # and store them in the same bucket if they are the same
                for hash_compare in range(hash_code + 1, bucket_hash_cols):
                    if band[hash_code] == band[hash_compare]:
                        same_hash.append(hash_compare)
                        band[hash_compare] = band[hash_compare].replace(band[hash_compare], 'done')
            # lastly, store the list of columns with same hash codes
            pairs_band.append(same_hash)
        buckets_per_band.append(pairs_band)

    # find buckets with two or more elements in order to determine which products are found to be possible duplicates
    processed_pairs = []
    candidate_pairs = pd.DataFrame(0, index=range(col_sig), columns=range(col_sig))

    for bucket in buckets_per_band:
        for buck in bucket:
            if len(buck) >= 2:

                # immediately update the candidate pair matrix with a 1
                for i in range(len(buck)):
                    for j in range(i + 1, len(buck)):
                        pair = (buck[i], buck[j])
                        if pair not in processed_pairs:
                            candidate_pairs.at[buck[i], buck[j]] = 1

                            processed_pairs.append(pair)

    candidate_pairs.columns = candidate_pairs.columns + 1
    new_index = candidate_pairs.index + 1
    candidate_pairs = candidate_pairs.rename(index=dict(zip(candidate_pairs.index, new_index)))

    return candidate_pairs


def find_coordinates(signature_matrix):
    coordinates = []

    for i in range(len(signature_matrix)):
        for j in range(len(signature_matrix.columns)):
            if signature_matrix.iloc[i, j] == 1:
                coordinates.append((i, j))

    return coordinates


def separate_coordinates_lists(binary_matrix):
    coordinates = find_coordinates(binary_matrix)

    first_coordinates, second_coordinates = zip(*coordinates)

    return list(first_coordinates), list(second_coordinates)


def jaccard_similarity(str1, str2, n=3):
    set1 = set(ngrams(str1, n))
    set2 = set(ngrams(str2, n))
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    if union == 0:
        return 1
    else:
        return intersection / union


def similarity_matrices(strings, rows, columns):
    jaccard = pd.DataFrame()
    for first_coord, second_coord in zip(rows, columns):
        value = jaccard_similarity(strings[first_coord], strings[second_coord])
        row = {'first_coord': first_coord,
            'second_coord': second_coord,
            'calculated_value': value}
        jaccard = jaccard._append(row, ignore_index=True)

    dissimilarity = jaccard
    dissimilarity['calculated_value'] = 1 - jaccard['calculated_value']

    return jaccard, dissimilarity


def compare_kpv(candidate_pairs, jaccard, dissimilarity, equalized_df, selection_most_used_keys):
    for row in range(len(jaccard)):
        first_coord_info = equalized_df.iloc[jaccard['first_coord'].astype(int)]
        second_coord_info = equalized_df.iloc[jaccard['second_coord'].astype(int)]
        for key in selection_most_used_keys:
            if first_coord_info.iloc[row][key] is None or second_coord_info.iloc[row][key] is None:
                continue

            if key == 'WebShop':
                if first_coord_info.iloc[row][key] == second_coord_info.iloc[row][key]:
                    dissimilarity.at[row, 'calculated_value'] = np.inf
                    break

            # our extension: we consider products with unequal features also having distance infinity
            elif first_coord_info.iloc[row][key] != second_coord_info.iloc[row][key]:
                dissimilarity.at[row, 'calculated_value'] = np.inf
                break

    pairs_msm = candidate_pairs
    for row in range(len(dissimilarity)):
        if dissimilarity.iloc[row, dissimilarity.columns.get_loc('calculated_value')] == np.inf:
            first_coord = int(jaccard.iloc[row, jaccard.columns.get_loc('first_coord')])
            second_coord = int(jaccard.iloc[row, jaccard.columns.get_loc('second_coord')])
            pairs_msm.loc[first_coord + 1, second_coord + 1] = 0
            pairs_msm.loc[second_coord + 1, first_coord + 1] = 0

    return pairs_msm, dissimilarity


def get_f1_score(candidate_pairs, equalized_df, dissimilarity):
    comparisons_made = (candidate_pairs.sum().sum())
    real_duplicates_number = real_duplicates(equalized_df)
    duplicates_found = 0

    # Filter rows containing inf values out
    dissimilarity = dissimilarity.replace([np.inf, -np.inf], np.nan)
    dissimilarity = dissimilarity.dropna(axis=0, how = 'any')

    for row in range(len(dissimilarity)):
        first_coord = dissimilarity.iloc[row]['first_coord']
        second_coord = dissimilarity.iloc[row]['second_coord']

        if equalized_df.loc[first_coord] == equalized_df.loc[second_coord]:
            duplicates_found += 1

    pq = duplicates_found / comparisons_made
    pc = duplicates_found / real_duplicates_number
    f1_score = (2 * pq * pc) / (pq + pc)

    return pq, pc, f1_score

def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
