# This function uses rapid fuzz to get a fuzzy match

def get_best_match(x, choices, threshold=70):
    match = process.extractOne(x, choices, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= threshold:
        return match[0]
    else:
        return None