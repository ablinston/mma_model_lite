# This function gives a true or false if one string is fuzzily contained in another

def fuzzy_contains(substring, text, threshold=80):
    return fuzz.partial_ratio(substring.lower(), text.lower()) >= threshold