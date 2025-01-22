# This function takes a string and cleans it ready for the odds search

def clean_name_for_odds_search(name_string):

    return name_string.lower().replace('.', '').replace(' ', '').replace("'", '').replace('-', '').replace('saint', 'st')
