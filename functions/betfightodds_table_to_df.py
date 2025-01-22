# This function takes an html table from BetFightOdds.com and turns it into a properly formatted pandas data frame

def betfightodds_table_to_df(table, fighter):
    
    # pdb.set_trace()
    
    # Extract headers and rows in one go using list comprehensions
    # First remove all tds
    # Find all <td> elements and remove them
    decomposed_table = copy.deepcopy(table)
    
    for td in decomposed_table.find_all('td'):
        td.decompose()  # This removes the <td> element from the tree

    headers = [header.text for header in decomposed_table.find_all('th')]
    rows = [[cell.text for cell in row.find_all('td')] for row in table.find_all('tr')[1:]]

    # The rows contain 3 vectors, one with the event, the next with the fighter's odds, the final with the opponents odds and the date
    # The event names on the website are not always correct, so only provide Bout and date (date not always correct either. Sometimes off by 1 day!)

    # Remove lists of length 1, as these are just titles
    filtered_list = [lst for lst in rows if len(lst) > 1]
    # Now split the data, as its odds for first fighter then second fighter
    even_list = [lst for idx, lst in enumerate(filtered_list) if idx % 2 == 0]
    odd_list = [lst for idx, lst in enumerate(filtered_list) if idx % 2 != 0]

    # Create new column headers
    custom_headers = ['open_odds', 'closing_odds_low', 'delete1', 'closing_odds_high', 'delete2', 'delete3', 'event']

    # Convert directly to DataFrame
    df = pd.DataFrame(even_list, columns = custom_headers)

    # Remove columns not needed
    df.drop(['delete1', 'delete2', 'delete3'], axis = 1, inplace = True)
    
    # Finally, Add on the dates and opponent
    dates = pd.DataFrame(odd_list)[4]
    df['date'] = betfightodds_dates_to_datetime(dates)

    # Make sure the cases of the letters in the fighters name in the headers matches what is expected from the UFC data
    headers = [fighter if fighter.lower() == item.lower() else item for item in headers]

    df['fighter'] = fighter
    df['opponent'] = [item for item in headers if clean_name_for_odds_search(item) not in ['matchup', 'open', 'closingrange', 'movement', '', 'event', clean_name_for_odds_search(fighter), clean_name_for_odds_search(max(set(headers), key=headers.count))]] # exclude the most common element as it is usually their name

    return df
