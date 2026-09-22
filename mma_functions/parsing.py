from ablinston_ml.scraping import get_website_soup
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import numpy as np
import pandas as pd
import re
import sys

from mma_functions import scraping, utils

def combine_fighter_stats_dfs(fighter_a_stats_df, fighter_b_stats_df, soup):
    """
    concat both fighter's stats into one df
    create new event and bout column as a key
    results in a dataframe of stats for both fighters for a fight

    arguments:
    fighter_a_stats_df (df): a df output from convert_fight_stats_to_df()
    fighter_b_stats_df (df): a df output from convert_fight_stats_to_df()
    soup (html): output of get_soup() parser

    returns
    a dataframe of stats for the fight
    """
    fight_stats = pd.concat([fighter_a_stats_df, fighter_b_stats_df])
    fight_stats['EVENT'] = soup.find('h2', class_='b-content__title'
        ).text.strip()
    fighters_names = []
    for tag in soup.find_all('a', class_='b-link b-fight-details__person-link'
        ):
        fighters_names.append(tag.text.strip())
    fight_stats['BOUT'] = ' vs. '.join(fighters_names)
    fight_stats = utils.move_columns(fight_stats, ['EVENT', 'BOUT'], 'ROUND',
        'before')
    return fight_stats

def convert_fight_stats_to_df(clean_fighter_stats, totals_column_names,
    significant_strikes_column_names):
    """
    convert a list of fighter stats from organise_fight_stats() into a structured dataframe
    check if list of stats is empty, there are old fights that do not have stats
    if fight has no stats, then fill stat columns with nans
    if fight has stats continue and get number of rounds in the fight
    for each round in fight, get stats for totals and significant strikes
    the summary of stats for the fights are ignored
    merge totals and significant stike stats together and return as one df

    arguments:
    clean_fighter_stats (list): list of fighter stats from organise_fight_stats()
    totals_column_names (list): list of column names for totals type stats
    significant_strikes_column_names (list): list of column names for significant strike type stats

    returns:
    a dataframe of fight stats
    """
    totals_df = pd.DataFrame(columns=totals_column_names)
    significant_strikes_df = pd.DataFrame(columns=
        significant_strikes_column_names)
    if len(clean_fighter_stats) == 0:
        totals_df.loc[len(totals_df)] = [np.nan] * len(list(totals_df))
        significant_strikes_df.loc[len(significant_strikes_df)] = [np.nan
            ] * len(list(significant_strikes_df))
    else:
        number_of_rounds = int((len(clean_fighter_stats) - 2) / 2)
        totals_df = pd.DataFrame(columns=totals_column_names)
        significant_strikes_df = pd.DataFrame(columns=
            significant_strikes_column_names)
        for round in range(number_of_rounds):
            totals_df.loc[len(totals_df)] = ['Round ' + str(round + 1)
                ] + clean_fighter_stats[round + 1]
            significant_strikes_df.loc[len(significant_strikes_df)] = [
                'Round ' + str(round + 1)] + clean_fighter_stats[round + 1 +
                int(len(clean_fighter_stats) / 2)]
    fighter_stats_df = totals_df.merge(significant_strikes_df, how='inner')
    return fighter_stats_df

# The 'bout' column on the fight details page of UFCstats sometimes has the fighters in the 
# reverse order. This function takes the fight stats data frame as a reference (where the order of the fighters in the bout is correct)
# and uses that to correct the bout on the fight details data frame

# It returns a fixed fight_details data frame

def fix_fight_details_bout(fight_details, fight_stats_df):

    # pdb.set_trace()
    
    # Create a copy of stats so the original isn't modified and with stripped columns
    fight_stats = fight_stats_df[['EVENT', 'BOUT']].drop_duplicates()
    
    # This function requires spaces to be removed, or they cause issues
    for df in [fight_details, fight_stats]:
        for col in ['EVENT', 'BOUT']:
            try:
                df[[col]] = utils.clean_text_columns(df[[col]])
            except:
                continue

    
    # First, get individual columns for the fighters on the details data
    fight_details[['fighter1', 'fighter2']] = fight_details['BOUT'].str.split('vs\\.', expand = True)
    
    # Now reverse the bout
    fight_details['BOUT_ALT'] = fight_details['fighter2'] + 'vs.' + fight_details['fighter1']

    ## Now find which fights are in the stats data frame under the alternate ordering on the same event
    
    # Create a data frame to merge with a flag
    fight_alt_flag_df = fight_stats.rename(columns = {'BOUT': 'BOUT_STATS'})
    fight_alt_flag_df.loc[:, 'ALT_FLAG'] = 1
    
    # Create a new fight data frame merging whether the alt bout was found
    new_fight_details = fight_details.merge(fight_alt_flag_df, how = 'left', left_on = ['EVENT', 'BOUT_ALT'], right_on = ['EVENT', 'BOUT_STATS'])
    
    # Check the new data has the same length of rows
    if len(fight_details) != len(new_fight_details):
        print("Error when fixing bouts. Inconsistent row number before and after flag merge.")
        return None
    
    # Replace bouts where required
    new_fight_details['BOUT'] = np.where(new_fight_details['ALT_FLAG'] == 1, new_fight_details['BOUT_ALT'], new_fight_details['BOUT'])
    
    # Drop all columns that were added
    new_fight_details.drop(['BOUT_ALT', 'fighter1', 'fighter2', 'ALT_FLAG', 'BOUT_STATS'], axis = 1, inplace = True)

    return new_fight_details

def organise_fight_results(results_from_soup, fight_results_column_names):
    """
    organise list of fight results
    fighters' names should be from index 1 and 2
    fight outcome should be from index 3 and 4
    other results includes from index 5 onwards
    weightclass, method, round, time, time format, referee, and details, should be 
    append all results into list and convert to a df

    arguments:
    results_from_soup (list): list of results from parse_fight_results()
    fight_results_column_names (list): list of column names for fight results

    returns:
    an organised list of fight results
    """
    fight_results_clean = []
    fight_results_clean.append(results_from_soup[0])
    fight_results_clean.append(' vs. '.join(results_from_soup[1:3]))
    fight_results_clean.append('/'.join(results_from_soup[3:5]))
    fight_results_clean.extend([re.sub('^(.+?): ?', '', text) for text in
        results_from_soup[5:]])
    fight_result_df = pd.DataFrame(columns=fight_results_column_names)
    fight_result_df.loc[len(fight_result_df)] = fight_results_clean
    return fight_result_df

def organise_fight_stats(stats_from_soup):
    """
    organise a list of raw stats extracted from soup
    each set of stats starts with the fighter's name, the function groups each set together into a list of lists by the fighter's name

    there are two different types of stats, totals and significant strikes
    Totals include KD, SIG.STR., SIG.STR. %, TOTAL STR., TD, TD %, SUB.ATT, REV., CTRL
    Significant Strikes include SIG.STR., SIG.STR. %, HEAD, BODY, LEG, DISTANCE, CLINCH, GROUND
    
    each type of stat has a summary of total stats for the fight, and individual round stats
    the sets of stats are returned as a list of lists
    e.g. [[totals - summary], [totals - round 1], [totals - round n]..., [significant strikes - summary], [significant strikes - round 1], [significant strikes - round n]...] 

    arguments:
    stats_from_soup (list): a list of fight stats from parse_fight_stats()

    returns: 
    a list of lists of fight stats
    """
    fighter_stats_clean = []
    for name, stats in itertools.groupby(stats_from_soup, lambda x: x ==
        stats_from_soup[0]):
        if name:
            fighter_stats_clean.append([])
        fighter_stats_clean[-1].extend(stats)
    return fighter_stats_clean

def organise_fighter_tott(tott_from_soup, fighter_tott_column_names, url):
    """
    organise list of fighter tale of the tape
    remove label of tale of the tape using regex
    e.g. 'Height:5'7"' to '5'7"
    convert and return list as df

    arguments:
    tott_from_soup (list): list of fighter tale of the tale from parse_fighter_tott()
    fighter_tott_column_names (list): list of column names for fighter tale of the tape
    url (str): url of fighter

    results:
    a df of fighter tale of the tape
    """
    # pdb.set_trace()
    fighter_tott_clean = [re.sub('^(.+?): ?', '', text) for text in
        tott_from_soup]
    fighter_tott_clean.append(url)
    fighter_tott_df = pd.DataFrame(columns=fighter_tott_column_names)
    fighter_tott_df.loc[len(fighter_tott_df)] = fighter_tott_clean
    return fighter_tott_df

def parse_event_details(soup, upcoming = False):
    """
    parse event details from soup
    includes names, urls, dates, locations of events
    clean each element in the list, removing '
' and ' ' 
    e.g cleans '
      Las Vegas, Nevada, USA
' into 'Las Vegas, Nevada, USA'
    return details as a df

    arguments:
    soup (html): output of get_soup()

    returns:
    a dataframe of event details
    """
    
    # pdb.set_trace()
    
    event_names = []
    event_urls = []
    event_dates = []
    event_locations = []
    for tag in soup.find_all('a', class_='b-link b-link_style_black'):
        event_names.append(tag.text.strip())
        event_urls.append(tag['href'])
    for tag in soup.find_all('span', class_='b-statistics__date'):
        event_dates.append(tag.text.strip())
    for tag in soup.find_all('td', class_=
        'b-statistics__table-col b-statistics__table-col_style_big-top-padding'
        ):
        event_locations.append(tag.text.strip())
        
    # If we're not doing upcoming events, ignore the first row as it's in the future
    if not upcoming:
        event_dates = event_dates[1:]
        event_locations = event_locations[1:]
        
    event_details_df = pd.DataFrame({'EVENT': event_names, 'URL':
        event_urls, 'DATE': event_dates, 'LOCATION': event_locations})
    return event_details_df

def parse_fight_details(soup):
    """
    parse fight details from soup
    includes urls, and fights
    create bout from fighters' names and create event column as keys
    return a df of fight details of an event

    arguments:
    soup (html): output of get_soup()
    
    returns:
    a df of fight details
    """
    fight_urls = []
    for tag in soup.find_all('tr', class_=
        'b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click'
        ):
        fight_urls.append(tag['data-link'])
    fighters_in_event = []
    for tag in soup.find_all('a', class_='b-link b-link_style_black'):
        fighters_in_event.append(tag.text.strip())
    fights_in_event = [(fighter_a + ' vs. ' + fighter_b) for fighter_a,
        fighter_b in zip(fighters_in_event[::2], fighters_in_event[1::2])]
    fight_details_df = pd.DataFrame({'BOUT': fights_in_event, 'URL':
        fight_urls})
    fight_details_df['EVENT'] = soup.find('h2', class_='b-content__title'
        ).text.strip()
    fight_details_df = utils.move_columns(fight_details_df, ['EVENT'], 'BOUT',
        'before')
    return fight_details_df

def parse_fight_details_url(url, config):
    """
    parse fight details from a url
    return a df of fight details of an event
    
    arguments:
    url (string):
    
    returns:
    a df of fight details
    """
    
    soup = get_website_soup(url, fast = True, chrome_binary = config['chrome_binary_location'], time_delay = True)
    
    # pdb.set_trace()
    
    fight_urls = []
    
    for tag in soup.find_all('tr', class_=
        'b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click'
        ):
        fight_urls.append(tag['data-link'])
        
    fighters_in_event = []
    
    for tag in soup.find_all('a', class_='b-link b-link_style_black'):
        fighters_in_event.append(tag.text.strip())
        
    weightclass = []
    
    for tag in soup.find_all('p', class_='b-fight-details__table-text'):
        weightclass.append(tag.text.strip())
        
    weightclass = [item for item in weightclass if 'weight' in item]
        
    # Remove any 'view matchup that might have been introduced
    fighters_in_event = [fighter for fighter in fighters_in_event if fighter != 'View Matchup']
    
    fights_in_event = [(fighter_a + ' vs. ' + fighter_b) for fighter_a,
        fighter_b in zip(fighters_in_event[::2], fighters_in_event[1::2])]
    
    fight_details_df = pd.DataFrame({'BOUT': fights_in_event, 'URL':
        fight_urls})
        
    fight_details_df['EVENT'] = soup.find('h2', class_='b-content__title'
        ).text.strip()
    
    fight_details_df = utils.move_columns(fight_details_df, ['EVENT'], 'BOUT',
        'before')
    
    fight_details_df['weightclass'] = pd.Series(weightclass)
   
    return fight_details_df

def parse_fight_results(soup):
    """
    parase fight results from soup
    results include event, bout, outcome weightclass, method, round, time, timeformat, referee, details
    clean each element in the list, removing '
' and ' ' 
    e.g cleans '
      Welterweight Bout
' into 'Welterweight Bout'
    details include description of finish or judges and scores
    judges and scores also include details of point deduction
    e.g. 'Point Deducted: Illegal Knee by Menne Tony Weeks 45 - 49.Doug Crosby 42 - 49.Jeff Mullen 44 - 49.'
    return fight results as a list

    arguments:
    soup (html): output of get_soup() parser

    returns:
    a list of fight results
    """
    fight_results = []
    fight_results.append(soup.find('h2', class_='b-content__title').text)
    for tag in soup.find_all('a', class_='b-link b-fight-details__person-link'
        ):
        fight_results.append(tag.text)
    for tag in soup.find_all('div', class_='b-fight-details__person'):
        for i_text in tag.find_all('i'):
            fight_results.append(i_text.text)
    fight_results.append(soup.find('div', class_=
        'b-fight-details__fight-head').text)
    fight_results.append(soup.find('i', class_=
        'b-fight-details__text-item_first').text)
    remaining_results = soup.find_all('p', class_='b-fight-details__text')
    for tag in remaining_results[0].find_all('i', class_=
        'b-fight-details__text-item'):
        fight_results.append(tag.text.strip())
    fight_results.append(remaining_results[1].get_text())
    fight_results = [text.replace('\n', '').replace('  ', '') for text in
        fight_results]
    return fight_results

def parse_fight_stats(soup):
    """
    parse full fight stats for both fighters from soup
    loop through soup to find all 'td' tags with the class 'b-fight-details__table-col'
    this returns a list of stats for both fighters in alternate order
    e.g. [0, 1, 2, 2, 20, 30] stats [0, 2, 20] belong to the first fighter and [1, 2, 30] belong to the second fighter
    use enumerate to add index to results
    stats with even indexes belongs to the first fighter and odd indexes belong to the second fighter
    clean each element in the list, removing '
' and ' ' 
    e.g cleans '
 fighter name 
' into 'fighter name' and  '
      19 of 32
    ' into '19 of 32'
    
    arguments:
    soup (html): output of get_soup() parser

    returns:
    two lists of fighter stats, one for each fighter
    """
    fighter_a_stats = []
    fighter_b_stats = []
    for tag in soup.find_all('td', class_='b-fight-details__table-col'):
        for index, p_text in enumerate(tag.find_all('p')):
            if index % 2 == 0:
                fighter_a_stats.append(p_text.text.strip())
            else:
                fighter_b_stats.append(p_text.text.strip())
    return fighter_a_stats, fighter_b_stats

def parse_fighter_details(soup, fighter_details_column_names):
    """
    parse fighter details from soup
    fighter details include first name, last name, nickname, and url
    returns dataframe with first, last, nickname, url

    arguments:
    soup (html): output of get_soup() parser

    returns:
    a dataframe of fighter details
    """
    fighter_names = []
    for tag in soup.find_all('a', class_='b-link b-link_style_black'):
        fighter_names.append(tag.text)
    fighter_urls = []
    for tag in soup.find_all('a', class_='b-link b-link_style_black'):
        fighter_urls.append(tag['href'])
    fighter_details = list(zip(fighter_names[0::3], fighter_names[1::3],
        fighter_names[2::3], fighter_urls[0::3]))
    fighter_details_df = pd.DataFrame(fighter_details, columns=
        fighter_details_column_names)
    return fighter_details_df

def parse_fighter_tott(soup):
    """
    parse fighter tale of the tape from soup
    fighter details contain fighter, height, weight, reach, stance, dob
    clean each element in the list, removing '
' and ' ' 
    e.g cleans '
      Jose Aldo
' into 'Jose Aldo'
    returns a list of fighter tale of the tape

    arguments:
    soup (html): output of get_soup() parser

    returns:
    a list of fighter tale of the tape
    """
    fighter_tott = []
    fighter_name = soup.find('span', class_='b-content__title-highlight').text
    fighter_tott.append('Fighter:' + fighter_name)
    tott = soup.find_all('ul', class_='b-list__box-list')[0]
    for tag in tott.find_all('i'):
        fighter_tott.append(tag.text + tag.next_sibling)
    fighter_tott = [text.replace('\n', '').replace('  ', '') for text in
        fighter_tott]
    return fighter_tott

def parse_organise_fight_results_and_stats(soup, url,
    fight_results_column_names, totals_column_names,
    significant_strikes_column_names):
    """
    parse and organise fight results and fight stats from soup
    this function combines other functions that parse fight results and stats into one
    and returns two dfs, one for fight results and the other for fight stats

    arguments:
    soup (html): output of get_soup() parser
    url (str): url of fight
    fight_results_df (df): an df
    fight_results_column_names (list): list of column names for fight results
    fight_stats_df (df):
    totals_column_names (list): list of column names for totals type stats
    significant_strikes_column_names (list): list of column names for significant strike type stats

    returns:
    two dfs for fight results and stats
    """
    # pdb.set_trace()
    fight_results = parse_fight_results(soup)
    fight_results.append('URL:' + url)
    fight_results_df = organise_fight_results(fight_results,
        fight_results_column_names)
    fighter_a_stats, fighter_b_stats = parse_fight_stats(soup)
    fighter_a_stats_clean = organise_fight_stats(fighter_a_stats)
    fighter_b_stats_clean = organise_fight_stats(fighter_b_stats)
    fighter_a_stats_df = convert_fight_stats_to_df(fighter_a_stats_clean,
        totals_column_names, significant_strikes_column_names)
    fighter_b_stats_df = convert_fight_stats_to_df(fighter_b_stats_clean,
        totals_column_names, significant_strikes_column_names)
    fight_stats_df = combine_fighter_stats_dfs(fighter_a_stats_df,
        fighter_b_stats_df, soup)
    return fight_results_df, fight_stats_df

# This function takes a data frame of fight details and a list of new UFC events. It then
# scrapes for the UFC events and updates the data

def update_fight_details_data(existing_fights_data, new_events_url_list, config, num_threads = 1, debug = False):

    # Create empty list for results
    scrape_results = []

    # Run a parallel scrape of the new fights data
    with ThreadPoolExecutor(max_workers = num_threads) as executor:
        future_to_url = {executor.submit(parse_fight_details_url, url, config): url for url in new_events_url_list}

        # Now iterate through the functions
        for future in as_completed(future_to_url):
            try:
                result = future.result()
                scrape_results.append(result)  # Collect the result (url, title)
            except Exception as e:
                url = future_to_url[future]  # Get the URL associated with the failing future
                print(f"Error for URL {url}: {e}")

    # If we want to debug, return the raw scrape results, otherwise clean the results
    if debug:
        return scrape_results

    # Now concatenate the new scrape results into a data frame
    if (len(scrape_results) == 0):
        print("Error: No new fight details returned")
        sys.exit()
    elif (len(scrape_results) > 1):
        new_fights_data = pd.concat(scrape_results)
    else:
        new_fights_data = scrape_results[0]
    
    # Check we have the correct columns
    if (list(new_fights_data.columns) == config['fight_details_column_names']):
        # If they are, add on the rows to the existing data
        parsed_fight_details_df = pd.concat([new_fights_data, existing_fights_data])
        del([existing_fights_data, new_fights_data])
    else:
        print("Error: new parsed fight details have different columns")
        print("Got:")
        print(list(new_fights_data.columns))
        print("Expecting:")
        print(config['fight_details_column_names'])
        sys.exit()

    return parsed_fight_details_df

# This function takes a data frame of fight results and a list of new UFC results urls. It then
# scrapes for the UFC events and updates the data. It does this using parallel scrapes

def update_fight_results_and_stats_data(existing_fight_results_data, existing_fight_stats_data, new_fights_url_list, config, num_threads = 1):

    # Create empty list for results
    scraped_fight_results = []
    scraped_fight_stats = []
    
    # Run a parallel scrape of the new fights data
    with ThreadPoolExecutor(max_workers = num_threads) as executor:
        
        # Get the html contents for the url
        future_to_url = {executor.submit(scraping.get_soup, url): url for url in new_fights_url_list}

        # Now iterate through the functions
        for future in as_completed(future_to_url):
            # Get the url used
            url = future_to_url[future]
            try:
                # Parse the data and get the results
                fight_result, fight_stats = parse_organise_fight_results_and_stats(
                    future.result(),
                    url,
                    config['fight_results_column_names'],
                    config['totals_column_names'],
                    config['significant_strikes_column_names']      
                )
                
                # Save these results to the results list then remove
                scraped_fight_results.append(fight_result)
                scraped_fight_stats.append(fight_stats)
                del([fight_result, fight_stats])
                
            except Exception as e:
                print(f"Error for URL {url}: {e}")
    
    # Now concatenate the new scrape results into a data frame
    if (len(scraped_fight_results) == 0):
        print("Error: No new fight details returned")
        sys.exit()
    elif (len(scraped_fight_results) > 1):
        new_fight_results_data = pd.concat(scraped_fight_results)
        new_fight_stats_data = pd.concat(scraped_fight_stats)
    else:
        new_fight_results_data = scraped_fight_results[0]
        new_fight_stats_data = scraped_fight_stats[0]
    
    del([scraped_fight_results, scraped_fight_stats])    
    
    # Now attach the new results to the original data
    new_fight_results_data = pd.concat([new_fight_results_data, existing_fight_results_data])
    new_fight_stats_data = pd.concat([new_fight_stats_data, existing_fight_stats_data])

    return new_fight_results_data, new_fight_stats_data
