def parse_fight_details_url(url):
    """
    parse fight details from a url
    return a df of fight details of an event
    
    arguments:
    url (string):
    
    returns:
    a df of fight details
    """
    soup = get_soup(url)
    
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
    
    fight_details_df = move_columns(fight_details_df, ['EVENT'], 'BOUT',
        'before')
    
    fight_details_df['weightclass'] = pd.Series(weightclass)
   
    return fight_details_df
