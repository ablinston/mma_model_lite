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
