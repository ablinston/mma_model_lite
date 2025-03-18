def find_fighter_odds_url_in_html(soup, fighter_name, odds_urls_overide_df):
    """
    This function looks through the bestfightodds.com html search results for a fighter and returns the best url for their odds stats
    """
    
    # skip if no possible matches found
    if soup.find('table') == None:
        print(f"Error scraping {fighter_name}: {e}")
        return ''

    fighter_name_parts = fighter_name.lower().split()

    # Create placeholder for the 
    link_url = ''
    found = False
    exact_find = False
    
    # See if url is in override
    if clean_name_for_odds_search(fighter_name) in odds_urls_overide_df['fighter'].tolist():
        link_url = odds_urls_overide_df[odds_urls_overide_df['fighter'] == clean_name_for_odds_search(fighter_name)]['url'].iloc[0]
        exact_find = True
    else:
        # Then try to find the fighter in the table
        # Search through all tables in the soup
        for table in soup.find_all('table'):
            # Iterate over all rows in the table
            for row in table.find_all('tr'):
                # Iterate over each cell in the row
                for cell in row.find_all('td'):
                    # Check if the cell contains the exact search string
                    if (clean_name_for_odds_search(fighter_name) == clean_name_for_odds_search(cell.get_text())) and not exact_find:
                        # Find the <a> tag inside the cell
                        link_tag = cell.find('a')
                        if link_tag and not exact_find:
                            # Extract the href attribute (the link)
                            link_url = 'https://www.bestfightodds.com' + link_tag.get('href')
                            print(f'Found exact string: {fighter_name}, Link: {link_url}')
                            # Take only the first entry, so flag as found
                            exact_find = True
                            found = True
                    # Check if the cell contains the search string
                    elif all(clean_name_for_odds_search(part) in clean_name_for_odds_search(cell.get_text()) for part in fighter_name_parts) and not found:
                        # Find the <a> tag inside the cell
                        link_tag = cell.find('a')
                        if link_tag and not found:
                            # Extract the href attribute (the link)
                            link_url = 'https://www.bestfightodds.com' + link_tag.get('href')
                            print(f'Found string: {fighter_name}, Link: {link_url}')
                            # Take only the first entry, so flag as found
                            found = True
    
    return link_url
