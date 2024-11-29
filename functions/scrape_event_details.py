# This function scrapes event details from UFCStats.com given a list of urls to events

def scrape_event_details(url_list):
    """
    Scrapes event details from each URL in url_list and returns a DataFrame with the results.

    Args:
        url_list (list of str): List of URLs to scrape.

    Returns:
        pd.DataFrame: DataFrame containing the URL, event name, date, and location.
    """
    event_details = []

    for url in url_list:
        try:
            # Use the provided get_soup function to get the BeautifulSoup object
            soup = get_soup(url)

            # Extract the event name (assumes it's in an h2 tag)
            event_name = soup.find('h2').get_text(strip=True)

            # Locate the list items for date and location within the ul with the specific class
            details_list = soup.find('ul', class_='b-list__box-list')
            date = None
            location = None

            if details_list:
                for item in details_list.find_all('li', class_='b-list__box-list-item'):
                    title = item.find('i', class_='b-list__box-item-title')
                    if title:
                        if "Date:" in title.get_text():
                            date = title.next_sibling.strip()  # Gets the date text after the <i> tag
                        elif "Location:" in title.get_text():
                            location = title.next_sibling.strip()  # Gets the location text after the <i> tag


            # Append the scraped details to the list
            event_details.append({
                "EVENT": event_name,
                "URL": url,
                "DATE": date,
                "LOCATION": location
            })

        except Exception as e:
            print(f"Error scraping {url}: {e}")

    # Convert the list of dictionaries to a DataFrame
    return pd.DataFrame(event_details)

# Example usage:
# url_list = ["https://example.com/event1", "https://example.com/event2"]
# event_df = scrape_event_details(url_list)
# print(event_df)
