from ablinston_ml.scraping import get_website_soup
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import pandas as pd
import re
import string

from mma_functions.config import config

import random

# This function extracts all the event urls from a fighter page on UFCstats.com
# This is used to find non-UFC events that haven't been parsed

# headers is the headers of the table we're looking in for the urls

def extract_event_urls(url, headers, url_starts_with = "http://ufcstats.com/event-details"):
    
    # Get soup from the url
    soup = get_soup(url)
    
    # Define the expected headers to locate the correct table
    # headers = ["W/L", "Fighter", "Kd", "Str", "Td", "Sub", "Event", "Method", "Round", "Time"]
    headers_pattern = [re.compile(rf"\b{header}\b", re.IGNORECASE) for header in headers]
    
    # Find the correct table
    table = None
    for tbl in soup.find_all("table"):
        table_headers = [th.text.strip() for th in tbl.find_all("th")]
        
        # Check if all expected headers are present in this table
        if all(any(pattern.search(header) for header in table_headers) for pattern in headers_pattern):
            table = tbl
            break

    if not table:
        # print("Table with specified headers not found.")
        return []  # Return empty list if the table is not found

    # Extract all URLs that match the specified pattern
    urls = []
    for row in table.find_all("tr")[1:]:  # Skip the header row
        for link in row.find_all("a", href=True):
            if link['href'].startswith(url_starts_with):
                urls.append(link['href'])

    return urls

def generate_alphabetical_urls():
    """
    generate a list of alphabetical urls for fighter details
    fighter urls are split by their last name and categorised alphabetically
    loop through each character in the alphabet from a to z to parse all the urls
    return all fighter urls as a list

    arguments:
    none

    returns:
    a list of urls of fighter details
    """
    list_of_alphabetical_urls = []
    for character in list(string.ascii_lowercase):
        list_of_alphabetical_urls.append(
            'http://ufcstats.com/statistics/fighters?char=' + character +
            '&page=all')
    return list_of_alphabetical_urls

def get_soup(url):
    """
    get soup from url using beautifulsoup

    arguments:
    url (str): url of page to parse

    returns:
    soup
    """
    # get_website_soup drops to a browser itself if the fast fetch gets blocked
    return get_website_soup(url, fast = True, time_delay = True,
                            chrome_binary = config['chrome_binary_location'])

# This function gets the response soup from a website either using selenium or a fast response

async def get_website_soup_async(
        url,
        playwright_context
    ):
    
    # Go to the page
    page = await context.new_page()
    await page.goto(url)
    content = await page.content()
    await page.close()
    
    # Get the soup
    soup = BeautifulSoup(page.content(), 'html.parser')
        
    return soup

# This function tables an html table and converts it into a data frame

def html_table_to_dataframe(html):
    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find the first table in the HTML
    table = soup.find('table')
    
    # Initialize a list to hold the rows
    data = []
    
    # Iterate through the table rows
    for row in table.find_all('tr'):
        # Extract cell text from each row
        cells = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
        data.append(cells)
    
    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data)
    
    # Set the first row as header if it looks like a header
    if df.iloc[0].apply(lambda x: x.isalpha()).all():  # Check if the first row contains only letters
        df.columns = df.iloc[0]  # Set the first row as header
        df = df[1:].reset_index(drop=True)  # Remove the header row from the data

    return df

# This function gets all event urls for UFCstats.com given a list of fighter urls

def parallel_get_all_event_urls_from_fighters(fighter_list_urls, fighter_result_headers, num_threads = 3):

    # Keep counters to track progress
    total_fighters = len(fighter_list_urls)
    completed_count = 0

    scrape_results = []
    with ThreadPoolExecutor(max_workers = num_threads) as executor:
        future_to_url = {executor.submit(extract_event_urls, fighter_url, fighter_result_headers): fighter_url for fighter_url in fighter_list_urls}

        # Collect results as they complete
        for future in as_completed(future_to_url):
            try:
                result = future.result()
                scrape_results.append(result)  # Collect the result (url, title)
            except Exception as e:
                print(f"Error for URL {url}: {e}")

            # Increment the completed count
            completed_count += 1
            print(f"Completed {completed_count} out of {total_fighters} tasks")
            
    # This ensures that all threads are done
    executor.shutdown(wait=True)

    # Combine the results
    return [item for sublist in scrape_results for item in sublist]

def parallel_scrape(url_list, parse_function, output_df = pd.DataFrame(), num_threads = 1, **kwargs):
    """
    scrape from a list of urls

    arguments:
    url_list: a list of url strings
    parse_function: the function to be run on the data scraped from the page
    output_df: the output pandas df that the parsed results
    num_threads: the number of CPU threads to use
    **kwargs: any other keyword arguments to pass to the parse_function

    """
    #pdb.set_trace()
    with ThreadPoolExecutor(max_workers = num_threads) as executor:
        future_to_url = {executor.submit(get_soup, url): url for url in url_list}

        # Now iterate through the functions
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                # Check to see whether we need the url in the function and input it if needed
                if 'url' in inspect.signature(parse_function).parameters:
                    result = parse_function(future.result(), **kwargs, url = url)
                else:
                    result = parse_function(future.result(), **kwargs)
                output_df = pd.concat([output_df, result])
                print(f'Scraped url: {url}')
            except Exception as exc:
                print(f'Error scraping {url}: {exc}')

    return output_df

# This function is specifically for parallel scraping event details for a list of events

def parallel_scrape_event_details(
        url_list,
        num_threads = 1):

    # Keep counters to track progress
    total_urls = len(url_list)
    completed_count = 0

    scrape_results = []
    with ThreadPoolExecutor(max_workers = num_threads) as executor:
        future_to_url = {executor.submit(scrape_event_details, [url]): url for url in url_list}

        # Collect results as they complete
        for future in as_completed(future_to_url):
            try:
                result = future.result()
                scrape_results.append(result)  # Collect the result (url, title)
            except Exception as e:
                print(f"Error for URL {url}: {e}")

            # Increment the completed count
            completed_count += 1
            print(f"Completed {completed_count} out of {total_urls} tasks")
            
    # This ensures that all threads are done
    executor.shutdown(wait=True)

    # Combine the results
    return pd.concat(scrape_results)

# This function gets a random user agent

def random_user_agent():

    user_agents = [
        # Chrome (Windows)
        # "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        # "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        
        # # Chrome (Mac)
        # "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        # "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",

        # Firefox (Windows)
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:117.0) Gecko/20100101 Firefox/117.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:118.0) Gecko/20100101 Firefox/118.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:119.0) Gecko/20100101 Firefox/119.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        # "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
        # "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
        # "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:130.0) Gecko/20100101 Firefox/130.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
        # "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
        # "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0"

        # # Firefox (Mac)
        # "Mozilla/5.0 (Macintosh; Intel Mac OS X 12.6; rv:117.0) Gecko/20100101 Firefox/117.0",
        # "Mozilla/5.0 (Macintosh; Intel Mac OS X 11.7; rv:114.0) Gecko/20100101 Firefox/114.0",

        # Edge (Windows)
        # "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
        # "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",

        # Safari (Mac)
        # "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/537.36 (KHTML, like Gecko) Version/16.5 Safari/537.36",
        # "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_4) AppleWebKit/537.36 (KHTML, like Gecko) Version/16.4 Safari/537.36",

        # Android Chrome
        # "Mozilla/5.0 (Linux; Android 14; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Mobile Safari/537.36",
        # "Mozilla/5.0 (Linux; Android 13; Pixel 7 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",

        # # iPhone Safari
        # "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/537.36",
        # "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/537.36",
    ]

    return random.choice(user_agents)

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
