from ablinston_ml.scraping import clear_captcha, get_website_soup, start_browser
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import timedelta
from functools import partial
import json
import pandas as pd
import random
from rapidfuzz import fuzz
import re
from re import sub
import time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

from mma_functions import dataset, parsing, scraping, utils
from mma_functions.config import bet_config, config, odds_scrape_config

# This function takes a series of dates from BetFightOdds and converts them to pandas datetime

def betfightodds_dates_to_datetime(dates):
    
    clean_dates = dates.apply(lambda x: sub(r"(\d+)(st|nd|rd|th)", r"\1", x))

    return pd.to_datetime(clean_dates, format="%b %d %Y")

# This function takes an html table from BetFightOdds.com and turns it into a properly formatted pandas data frame

def betfightodds_table_to_df(table, fighter):
    
    # pdb.set_trace()
    
    # Extract headers and rows in one go using list comprehensions
    # First remove all tds
    # Find all <td> elements and remove them
    decomposed_table = copy.deepcopy(table)
    
    for td in decomposed_table.find_all("td"):
        td.decompose()  # This removes the <td> element from the tree

    headers = [header.text for header in decomposed_table.find_all("th")]
    rows = [[cell.text for cell in row.find_all("td")] for row in table.find_all("tr")[1:]]

    # The rows contain 3 vectors, one with the event, the next with the fighter's odds, the final with the opponents odds and the date
    # The event names on the website are not always correct, so only provide Bout and date (date not always correct either. Sometimes off by 1 day!)

    # Remove lists of length 1, as these are just titles
    filtered_list = [lst for lst in rows if len(lst) > 1]
    # Now split the data, as its odds for first fighter then second fighter
    even_list = [lst for idx, lst in enumerate(filtered_list) if idx % 2 == 0]
    odd_list = [lst for idx, lst in enumerate(filtered_list) if idx % 2 != 0]

    # Create new column headers
    custom_headers = ["open_odds", "closing_odds_low", "delete1", "closing_odds_high", "delete2", "delete3", "event"]

    # Convert directly to DataFrame
    df = pd.DataFrame(even_list, columns=custom_headers)

    # Remove columns not needed
    df.drop(["delete1", "delete2", "delete3"], axis=1, inplace=True)
    
    # Finally, Add on the dates and opponent
    dates = pd.DataFrame(odd_list)[4]
    df["date"] = betfightodds_dates_to_datetime(dates)

    # Make sure the cases of the letters in the fighters name in the headers matches what is expected from the UFC data
    headers = [fighter if fighter.lower() == item.lower() else item for item in headers]

    df["fighter"] = fighter
    
    # Find the most common element, it's usually a different spelling of the fighters name
    mode_name = clean_name_for_odds_search(max(set(headers), key=headers.count))
    
    # Check it at least semi matches, otherwise wipe it
    if fuzz.ratio(mode_name, clean_name_for_odds_search(fighter)) < 80:
        mode_name = ""
    
    df["opponent"] = [item for item in headers if clean_name_for_odds_search(item) not in ["matchup", "open", "closingrange", "movement", "", "event", clean_name_for_odds_search(fighter), mode_name]] # exclude the most common element as it is usually their name

    return df

# This function takes a string and cleans it ready for the odds search

def clean_name_for_odds_search(name_string):

    return name_string.lower().replace(".", "").replace(" ", "").replace("'", "").replace("-", "").replace("saint", "st")

def find_fighter_odds_url_in_html(soup, fighter_name, odds_urls_overide_df):
    """
    This function looks through the bestfightodds.com html search results for a fighter and returns the best url for their odds stats
    """
    
    # skip if no possible matches found
    if soup.find("table") == None:
        print(f"Error scraping {fighter_name}: {e}")
        return ""

    fighter_name_parts = fighter_name.lower().split()

    # Create placeholder for the 
    link_url = ""
    found = False
    exact_find = False
    
    # See if url is in override
    if clean_name_for_odds_search(fighter_name) in odds_urls_overide_df["fighter"].tolist():
        link_url = odds_urls_overide_df[odds_urls_overide_df["fighter"] == clean_name_for_odds_search(fighter_name)]["url"].iloc[0]
        exact_find = True
    else:
        # Then try to find the fighter in the table
        # Search through all tables in the soup
        for table in soup.find_all("table"):
            # Iterate over all rows in the table
            for row in table.find_all("tr"):
                # Iterate over each cell in the row
                for cell in row.find_all("td"):
                    # Check if the cell contains the exact search string
                    if (clean_name_for_odds_search(fighter_name) == clean_name_for_odds_search(cell.get_text())) and not exact_find:
                        # Find the <a> tag inside the cell
                        link_tag = cell.find("a")
                        if link_tag and not exact_find:
                            # Extract the href attribute (the link)
                            link_url = "https://www.bestfightodds.com" + link_tag.get("href")
                            print(f"Found exact string: {fighter_name}, Link: {link_url}")
                            # Take only the first entry, so flag as found
                            exact_find = True
                            found = True
                    # Check if the cell contains the search string
                    elif all(clean_name_for_odds_search(part) in clean_name_for_odds_search(cell.get_text()) for part in fighter_name_parts) and not found:
                        # Find the <a> tag inside the cell
                        link_tag = cell.find("a")
                        if link_tag and not found:
                            # Extract the href attribute (the link)
                            link_url = "https://www.bestfightodds.com" + link_tag.get("href")
                            print(f"Found string: {fighter_name}, Link: {link_url}")
                            # Take only the first entry, so flag as found
                            found = True
    
    return link_url

# This function takes a list of fighters fighting on a card and the parsed date of the event and finds the url on bestfightodds.com
# for the event they were competing in

## AB: THIS FUNCTION ISN'T FINISHED. I ABANDONED IT BECAUSE THE EVENT DATA IS TOO FULL OF RUBBISH

def get_event_odds_data(fighter_list, date, firefox_location, gecko_driver):

    ###########
    # Setup headless Firefox for getting odds information
    options = Options()
    options.headless = True
    options.accept_insecure_certs = True
    # Create a Service object with the path to Firefox driver
    options.binary_location = firefox_location # firefox_binary
    service = Service(gecko_driver)
    # Create a WebDriver instance
    driver = webdriver.Firefox(service=service, options=options)
    ###########

    # First, loop through the fighters to try to find the event url
    for fighter_name in fighter_list:
        try:
    
            # Navigate to the website
            driver.get("https://www.bestfightodds.com/search?query=" + fighter_name)
            
            # Extract the page source and convert to xml
            soup = BeautifulSoup(driver.page_source, "lxml")
    
            # skip if no possible matches found
            if soup.find("table") == None:
                # Quit the browser
                driver.quit()
                return pd.DataFrame()
    
            # Create placeholder for the 
            link_url = ""
            found = False
            # Then try to find the fighter in the table
            # Search through all tables in the soup
            for table in soup.find_all("table"):
                # Iterate over all rows in the table
                for row in table.find_all("tr"):
                    # Iterate over each cell in the row
                    for cell in row.find_all("td"):
                        # Check if the cell contains the search string
                        if (fighter_name in cell.get_text()) and not found:
                            # Find the <a> tag inside the cell
                            link_tag = cell.find("a")
                            if link_tag and not found:
                                # Extract the href attribute (the link)
                                link_url = "https://www.bestfightodds.com" + link_tag.get("href")
                                print(f"Found string: {fighter_name}, Link: {link_url}")
                                # Take only the first entry, so flag as found
                                found = True
    
            # Check if a valid output has been found
            if link_url == "":
                # Quit the browser
                driver.quit()
                # Try the next fighter
                break
    
            # Navigate to the website
            driver.get(link_url)
    
            # Get the html table from the page
            soup = BeautifulSoup(driver.page_source, "html.parser")
    
            # Find the url of the event from the fighter table
            event_url = get_event_odds_url(soup, date)
            
            # Check if it found one
            if event_url == None:
                # Quit the browser
                driver.quit()
                # Try the next fighter
                break
            
            # Otherwise, navigate to the event url
            driver.get("https://www.bestfightodds.com" + event_url)
            
            # Get the html table from the page
            soup = BeautifulSoup(driver.page_source, "html.parser")
            
            # Read the first table and convert it to a data frame
            event_df = scraping.html_table_to_dataframe(driver.page_source)
            
            ## AB: ABANDONED THIS FUNCTION HERE. THE IMPORTED TABLES ARE TOO FULL OF CRAP
            
        except Exception as e:
            print(f"Error scraping {fighter_name}: {e}")
            return pd.DataFrame()
    
        finally:
    
            # Quit the browser
            driver.quit()
    

    return soup

# This function takes the soup from a fighter's bestfightodds.com webpage and a parsed date.
# It looks to match the date to the listed events the fighter has been in and returns the url
# of the event page if it finds it.

def get_event_odds_url(soup, date):

    # Find the table in the soup object
    table = soup.find("table")
    
    # Specify the target date to search for in the correctly formatted way for the website
    target_date = date.strftime("%b ") + f"{date.day}" + utils.get_day_suffix(date.day) + " " + f"{date.year}"
    
    # Find all relevant <td> elements
    tds = soup.find_all("td", colspan="8", scope="row")
    
    # Initialize variable to hold the href
    found_href = None
    
    # Iterate through the cells to find the target date
    for td in tds:
        # Check if the target date is in the cell's text
        if target_date in td.get_text():
            # If found, get the previous <a> href
            link_tag = td.find("a")
            if link_tag:
                found_href = link_tag.get("href")  # Extract href
            break  # Exit the loop after finding the first match
    
    return found_href

def get_fighter_data(soup, url, **kwargs):
    """
    get fighter data and organise from a url

    arguments:
    soup: a soup output from the get_soup function
    url: the url that the soup is from

    returns:
    data frame
    """
    return parsing.organise_fighter_tott(parsing.parse_fighter_tott(soup), **kwargs, url=url)

# This function is specifically for parallel scraping odds data

def parallel_scrape_event_odds_fast(
        events_df, 
        output_df=pd.DataFrame(), 
        config=[],
        num_threads=1, 
        firefox_location="", 
        gecko_driver="",
        fast=True):

    
    scrape_results = []
    
    # Keep counters to track progress
    total_events = len(events_df)
    completed_count = 0
        
    # pdb.set_trace()
    
    ###########
    # Setup headless Firefox for getting odds information
    options = Options()
    options.headless = True
    options.accept_insecure_certs = True
    # Create a Service object with the path to Firefox driver
    options.binary_location = firefox_location # firefox_binary
    service = Service(gecko_driver)
    # Create a WebDriver instance
    driver = webdriver.Firefox(service=service, options=options)
    ###########
    
    # Get the cookies for the odds webpages using a Selenium call
    driver.get("https://www.bestfightodds.com/events/ufc-3570")

    # Get cookies from Selenium
    selenium_cookies = driver.get_cookies()
    cookies_dict = {cookie["name"]: cookie["value"] for cookie in selenium_cookies}
    # Quit the browser
    driver.quit()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_url = {executor.submit(scrape_event_odds_fast, row, cookies_dict, fast): row for row in events_df.iterrows()}

        # Collect results as they complete
        for future in as_completed(future_to_url):
            try:
                result = future.result()
                scrape_results.append(result)  # Collect the result (url, title)
            except Exception as e:
                print(f"Error for row {row}: {e}")

            # Increment the completed count
            completed_count += 1
            print(f"Completed {completed_count} out of {total_events} tasks")
            
    # This ensures that all threads are done
    executor.shutdown(wait=True)
    
    if len(scrape_results) == 0:
        return pd.DataFrame()
    else:    
        # Combine the results
        return pd.concat(scrape_results, ignore_index=True)

# This function is specifically for parallel scraping odds data

def parallel_scrape_odds_fast(
        fighter_list, 
        output_df=pd.DataFrame(), 
        config=[],
        num_threads=1, 
        odds_urls_overide_df=None,
        event_urls=False,
        fast=True,
        playwright=False):

    
    # Keep counters to track progress
    total_fighters = len(fighter_list)
    completed_count = 0

    odds_scrape_results = []
    event_scrape_results = []
            
    # ###########
    # # Setup headless Firefox for getting odds information
    # options = Options()
    # # options.headless = True
    # options.accept_insecure_certs = True
    # options.set_preference("general.useragent.override",
    #                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    #                "AppleWebKit/537.36 (KHTML, like Gecko) "
    #                "Chrome/121.0.0.0 Safari/537.36")
    # # Create a Service object with the path to Firefox driver
    # options.binary_location = firefox_location # firefox_binary
    # service = Service(gecko_driver)
    # options.set_preference("dom.webdriver.enabled", False)
    # options.set_preference("useAutomationExtension", False)
    # options.set_preference("dom.webnotifications.enabled", False)
    # # Create a WebDriver instance
    # driver = webdriver.Firefox(service=service, options=options)
    # ###########
    
    # # Get the cookies for the odds webpages using a Selenium call
    # driver.get("https://www.bestfightodds.com/")
    # pdb.set_trace()
    # Get cookies from Selenium
    # selenium_cookies = driver.get_cookies()
            
    # Look up the true global config (this function is defined via exec() into the
    # caller's namespace, so globals() here is that namespace, not the local
    # 'config' parameter above, which defaults to [] and is otherwise unused)
    _global_config = globals().get("config")
    chrome_binary_location = _global_config.get("chrome_binary_location") if isinstance(_global_config, dict) else None

    # Run in a window rather than headless so the Cloudflare checkbox can be clicked
    browser = start_browser(
        "https://www.bestfightodds.com/",
        chrome_binary=chrome_binary_location,
        headless=False)

    # Clear the checkbox, falling back to a delay to allow it to be manually done
    clear_captcha(browser)

    cookies = browser.get_all_cookies()

    cookies_dict = {cookie.name: cookie.value for cookie in cookies}
    # Quit the browser unless we are going through one at a time
    if num_threads > 1:
        browser.stop()
    
    scrape_func = partial(
        scrape_fighter_odds_fast,
        cookies_dict=cookies_dict,
        odds_urls_overide_df=odds_urls_overide_df,
        event_urls=event_urls,
        playwright=playwright,
        fast=fast,
    )
    
    # Check if we have more than 1 thread
    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_url = {executor.submit(scrape_func, fighter): fighter for fighter in fighter_list}
    
            # Collect results as they complete
            for future in as_completed(future_to_url):
                try:
                    odds_result, events_result = future.result()
                    odds_scrape_results.append(odds_result)  # Collect the result (url, title)
                    event_scrape_results.append(events_result)  # Collect the result (url, title)
                    
                except Exception as e:
                    print(f"Error for fighter {future}: {e}")
    
                # Increment the completed count
                completed_count += 1
                print(f"Completed {completed_count} out of {total_fighters} tasks")
                
        # This ensures that all threads are done
        executor.shutdown(wait=True)
    
    else:
        for fighter in fighter_list:
            
            odds_result, events_result = scrape_fighter_odds_fast(
                fighter,
                cookies_dict=cookies_dict,
                odds_urls_overide_df=odds_urls_overide_df,
                event_urls=event_urls,
                playwright=playwright,
                fast=fast,
                browser=browser)
            
            odds_scrape_results.append(odds_result)  # Collect the result (url, title)
            event_scrape_results.append(events_result)  # Collect the result (url, title)
        
    # pdb.set_trace()    
    
    # Combine into a df
    if len(odds_scrape_results) > 0:
        all_fighter_odds = pd.concat(odds_scrape_results)
    else:
        all_fighter_odds = pd.DataFrame()
    
    browser.stop()

    # Combine the results
    if event_urls:
        
        # Get rid of duplicate events and keep the most common date for the event
        if len(all_fighter_odds) > 0:
            event_urls_odds_agg = pd.concat(event_scrape_results).groupby("event_url", as_index=False)["event_date"].apply(lambda x: x.mode().iloc[0])
        else:
            event_urls_odds_agg = pd.DataFrame()
        
        return all_fighter_odds, event_urls_odds_agg
    else:
        return all_fighter_odds

# This function scrapes the fighter odds data for a fighter.

# It returns a data frame with the odds for all the fights those fighters have been involved with.

# This fast function requires cookies so can be more temperamental. That's why there is an alternate version sans '_fast'

# event_urls_only tells the function whether to only return the urls of all events for a fighter

def scrape_event_odds_fast(
        event_details, # A tuple with the event url and date together
        cookies_dict,
        fast=True):

    # pdb.set_trace()

    try:
        
        # Arrive from another page on the site rather than cold, to look like normal browsing
        referer = f"https://www.bestfightodds.com/events/ufc-{random.randint(3500, 3700)}"

        soup = get_website_soup("https://www.bestfightodds.com" + event_details[1]["event_url"], fast=fast, referer=referer, cookies=cookies_dict, chrome_binary=config["chrome_binary_location"])
        # # Navigate to the website
        # response = requests.get('https://www.bestfightodds.com' + event_details[1]['event_url'], headers=headers, cookies=cookies_dict)
        
        # if response.status_code in [403, 429]:
        #     print("Blocked! Try changing User-Agent, IP, or waiting.")

        
        # # Extract the page source and convert to xml
        # soup = BeautifulSoup(response.text, 'html.parser')

        # skip if no possible matches found
        if soup.find("table") == None:
            print(f"Error scraping {event_details[1]['event_url']}")
            return pd.DataFrame()

        # divs = soup.find_all('div', {'class': 'table-div'})
        # Find the table in the soup object
        tables = soup.find_all("table")
        
        # Check if table is found
        if len(tables) == 2:
        
            # Get the odds value tables
            table = tables[1]  # Selecting the second table

            rows = table.find_all("tr")[1:]  # Skip header row

            # Construct a list of the rows searching cells by class
            data = [
                [
                    row.find("th", {"scope": "row"}).text,  # Fighter name
                    *map(lambda td: td.find("span").text.strip(), row.find_all("td", class_=odds_scrape_config["odds_cell_class"])),  # Odds
                    *map(lambda td: td.text.strip(), (td for td in row.find_all("td") if td.get("class", []) != [odds_scrape_config["odds_cell_class"]]))  # Other columns
                ]
                for row in rows
            ]

            # Create column names dynamically
            odds_count = max(len(row) - 1 for row in data)  # First column is 'fighter'
            columns = ["fighter"] + [f"Odd {i+1}" for i in range(odds_count)] + [f"Missing {i+1}" for i in range(len(data[0]) - odds_count - 1)]

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=columns)
            
            # Remove prop bet columns
            exclude_strings = ["distance", " is ", "Fight", " other ", " round ", " rounds", "wins by ", "has more ", "by decision", " has ", " points ", "FOTN", "scorecard", "awarded", "wins", "bleed", "does not"]
            df = df[~df["fighter"].str.contains("|".join(exclude_strings), na=False, case=False)]
                        
            # Drop any columns not recognised
            df = df.iloc[:, 0:(len(odds_scrape_config["events_colnames"]) + 1)]
            df.columns = ["fighter"] + odds_scrape_config["events_colnames"]
            
            # Add opponent
            if len(df) % 2 == 0:
                df = dataset.assign_opponents(df)
            else:
                print(f"Error in length of data for {event_details[1]['event_url']}. Opponent can't be determined")
                return pd.DataFrame()
            
            # Add on the date
            df["date"] = event_details[1]["event_date"]
            
            # Sort column formats
            # df['date'] = betfightodds_dates_to_datetime(df['date']) #  Now set to date in events urls
            df.loc[:, odds_scrape_config["events_colnames"]] = df[odds_scrape_config["events_colnames"]].apply(pd.to_numeric, errors="coerce")
            
        else:
            print("Table not found")
            return pd.DataFrame()
    
    except Exception as e:
        print(f"Error scraping {event_details[1]['event_url']}")
        return pd.DataFrame()
    
    if len(df) > 0:
        print(f"Scrape of {event_details[1]['event_url']} successful")
        
    return df

# This function scrapes the fighter odds data for a fighter.

# It returns a data frame with the odds for all the fights those fighters have been involved with.

# This fast function requires cookies so can be more temperamental. That's why there is an alternate version sans '_fast'

# event_urls_only tells the function whether to only return the urls of all events for a fighter

async def scrape_fighter_odds_async(
        fighter_name,
        odds_urls_overide_df,
        event_urls_only=False,
        playwright_context=None):

    # pdb.set_trace()

    try:
                
        # Extract the page source and convert to xml
        soup = scraping.get_website_soup_async(
            url="https://www.bestfightodds.com/search?query=" + fighter_name,
            fast=fast,
            playwright_context=playwright_context
        )            

        # skip if no possible matches found
        if soup.find("table") == None:
            print(f"Error scraping {fighter_name}: {e}")
            return pd.DataFrame()

        link_url = find_fighter_odds_url_in_html(soup, fighter_name, odds_urls_overide_df)

        # Check if a valid output has been found or if we have gone directly to the fighters odds page
        if link_url == "" and (fighter_name not in soup.title.string):
            print(f"Error scraping {fighter_name}: {e}")
            return pd.DataFrame()

        # Navigate to the website if we're not already there
        if fighter_name not in soup.title.string:
            # Navigate to the website
            soup = get_website_soup(
                url=link_url,
                fast=fast,
                playwright_context=playwright_context,
                headers=headers, # Only needed for fast
                cookies=cookies_dict, # Only needed for fast
                firefox_binary=firefox_binary, # Only needed for NOT fast
                gecko_driver=gecko_driver
            )

        # pdb.set_trace()
        
        # See if we want to return odds data or event urls
        if event_urls_only:
            # Collect the event urls
            # Extract all event URLs and their corresponding dates efficiently
            events = [(a["href"], a.next_sibling.strip()) for a in soup.select('a[href^="/events/"]') if a.next_sibling]
            
            # Convert to DataFrame
            odds_df = pd.DataFrame(events, columns=["event_url", "event_date"])

        else:
            # Find the table in the soup object
            table = soup.find("table")
    
            # Convert the table to a data frame and add the rows to the results
            odds_df = betfightodds_table_to_df(table, fighter_name)
    
    except Exception as e:
        print(f"Error scraping {fighter_name}: {e}")
        return pd.DataFrame()

    return odds_df

# This function scrapes the fighter odds data for a fighter.

# It returns a data frame with the odds for all the fights those fighters have been involved with.

# This fast function requires cookies so can be more temperamental. That's why there is an alternate version sans '_fast'

# event_urls_only tells the function whether to only return the urls of all events for a fighter

def scrape_fighter_odds_fast(
        fighter_name, 
        cookies_dict, 
        odds_urls_overide_df,
        event_urls=False,
        fast=True,
        playwright=False,
        browser=None):

    # pdb.set_trace()
    
    if fast:
        # time.sleep(random.uniform(1, 3))  # Sleep for a random amount of time between requests to avoid being detected
        pass
    
    try:
                
        if playwright == True:
            # Use parallel scraping within playwright
            with sync_playwright() as p:
                playwright_browser = p.chromium.launch(headless=True)
                playwright_context = playwright_browser.new_context()
        else:
            playwright_context = None
            
    
        # Extract the page source and convert to xml
        soup = get_website_soup(
            url="https://www.bestfightodds.com/search?query=" + fighter_name,
            fast=fast,
            playwright_context=playwright_context,
            cookies=cookies_dict, # Only needed for fast
            chrome_binary=config["chrome_binary_location"], # Only needed for NOT fast
            browser=browser
        )

        # skip if no possible matches found
        if soup.find("table") == None:
            print(f"Error scraping {fighter_name} with {headers['User-Agent']}")
            return pd.DataFrame()

        link_url = find_fighter_odds_url_in_html(soup, fighter_name, odds_urls_overide_df)

        # Check if a valid output has been found or if we have gone directly to the fighters odds page
        if link_url == "" and (fighter_name not in soup.title.string):
            print(f"Error scraping {fighter_name}: {e}")
            return pd.DataFrame()

        # Navigate to the website if we're not already there
        if fighter_name not in soup.title.string:
            # Navigate to the website
            soup = get_website_soup(
                url=link_url,
                fast=fast,
                playwright_context=playwright_context,
                # headers = headers, # Only needed for fast
                cookies=cookies_dict, # Only needed for fast
                chrome_binary=config["chrome_binary_location"], # Only needed for NOT fast
                browser=browser
            )

        if playwright:
            # Close the browser
            playwright_browser.close()
            
        # pdb.set_trace()
                    
        # Find the table in the soup object
        table = soup.find("table")

        # Convert the table to a data frame and add the rows to the results
        odds_df = betfightodds_table_to_df(table, fighter_name)
        
        # See if we want to return odds data or event urls
        if event_urls:
            # Collect the event urls
            # Extract all event URLs and their corresponding dates efficiently
            events = [(a["href"], a.next_sibling.strip()) for a in soup.select('a[href^="/events/"]') if a.next_sibling]
            
            # Convert to DataFrame
            events_df = pd.DataFrame(events, columns=["event_url", "event_date"])
            
            # Convert date
            events_df["event_date"] = betfightodds_dates_to_datetime(events_df["event_date"])
            
        else:
            events_df = pd.DataFrame()
    
    except Exception as e:
        print(f"Error scraping {fighter_name}: {e}")
        return pd.DataFrame(), pd.DataFrame()

    return odds_df, events_df

# # This function takes the data frames for the event odds urls and event odds data along with a new data frame of fighters



# def update_event_urls_and_odds(existing_event_urls, existing_event_odds, new_fighter_list, config):

#     pdb.set_trace()    

#     # Read the fighter odds urls
#     fighter_odds_urls = pd.read_csv(config['fighter_odds_urls'])    

#     # Get all the event urls for full odds
#     event_urls_odds = parallel_scrape_odds_fast(
#         new_fighter_list, 
#         num_threads = config['num_threads'], 
#         firefox_location = config['firefox_location'], 
#         gecko_driver = config['gecko_driver_location'],
#         odds_urls_overide_df = fighter_odds_urls,
#         playwright = False,
#         event_urls_only = True,
#         fast = True)

#     # Get rid of duplicate events and keep the most common date for the event
#     event_urls_odds_agg = event_urls_odds.groupby('event_url', as_index=False)['event_date'].apply(lambda x: x.mode().iloc[0])

#     # Convert the date
#     event_urls_odds_agg['event_date'] = pd.to_datetime(event_urls_odds_agg['event_date'].str.replace(r'(\d+)(st|nd|rd|th)', r'\1', regex=True), format = '%b %d %Y')

#     # Scrape the odds of each event after 2017 (odds only really start to show after then)
#     event_odds = parallel_scrape_event_odds_fast(
#         event_urls_odds_agg[event_urls_odds_agg['event_date'] > '01-01-2018'], 
#         num_threads = config['num_threads'], 
#         firefox_location = config['firefox_location'], 
#         gecko_driver = './data/firefox_driver_32bit/geckodriver0_35.exe',
#         fast = False)

#     return new_fight_results_data, new_fight_stats_data

# This function takes existing odds data and a list of fighters and adds the new fights to the data. 
# It requires a list of dates for the events as well as the list of fighters that fought.

def update_odds_data(existing_odds_data, existing_event_urls, existing_event_odds, new_fighter_list, new_event_date_list, config, fast_scrape=True, num_threads=1):

    # pdb.set_trace()
    
    # Read the fighter odds urls
    fighter_odds_urls = pd.read_csv(config["fighter_odds_urls"])

    # Scrape all the odds data for the fighters in the list
    new_odds_data, new_events_urls = parallel_scrape_odds_fast(
        new_fighter_list, 
        num_threads=config["num_threads_odds_scrape"], # TO ADDRESS ISSUE WITH BESTFIGHTODDS CAPTCHA
        odds_urls_overide_df=fighter_odds_urls,
        event_urls=True,
        fast=fast_scrape)
        
    # Get the date list and expand it to include dates on either side due to timezone differences potentially causing discrepencies
    full_date_list = list(new_event_date_list.copy())
    for date in new_event_date_list:
        full_date_list.append(date + timedelta(days=1))
        full_date_list.append(date - timedelta(days=1))
        full_date_list.append(date + timedelta(days=2))
        full_date_list.append(date - timedelta(days=2))
        
    # Filter only the fights that are on the specified dates
    if len(new_odds_data) > 0:
        new_odds_data = new_odds_data[new_odds_data["date"].isin(full_date_list)]
    
        # Add on the date of the odds scrape
        new_odds_data["notformodel_date_fighter_odd_scrape"] = str(pd.to_datetime("now", utc=True)) # Give the datetime in GMT
        
        # Add this onto the existing odds data
        new_odds_data = pd.concat([new_odds_data, existing_odds_data])
    
        # Remove any duplicate records
        new_odds_data = new_odds_data.drop_duplicates(subset=["event", "date", "fighter", "opponent"], keep="first", inplace=False)
        
        ###############################
        # Deal with missing odds data
        ###############################
        
        # Filter rows with NaT in 'date'
        nat_date_rows = new_odds_data[new_odds_data["date"].isna()]
    
        # Total count of rows with NaT in 'date'
        total_nat_count = nat_date_rows.shape[0]
    
        # Count and percentage for rows where 'event' contains "UFC"
        ufc_nat_count = nat_date_rows[nat_date_rows["event"].str.contains("UFC", case=False, na=False)].shape[0]
        ufc_nat_percent = (ufc_nat_count / total_nat_count * 100) if total_nat_count > 0 else 0
    
        # Count and percentage for rows where 'event' does not contain "UFC"
        non_ufc_nat_count = nat_date_rows[~nat_date_rows["event"].str.contains("UFC", case=False, na=False)].shape[0]
        non_ufc_nat_percent = (non_ufc_nat_count / total_nat_count * 100) if total_nat_count > 0 else 0
    
        print(f"Rows with NaT in 'date' and 'event' containing 'UFC': {ufc_nat_count} ({ufc_nat_percent:.2f}%)")
        print(f"Rows with NaT in 'date' and 'event' not containing 'UFC': {non_ufc_nat_count} ({non_ufc_nat_percent:.2f}%)")
    
    else:
        # The scrape was unsuccessful
        new_odds_data = existing_odds_data
    
    ###############################
    ### Update events data
    ###############################
    
    # if len(new_events_urls) > 0:
    #     # Get a list of events that haven't already been parsed
    #     new_events_to_scrape = new_events_urls[(~new_events_urls['event_url'].isin(existing_event_urls['event_url'].tolist())) | (new_events_urls['event_date'] >= date.today())]
            
    #     # Scrape the new event odds
    #     new_events_odds = parallel_scrape_event_odds_fast(
    #         new_events_to_scrape, 
    #         num_threads = config['num_threads'], 
    #         firefox_location = config['firefox_location'], 
    #         gecko_driver = config['gecko_driver_location'],
    #         fast = fast_scrape)
        
    #     # Concat and replace duplicates
    #     final_event_urls = pd.concat([new_events_to_scrape, existing_event_urls]).drop_duplicates()
    #     final_event_odds = pd.concat([new_events_odds, existing_event_odds]).drop_duplicates(['date', 'fighter', 'opponent'], keep = 'first')
    #     # Ensure format of final cols correct
    #     final_event_odds.loc[:, odds_scrape_config['events_colnames']] = final_event_odds[odds_scrape_config['events_colnames']].apply(pd.to_numeric, errors = 'coerce')
        
    # else:
        # The scrape was unsuccesful
    final_event_urls = existing_event_urls
    final_event_odds = existing_event_odds
        
    return new_odds_data, final_event_urls, final_event_odds


##########################################################################################
### oddschecker. Kept together rather than slotted in alphabetically with the bestfightodds ones ###
##########################################################################################

# Page titles that mean the site has blocked us rather than served the page. Checked against the title
# only, because the word cloudflare on its own appears in the real pages too
blocked_page_titles = ["just a moment", "attention required", "verifying you are human",
                       "checking your browser", "access denied"]

# The challenge page loads this, and the real pages do not
challenge_script = "challenges.cloudflare.com"

# The real pages run to several hundred thousand characters and a challenge page to a few thousand, so
# anything short is a block I haven't seen the wording of
min_page_characters = 50000

# Wait between pages so a card's worth of fights isn't hammered out in a few seconds
seconds_between_pages = 5

# The market to read odds from, as oddschecker names it
market_to_use = "Win Market"

def clean_bookie_name(name):

    return name.lower().replace(" ", "")


def check_page_source(url, page_source):

    """
    Stops with a message saying what came back if the page is a block or an error rather than the real
    thing. Without this a challenge page just turns into nothing found much later on.
    """

    title_match = re.search(r"<title>(.*?)</title>", page_source, re.S)
    title = title_match.group(1).strip() if title_match else ""

    print("  " + str(len(page_source)) + " characters, title: " + title)

    for marker in blocked_page_titles:
        if marker in title.lower():
            raise RuntimeError("Blocked by oddschecker on " + url + ', it returned "' + title
                               + '" instead of the page. Leave it a while before running again.')

    if challenge_script in page_source[:5000]:
        raise RuntimeError("Got a Cloudflare challenge on " + url
                           + ". Leave it a while before running again.")

    if len(page_source) < min_page_characters:
        raise RuntimeError("Only got " + str(len(page_source)) + " characters from " + url
                           + ', titled "' + title + '". Too short to be the real page.')

    if "404" in title:
        raise RuntimeError("Got a 404 page for " + url + ", so the url has probably moved")


def get_page_source(url):

    """
    Opens a page in a real browser, clears the Cloudflare check and returns the html. A browser per
    page is slow but it is the only path that gets through.
    """

    print("Fetching " + url)

    browser = start_browser(
        url,
        chrome_binary=config["chrome_binary_location"],
        headless=False)

    clear_captcha(browser)

    page_source = browser.get_page_source()

    browser.stop()

    check_page_source(url, page_source)

    time.sleep(seconds_between_pages)

    return page_source


def get_fight_urls(page_source):

    """
    Pulls the fight page urls off an oddschecker mma listing page. They look like
    /ufc-mma/ufc/ernesta-kareckaite-v-melissa-gatto/winner, so the fighter names are in the url itself.
    Returns a dict of the two cleaned fighter names to the full url.
    """

    urls = {}

    for path in set(re.findall(r'"(/ufc-mma/[a-z0-9-]+/[a-z0-9-]+-v-[a-z0-9-]+/winner)"', page_source)):

        slug = path.split("/")[-2]
        names = tuple(clean_name_for_odds_search(name) for name in slug.split("-v-"))

        urls[names] = "https://www.oddschecker.com" + path

    return urls


def find_fight_url(fight_urls, fighter_name):

    """
    Finds the fight url where one of the two fighters in the url matches the fighter being backed.
    Returns an empty string if the fighter isn't on the card.
    """

    cleaned_name = clean_name_for_odds_search(fighter_name)

    for names, url in fight_urls.items():
        if cleaned_name in names:
            return url

    return ""


def get_subevent_markets(page_source):

    """
    Pulls the odds data out of an oddschecker fight page. The page is server rendered React, so the
    whole thing sits in a json payload in a script tag rather than in the html itself.
    """

    payload = re.search(
        r'<script type="application/json" data-hypernova-key="subeventmarkets"[^>]*>(.*?)</script>',
        page_source,
        re.S)

    if payload is None:
        raise RuntimeError("No odds data in the page. The layout has changed, or the page is not a fight page")

    payload = payload.group(1).strip()

    # Hypernova wraps its payloads in an html comment
    payload = re.sub(r"-->$", "", re.sub(r"^<!--", "", payload)).strip()

    return json.loads(payload)["bestOdds"]


def get_fighter_odds(markets, fighter_name):

    """
    Returns a data frame of the odds for one fighter in the win market, one row per bookmaker, with
    the odds after commission. Commission only comes off exchange winnings, and which bookmakers are
    exchanges comes from oddschecker rather than from a list I keep.
    """

    # Find the win market rather than method of victory, round betting and the rest
    win_market_ids = [market["ocMarketId"] for market in markets["markets"]["entities"].values()
                      if market["marketTypeName"] == market_to_use]

    # Find the bet for this fighter in that market
    cleaned_name = clean_name_for_odds_search(fighter_name)

    bet_ids = [bet["ocBetId"] for bet in markets["bets"]["entities"].values()
               if bet["marketId"] in win_market_ids and clean_name_for_odds_search(bet["betName"]) == cleaned_name]

    wanted_bookies = [clean_bookie_name(bookie) for bookie in bet_config["bookies_to_use"]]

    odds_rows = []

    for bet_id in bet_ids:
        for bookie_code, price in markets["odds"].get(str(bet_id), {}).items():

            # oddschecker sometimes prices a bookmaker code that isn't in the bookmakers list on
            # the page, and there is then no name or type to look up. Say which one it was and
            # carry on, rather than losing the whole card over one missing bookie
            if bookie_code not in markets["bookmakers"]["entities"]:
                # print("No bookmaker found for code " + bookie_code + " when pricing " + fighter_name)
                continue

            bookie = markets["bookmakers"]["entities"][bookie_code]

            if clean_bookie_name(bookie["bookmakerName"]) not in wanted_bookies:
                continue

            if price["status"] != "ACTIVE" or price["oddsDecimal"] is None:
                continue

            # Take the commission off the winnings for an exchange, leaving the stake alone
            if bookie["bookmakerType"] == "exchange":
                effective_odds = 1 + (price["oddsDecimal"] - 1) * (1 - bet_config["exchange_commission"])
            else:
                effective_odds = price["oddsDecimal"]

            odds_rows.append({
                "bookie": bookie["bookmakerName"],
                "is_exchange": bookie["bookmakerType"] == "exchange",
                "odds": price["oddsDecimal"],
                "effective_odds": round(effective_odds, 3)
                })

    return pd.DataFrame(odds_rows)


def report_missing_bookies(markets):

    """
    Says which of the configured bookmakers oddschecker isn't offering, so a name that never matches
    doesn't just quietly produce no odds.
    """

    available = [clean_bookie_name(bookie["bookmakerName"]) for bookie in markets["bookmakers"]["entities"].values()]

    missing = [bookie for bookie in bet_config["bookies_to_use"] if clean_bookie_name(bookie) not in available]

    if len(missing) > 0:
        print("Not offered on this page: " + ", ".join(missing))

