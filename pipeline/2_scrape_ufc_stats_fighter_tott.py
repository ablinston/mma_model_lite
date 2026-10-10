
"""
Overview
this notebook parses all fighters' details and tale of the tape

scrape ufc fighters' details
includes first, last, nickname, url
from url scrape scrape fighter's tale of the tape, 
includes fighter, height, weight, reach, stance, dob
"""

# Import
import fastparquet as fp

from mma_functions.odds_scraping import get_fighter_data
from mma_functions.parsing import parse_fighter_details
from mma_functions.scraping import generate_alphabetical_urls, parallel_scrape
from mma_functions.config import config

# generate list of urls for fighter details
list_of_alphabetical_urls = generate_alphabetical_urls()

# Go through list of alphabetical urls
all_fighter_details_df = parallel_scrape(
    url_list=list_of_alphabetical_urls,
    parse_function=parse_fighter_details,
    num_threads=config["num_threads"],
    fighter_details_column_names=config["fighter_details_column_names"])

# write to file
fp.write("./data/" + config["fighter_details_file_name"], all_fighter_details_df, compression="GZIP")

# define list of urls of fighters to parse
list_of_fighter_urls = list(all_fighter_details_df["URL"])

# loop through list_of_fighter_urls
all_fighter_tott_df = parallel_scrape(
    url_list=list_of_fighter_urls,
    parse_function=get_fighter_data,
    num_threads=config["num_threads"],
    fighter_tott_column_names=config["fighter_tott_column_names"]    
)

# write to file
fp.write("./data/" + config["fighter_tott_file_name"], all_fighter_tott_df, compression="GZIP")
