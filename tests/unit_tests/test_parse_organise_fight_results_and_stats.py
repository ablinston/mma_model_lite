# Load all the packages and functions needed
import fastparquet as fp

from mma_functions.parsing import parse_organise_fight_results_and_stats
from mma_functions.scraping import get_soup
from mma_functions.config import config

# Load what answers are expected
parse_organise_fight_results_correct = fp.ParquetFile("./tests/data/parse_organise_fight_results_correct.parquet").to_pandas()
parse_organise_fight_stats_correct = fp.ParquetFile("./tests/data/parse_organise_fight_stats_correct.parquet").to_pandas()

# This test looks at a specific URL for an historic fight and checks that the results  and stats match what is expected
def test_parse_organise_fight_results_and_stats():
    
    # Define a fight to test out
    url = "http://ufcstats.com/fight-details/f1613b372748c8bb"
    
    fight_results, fight_stats = parse_organise_fight_results_and_stats(
        get_soup(url),
        url,
        config["fight_results_column_names"],
        config["totals_column_names"],
        config["significant_strikes_column_names"])
    
    assert (fight_results.equals(parse_organise_fight_results_correct)) & (fight_stats.equals(parse_organise_fight_stats_correct)) == True
    
