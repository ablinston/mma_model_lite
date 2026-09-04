# Load all the packages and functions needed
exec(open('global.py').read())

# Read the fighter odds urls
fighter_odds_urls = pd.read_csv(config['fighter_odds_urls'])

# Load what answers are expected
scrape_fighter_odds_data_correct = fp.ParquetFile('./tests/data/scrape_fighter_odds_data_correct.parquet').to_pandas()

# This test looks at a specific URL for an historic event and checks that the results match what is expected
def test_scrape_fighter_odds():
    
    # Start the driver
    options = uc.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--start-maximized")
        
    if os.name != "nt":
        driver = uc.Chrome(options = options, headless = False, driver_executable_path = config['chrome_driver_location_pi'])
    else:
        driver = uc.Chrome(options = options, headless = False, browser_executable_path = config['chrome_binary_location'], version_main = config.get('chrome_version_main'))

    driver.get("https://www.bestfightodds.com")
    
    # Allow captcha solve
    time.sleep(20)
    
    result1, result2 = scrape_fighter_odds_fast(
        'Georges St-Pierre', 
        cookies_dict = {}, 
        odds_urls_overide_df = pd.DataFrame({'fighter':[]}), 
        firefox_binary = config['firefox_location'],
        gecko_driver = config['gecko_driver_location'], 
        fast = False,
        driver = driver)
    
    driver.quit()
    
    assert result1.equals(scrape_fighter_odds_data_correct) == True
    
# te = scrape_fighter_odds_fast('Georges St-Pierre', cookies_dict = {}, odds_urls_overide_df = fighter_odds_urls, fast = False)
