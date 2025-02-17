# This function gets the response soup from a website either using selenium or a fast response

def get_website_soup(
        url, 
        fast,
        playwright_context = None,
        headers = {}, # Only needed for fast
        cookies = {}, # Only needed for fast
        firefox_binary = '', # Only needed for NOT fast
        gecko_driver = '' # Only needed for NOT fast):
    ):
    
    
    if fast == True:
        
        response = requests.get(url, headers=headers, cookies=cookies)
        
        # Check if been blocked by anti bot pretection
        if response.status_code in [403, 429]:
            print("Blocked! Try changing User-Agent, IP, or waiting.")
            print(f"Using user agent: {headers['User-Agent']}")
            
            return ""
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
    # Else, if we have a playwright context, we use that
    elif playwright_context is not None:
                
        # Get a new page
        page = playwright_context.new_page()
        page.goto(url, wait_until="domcontentloaded")
        
        soup = BeautifulSoup(page.content(), 'html.parser')
        page.close()
        
    # Otherwise, use selenium, which is the slowest
    else:
        
        ###########
        # Setup headless Firefox for getting odds information
        options = Options()
        options.headless = True
        options.accept_insecure_certs = True
        # Create a Service object with the path to Firefox driver
        options.binary_location = firefox_binary # firefox_binary
        service = Service(gecko_driver)
        # Create a WebDriver instance
        driver = webdriver.Firefox(service=service, options=options)
        ###########
    
        # Use selenium to navigate to the page
        driver.get(url)
        
        # Extract the page source and convert to xml
        soup = BeautifulSoup(driver.page_source, 'lxml')
        
        driver.quit()
        
    return soup
