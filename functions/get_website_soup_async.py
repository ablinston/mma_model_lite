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
