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