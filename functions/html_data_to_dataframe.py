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
