def parse_fight_results(soup):
    """
    parase fight results from soup
    results include event, bout, outcome weightclass, method, round, time, timeformat, referee, details
    clean each element in the list, removing '
' and ' ' 
    e.g cleans '
      Welterweight Bout
' into 'Welterweight Bout'
    details include description of finish or judges and scores
    judges and scores also include details of point deduction
    e.g. 'Point Deducted: Illegal Knee by Menne Tony Weeks 45 - 49.Doug Crosby 42 - 49.Jeff Mullen 44 - 49.'
    return fight results as a list

    arguments:
    soup (html): output of get_soup() parser

    returns:
    a list of fight results
    """
    fight_results = []
    fight_results.append(soup.find('h2', class_='b-content__title').text)
    for tag in soup.find_all('a', class_='b-link b-fight-details__person-link'
        ):
        fight_results.append(tag.text)
    for tag in soup.find_all('div', class_='b-fight-details__person'):
        for i_text in tag.find_all('i'):
            fight_results.append(i_text.text)
    fight_results.append(soup.find('div', class_=
        'b-fight-details__fight-head').text)
    fight_results.append(soup.find('i', class_=
        'b-fight-details__text-item_first').text)
    remaining_results = soup.find_all('p', class_='b-fight-details__text')
    for tag in remaining_results[0].find_all('i', class_=
        'b-fight-details__text-item'):
        fight_results.append(tag.text.strip())
    fight_results.append(remaining_results[1].get_text())
    fight_results = [text.replace('\n', '').replace('  ', '') for text in
        fight_results]
    return fight_results
