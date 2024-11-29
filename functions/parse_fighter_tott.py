def parse_fighter_tott(soup):
    """
    parse fighter tale of the tape from soup
    fighter details contain fighter, height, weight, reach, stance, dob
    clean each element in the list, removing '
' and ' ' 
    e.g cleans '
      Jose Aldo
' into 'Jose Aldo'
    returns a list of fighter tale of the tape

    arguments:
    soup (html): output of get_soup() parser

    returns:
    a list of fighter tale of the tape
    """
    fighter_tott = []
    fighter_name = soup.find('span', class_='b-content__title-highlight').text
    fighter_tott.append('Fighter:' + fighter_name)
    tott = soup.find_all('ul', class_='b-list__box-list')[0]
    for tag in tott.find_all('i'):
        fighter_tott.append(tag.text + tag.next_sibling)
    fighter_tott = [text.replace('\n', '').replace('  ', '') for text in
        fighter_tott]
    return fighter_tott
