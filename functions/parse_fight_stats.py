def parse_fight_stats(soup):
    """
    parse full fight stats for both fighters from soup
    loop through soup to find all 'td' tags with the class 'b-fight-details__table-col'
    this returns a list of stats for both fighters in alternate order
    e.g. [0, 1, 2, 2, 20, 30] stats [0, 2, 20] belong to the first fighter and [1, 2, 30] belong to the second fighter
    use enumerate to add index to results
    stats with even indexes belongs to the first fighter and odd indexes belong to the second fighter
    clean each element in the list, removing '
' and ' ' 
    e.g cleans '
 fighter name 
' into 'fighter name' and  '
      19 of 32
    ' into '19 of 32'
    
    arguments:
    soup (html): output of get_soup() parser

    returns:
    two lists of fighter stats, one for each fighter
    """
    fighter_a_stats = []
    fighter_b_stats = []
    for tag in soup.find_all('td', class_='b-fight-details__table-col'):
        for index, p_text in enumerate(tag.find_all('p')):
            if index % 2 == 0:
                fighter_a_stats.append(p_text.text.strip())
            else:
                fighter_b_stats.append(p_text.text.strip())
    return fighter_a_stats, fighter_b_stats
