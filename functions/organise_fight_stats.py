def organise_fight_stats(stats_from_soup):
    """
    organise a list of raw stats extracted from soup
    each set of stats starts with the fighter's name, the function groups each set together into a list of lists by the fighter's name

    there are two different types of stats, totals and significant strikes
    Totals include KD, SIG.STR., SIG.STR. %, TOTAL STR., TD, TD %, SUB.ATT, REV., CTRL
    Significant Strikes include SIG.STR., SIG.STR. %, HEAD, BODY, LEG, DISTANCE, CLINCH, GROUND
    
    each type of stat has a summary of total stats for the fight, and individual round stats
    the sets of stats are returned as a list of lists
    e.g. [[totals - summary], [totals - round 1], [totals - round n]..., [significant strikes - summary], [significant strikes - round 1], [significant strikes - round n]...] 

    arguments:
    stats_from_soup (list): a list of fight stats from parse_fight_stats()

    returns: 
    a list of lists of fight stats
    """
    fighter_stats_clean = []
    for name, stats in itertools.groupby(stats_from_soup, lambda x: x ==
        stats_from_soup[0]):
        if name:
            fighter_stats_clean.append([])
        fighter_stats_clean[-1].extend(stats)
    return fighter_stats_clean
