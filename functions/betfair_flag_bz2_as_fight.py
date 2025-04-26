def betfair_flag_bz2_as_fight(bz2_path):
    
    """
    This function takes a path to a bz2 file and flags whether it's a fight or not, returning dictionary indicating 1 or 0
    
    """
    
    with bz2.open(bz2_path, "rt", encoding="utf-8") as f:
        for line in f:
            if "Fight Result" in line:
                return {'filepath': bz2_path, 'fight_flag': 1}
                break
        
    return {'filepath': bz2_path, 'fight_flag': 0}
            