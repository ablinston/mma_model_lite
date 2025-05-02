def betfair_flag_bz2_as_fight(bz2_path):
    
    """
    This function takes a path to a bz2 file and flags whether it's a fight or not, returning dictionary indicating 1 or 0
    
    """
    # pdb.set_trace()
    try:
        with bz2.open(bz2_path, "rt", encoding="utf-8") as f:
            for line in f:
                if "Fight Result" in line:
                    print("found")
                    return {'filepath': bz2_path, 'fight_flag': 1}
                    break
    except:
        print(f"File not able to be opened: {bz2_path}")
        # pdb.set_trace()
        
    return {'filepath': bz2_path, 'fight_flag': 0}
            