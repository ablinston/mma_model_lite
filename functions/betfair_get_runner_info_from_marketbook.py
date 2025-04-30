def betfair_get_runner_info_from_marketcatalogue(marketcatalogue):
    
    """
    This function takes a marketbook and extracts the information market name and runner ids with runner names
    
    Returns a dictionary of selection_id and runner names
    
    """
    # pdb.set_trace()
    
    # Initiate a runner dictionary
    runner_dict = {}
    
    for runner in marketcatalogue.runners:
        runner_dict.update({runner.selection_id: runner.runner_name})
    
    return runner_dict