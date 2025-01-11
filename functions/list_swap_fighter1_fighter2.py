# This function takes a list and swaps 'fighter1' and 'fighter2' around.

def list_swap_fighter1_fighter2(swap_list):
    
    new_list = pd.Series(swap_list).str.replace('fighter2', 'fighterX')
    new_list = new_list.str.replace('fighter1', 'fighter2')
    new_list = new_list.str.replace('fighterX', 'fighter1')
    
    return new_list.tolist()

