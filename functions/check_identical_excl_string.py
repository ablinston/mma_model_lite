
def check_identical_excl_string(list1, list2, string1, string2):
    """
    This function checks whether two lists are identical once strings are removed from each

    arguments:

    """
    list1 = [element.replace(string1, '') for element in list1]
    list2 = [element.replace(string2, '') for element in list2]

    return list1 == list2

