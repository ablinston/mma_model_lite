# This function gets the correct suffix for a day of the month
def get_day_suffix(day):
    # Determine the suffix for the day
    if 10 <= day % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    return suffix