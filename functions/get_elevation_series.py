# This function gets the elevation of a location

import requests

def get_elevation_series(locations, api_key):
    """
    Fetch the elevation above sea level for a pandas Series of locations using Google Maps Elevation API.

    Parameters:
        locations (pd.Series): A pandas Series containing location strings.
        api_key (str): Your Google Maps API key.

    Returns:
        pd.Series: A pandas Series containing elevations in meters or None if unavailable.
    """
    elevation_cache = {}  # Cache to store elevations for unique locations

    def fetch_elevation(location):
        if location in elevation_cache:
            return elevation_cache[location]  # Return cached elevation

        try:
            # Geocode the location to get coordinates
            geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json"
            geocode_params = {'address': location, 'key': api_key}
            geocode_response = requests.get(geocode_url, params=geocode_params, timeout=10).json()

            if geocode_response['status'] != 'OK':
                elevation_cache[location] = None
                return None  # Location not found

            # Get latitude and longitude
            lat = geocode_response['results'][0]['geometry']['location']['lat']
            lon = geocode_response['results'][0]['geometry']['location']['lng']

            # Fetch elevation using Google Maps Elevation API
            elevation_url = "https://maps.googleapis.com/maps/api/elevation/json"
            elevation_params = {'locations': f"{lat},{lon}", 'key': api_key}
            elevation_response = requests.get(elevation_url, params=elevation_params, timeout=10).json()

            if elevation_response['status'] == 'OK' and 'results' in elevation_response:
                elevation = elevation_response['results'][0]['elevation']
                elevation_cache[location] = elevation  # Cache the result
                return elevation

            elevation_cache[location] = None
            return None  # Elevation data not available

        except Exception as e:
            elevation_cache[location] = None
            return None  # Handle errors gracefully

    return locations.apply(fetch_elevation)
