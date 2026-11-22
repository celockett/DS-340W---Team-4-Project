import requests
def get_current_weather():

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": 40.7982,
        "longitude": -77.8599,
        "current": [
            "temperature_2m",
            "precipitation",
            "rain",
            "showers",
            "snowfall",
            "weather_code"
        ],
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York"
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    return response.json()["current"]

def get_weather_description(weather_code):
    """Convert Open-Meteo weather codes into readable descriptions."""

    weather_conditions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Freezing fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Thunderstorm with heavy hail"
    }

    return weather_conditions.get(weather_code, "Unknown conditions")

def check_weather_warnings(weather, route_surfaces=None):
    warnings = []

    if weather["rain"] > 0 or weather["showers"] > 0:
        if route_surfaces and any("brick" in surface.lower() for surface in route_surfaces):
            warnings.append(
                "Rain is currently occurring. Your selected route includes "
                "brick surfaces, which may become slippery when wet."
            )
    if weather["snowfall"] > 0:
        warnings.append(
            "Snowfall is currently occurring. Sidewalks may become "
            "snow-covered or difficult to navigate."
        )

    if weather["temperature_2m"] <= 32 and weather["precipitation"] > 0:
        warnings.append(
            "Temperatures are at or below freezing with precipitation. "
            "Some sidewalks may be icy."
        )

    return warnings

if __name__ == "__main__":
    # weather = get_current_weather()

    # description = get_weather_description(weather["weather_code"])

    # print(f"Current weather: {description}")
    # print(f"Temperature: {weather['temperature_2m']}°F")
    # print(f"Rain: {weather['rain']} mm")
    # print(f"Snowfall: {weather['snowfall']} cm")

    # weather = get_current_weather()

    weather = get_current_weather()
    weather["rain"] = 1.0
    weather["showers"] = 0.0

    test_weather = weather.copy()
    test_weather["rain"] = 1.5

    print("Testing simulated rainy weather...")

    warnings = check_weather_warnings(test_weather)

    for warning in warnings:
        print(f"WARNING: {warning}")
