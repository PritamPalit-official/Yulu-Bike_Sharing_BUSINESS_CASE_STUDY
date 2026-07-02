import unittest

def map_season(season_code):
    """Reflected logic: Maps numeric season code to name."""
    season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    return season_map.get(season_code, "Unknown")

def map_weather(weather_code):
    """Reflected logic: Maps numeric weather code to description."""
    weather_map = {
        1: "Clear / Few Clouds",
        2: "Mist / Cloudy",
        3: "Light Snow / Light Rain",
        4: "Heavy Rain / Ice Pallets"
    }
    return weather_map.get(weather_code, "Unknown")

class TestYuluPipeline(unittest.TestCase):
    def test_season_mapping(self):
        self.assertEqual(map_season(1), "Spring")
        self.assertEqual(map_season(4), "Winter")
        self.assertEqual(map_season(5), "Unknown")
        
    def test_weather_mapping(self):
        self.assertEqual(map_weather(1), "Clear / Few Clouds")
        self.assertEqual(map_weather(3), "Light Snow / Light Rain")
        self.assertEqual(map_weather(0), "Unknown")

if __name__ == '__main__':
    unittest.main()
