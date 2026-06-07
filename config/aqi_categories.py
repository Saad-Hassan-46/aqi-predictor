AQI_CATEGORIES = [
    {"label": "Good",            "min": 0,   "max": 50,  "color": "#00E400", "advice": "Air quality is satisfactory."},
    {"label": "Moderate",        "min": 51,  "max": 100, "color": "#FFFF00", "advice": "Unusually sensitive people should consider reducing prolonged outdoor exertion."},
    {"label": "Unhealthy (SG)",  "min": 101, "max": 150, "color": "#FF7E00", "advice": "Sensitive groups should reduce outdoor activity."},
    {"label": "Unhealthy",       "min": 151, "max": 200, "color": "#FF0000", "advice": "Everyone may begin to experience health effects."},
    {"label": "Very Unhealthy",  "min": 201, "max": 300, "color": "#8F3F97", "advice": "Health alert: everyone may experience more serious effects."},
    {"label": "Hazardous",       "min": 301, "max": 500, "color": "#7E0023", "advice": "Health warning of emergency conditions."},
]

def get_aqi_category(aqi: float) -> dict:
    for cat in AQI_CATEGORIES:
        if cat["min"] <= aqi <= cat["max"]:
            return cat
    return AQI_CATEGORIES[-1]