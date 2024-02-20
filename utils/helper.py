from datetime import datetime

def convert_to_timestamp(year, month, day, weekday, hour, minues=0, flag='Weather'):
    # Convert to integers if inputs are not already integers
    year = int(year)
    month = int(month)
    day = int(day)
    weekday = int(weekday)
    hour = int(hour)
    minues = int(minues)

    # The rest of your logic for converting minutes based on the 'flag'
    minues_convert = {0: 0, 1: 15, 2: 30, 3: 45}
    minues_convert_weather = {0: 0, 1: 10, 2: 20, 3: 30, 4:40, 5: 50}
    if flag == 'Weather':
        minues = minues_convert_weather.get(minues, minues)
    else:
        minues = minues_convert.get(minues, minues)

    # Now create the timestamp
    timestamp = datetime(year, month, day, hour, minues).strftime('%Y-%m-%d %H:%M:%S')
    return timestamp
