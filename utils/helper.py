from datetime import datetime

def convert_to_timestamp(year, month, day, weekday, hour, minues=0, flag='Weather'):
        # Assuming a fixed year, as it's not provided in your data
        minues_convert = {0: 0, 1: 15, 2: 30, 3: 45}
        minues_convert_weather = {0: 0, 1: 10, 2: 20, 3: 30, 4:40, 5: 50}
        year = int(year.item())
        month = int(month.item())
        day = int(day.item())
        weekday = int(weekday.item())
        hour = int(hour.item())
        if type(minues) != int:
            minues = int(minues.item())
            # print(minues)
            if flag == 'Weather':
                minues = minues_convert_weather[minues]
            else:
                minues = minues_convert[minues]
        # print('minues:', minues)
        if not (1 <= month <= 12):
            raise ValueError(f'Invalid month value: {month}')
        # if minues is None:
        #     res = datetime(year, month, day, hour).strftime('%Y-%m-%d %H:%M:%S')
        # else:
        res = datetime(year, month, day, hour, minues).strftime('%Y-%m-%d %H:%M:%S')
        # print("res: ", res)
        return res
