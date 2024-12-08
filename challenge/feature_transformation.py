import pandas as pd
import numpy as np
from datetime import datetime


def is_high_season(date):
    date_year = int(date.split('-')[0])
    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

    ranges = [
        ('15-Dec', '31-Dec'),
        ('1-Jan', '3-Mar'),
        ('15-Jul', '31-Jul'),
        ('11-Sep', '30-Sep')
    ]

    for start_date, end_date in ranges:
        range_min = datetime.strptime(
            start_date, '%d-%b').replace(year=date_year)
        range_max = datetime.strptime(
            end_date, '%d-%b').replace(year=date_year)

        if range_min <= date <= range_max:
            return 1

    return 0


def get_period_day(date):
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()

    periods = [
        ('mañana', datetime.strptime("05:00", '%H:%M').time(),
         datetime.strptime("11:59", '%H:%M').time()),
        ('tarde', datetime.strptime("12:00", '%H:%M').time(),
         datetime.strptime("18:59", '%H:%M').time()),
        ('noche', datetime.strptime("19:00", '%H:%M').time(),
         datetime.strptime("23:59", '%H:%M').time()),
        ('noche', datetime.strptime("00:00", '%H:%M').time(),
         datetime.strptime("04:59", '%H:%M').time())
    ]
    # to review 2 cats with same name
    for period, start_time, end_time in periods:
        if start_time <= date_time <= end_time:
            return period

    return None


def get_min_diff(data: pd.DataFrame) -> np.Series:
    try:
        operation_date = datetime.strptime(
            data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        scheduled_date = datetime.strptime(
            data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((operation_date - scheduled_date).total_seconds()) / 60
    except ValueError:
        min_diff = np.nan
    return min_diff


def create_target(target_input_data: pd.DataFrame, threshold_in_minutes: int, 
                  target_required_columns: list) -> np.Series:
    
    if target_required_columns not in target_input_data:
        raise ValueError()
    target_input_data['min_diff'] = target_input_data.apply(
        get_min_diff, axis=1)
    return np.where(target_input_data['min_diff'] > threshold_in_minutes, 1, 0)
