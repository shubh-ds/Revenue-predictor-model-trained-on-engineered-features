# Import libraries
import numpy as np
import pandas as pd
import os
import yaml
import logging

# Logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

# Set console logger
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Set file logger
file_handler = logging.FileHandler('data_preprocessing_errors.log')
file_handler.setLevel('ERROR')

# Set format for logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to loggers
logger.addHandler(console_handler)
logger.addHandler(file_handler)
print()
logger.debug('------------------------------------------------ DATA PREPROCESSING STARTED ------------------------------------------------')


def basic_assess_data(name: str, df: pd.DataFrame) -> None:
    '''Performs basic data assesment for a given dataframe'''
    try:
        # Size
        logger.debug(f'Size of %s df is %s', name, df.shape)

        # Check missing values
        logger.debug(f'Number of missing values in %s df is %s', name, df.isnull().sum().sum())

        # Duplicate entries
        logger.debug('Number of duplicate entries in %s df: %s ', name, df.duplicated().sum().sum())

    except Exception as e:
        logger.error('Basic assesment of individual data cannot be performed. Something went wrong: %s', e)


def process_order_numbers_df(df: pd.DataFrame) -> pd.DataFrame:
    '''Processes order_numbers df'''
    
    # Handle duplicate date entries in order_numbers
    logger.debug('Handle duplicate dates in order_numbers')
    duplicate_date_processed_df = handle_duplicate_dates(df)
    print()

    # Handle non-monotonic points in order_numbers
    logger.debug('Remove bad points from order_numbers: Orders cannot decrease with time')
    monotonic_points_df = remove_bad_points(duplicate_date_processed_df)

    # Normalize order growth by time
    logger.debug('Normalize order growth by time')
    order_growth_normalized_df = normalize_order_growth(monotonic_points_df)

    # Treat outliers
    logger.debug('Treat outliers in normalized order growth')
    outlier_treated_df = treat_outliers_in_normalized_order_growth(order_growth_normalized_df)
    logger.debug('Updated shape of order_numbers df is %s', outlier_treated_df.shape)

    # Expand order numbers into a daily series
    order_daily, order_daily_win = expand_order_numbers_in_daily_series(outlier_treated_df)

    return order_daily, order_daily_win


def expand_order_numbers_in_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    '''Expands order numers in a daily series'''
    df = df.copy()

    df = df.reset_index(drop=True)

    pieces = []
    pieces_win = []

    for i in range(1, len(df)):
        start = df.loc[i - 1, 'date']
        end = df.loc[i, 'date']
        rng = pd.date_range(start + pd.Timedelta(days=1), end, freq='D')

        if len(rng) == 0:
            continue

        pieces.append(pd.DataFrame({
            'date': rng,
            'estimated_orders_from_order_per_day': df.loc[i, 'orders_per_day']
        }))

        pieces_win.append(pd.DataFrame({
            'date': rng,
            'est_orders_from_orders_per_day_capped': df.loc[i, 'orders_per_day_capped']
        }))

    order_daily = pd.concat(pieces, ignore_index=True)
    order_daily_win = pd.concat(pieces_win, ignore_index=True)

    logger.debug('Shape of order_daily is %s', order_daily.shape)
    logger.debug('Shape of order_daily_win is %s', order_daily_win.shape)

    return order_daily, order_daily_win


def treat_outliers_in_normalized_order_growth(df: pd.DataFrame) -> pd.DataFrame:
    '''Treats outliers in normalized order growth'''

    q1 = df['orders_per_day'].quantile(0.25)
    q3 = df['orders_per_day'].quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr

    df['orders_per_day_capped'] = df['orders_per_day'].clip(upper=upper_fence)

    return df


def normalize_order_growth(df: pd.DataFrame) -> pd.DataFrame:
    '''Normalizes data growth by time in order_numbers df'''
    df = df.copy()
    # Calculate days gap in df
    df['days_gap'] = df['date'].diff().dt.days

    # Calculate order number difference
    df['order_diff'] = df['order_number'].diff()

    # Calculate orders per day
    df['orders_per_day'] = df['order_diff'] / df['days_gap']

    return df


def load_data(name: str, path: str) -> pd.DataFrame:
    '''Loads data from a given path'''
    try:
        df = pd.read_csv(path)
        logger.debug(f'%s file successfully loaded', name)
        return df
    except Exception as e:
        logger.error(f'Error while loading %s file', name)
        raise


def handle_duplicate_dates(df: pd.DataFrame) -> pd.DataFrame:
    '''Handles duplicate date entries in order_numbers df'''
    try:
        # Sort dates in ascending order
        df = df.sort_values('date')

        # Group by dates and take the maximum order number available as order number
        df = df.groupby('date', as_index=False)['order_number'].max()

        # Updated shape
        logger.debug('Updated shape of order_numbers df is %s', df.shape)

        return df
    except Exception as e:
        logger.error('Something went wrong while handling duplicate date entries in order_numbers df: %s', e)


def remove_bad_points(df: pd.DataFrame) -> pd.DataFrame:
    '''Removes Non-monotonic points from order_numbers df'''
    df = df[df['order_number'].cummax() == df['order_number']]
    logger.debug('Updated shape of order_numbers df: %s', df.shape)
    return df


def main():
    try:
        # Load individual files
        order_numbers = load_data('order_numbers', 'data/raw/order_numbers.csv')
        order_numbers['date'] = pd.to_datetime(order_numbers['date'])

        reported_data = load_data('reported_data', 'data/raw/reported_data.csv')
        reported_data['start_date'] = pd.to_datetime(reported_data['start_date'])
        reported_data['end_date'] = pd.to_datetime(reported_data['end_date'])

        transaction_data = load_data('transaction_data', 'data/raw/transaction_data.csv')
        transaction_data['date'] = pd.to_datetime(transaction_data['date'])

        # Basic data assessment
        print()
        basic_assess_data('order numbers', order_numbers)
        print()
        basic_assess_data('reported_data', reported_data)
        print()
        basic_assess_data('transaction_data', transaction_data)
        logger.debug('Basic data assessment completed')
        print()
        
        # Process  order_numbers df specifically
        logger.debug('Processing order_numbers df separately')
        order_daily, order_daily_win = process_order_numbers_df(order_numbers)

        # Save order_daily, order_daily_win to interim data
        try:
            os.makedirs(os.path.dirname(os.path.join("./data", "interim")), exist_ok=True)
            order_daily.to_csv('./data/interim/order_daily.csv', index=False)
            order_daily_win.to_csv('./data/interim/order_daily_win.csv', index=False)
        except Exception as e:
            logger.error('Some error happened while saving interim files: %s', e)
            
        logger.debug('Order numbers df processed successfully')
        logger.debug('Data pre-processing completed !')

    except Exception as e:
        logger.error(f'Failed to perform basic assesment of individual files. Something went wrong: %s', e)
        raise


if __name__ == '__main__':
    main()


