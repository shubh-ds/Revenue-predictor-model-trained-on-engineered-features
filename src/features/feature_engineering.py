# Import libraries
import numpy as np
import pandas as pd
import os
import yaml
import logging

# Logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

# Set console logger
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Set file logger
file_handler = logging.FileHandler('feature_engineering_errors.log')
file_handler.setLevel('ERROR')

# Set format for logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to loggers
logger.addHandler(console_handler)
logger.addHandler(file_handler)
print()
logger.debug('------------------------------------------- FEATURE ENGINEERING STARTED -------------------------------------------------------')

def load_data(path: str) -> pd.DataFrame:
    '''Loads a file from a given path'''
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.error('Unexpected error occured while loading file: %s', e)

def engineer_features(df_transaction_data: pd.DataFrame, order_daily: pd.DataFrame, order_daily_win: pd.DataFrame) -> pd.DataFrame:
    '''Returns a dataframe with engineered features from available data'''

    try:
        # Create dataframe covering all days between min and max transaction dates
        full_days = pd.DataFrame({'date': pd.date_range(df_transaction_data['date'].min(), df_transaction_data['date'].max(), freq='D')})

        # Merge with daily order df
        order_daily = full_days.merge(order_daily, on='date', how='left')

        # Fill missing order estimate values with nearest available values
        order_daily['estimated_orders_from_order_per_day'] = order_daily['estimated_orders_from_order_per_day'].ffill().bfill()

        # Merge with capped version of daily orders
        order_daily_win = full_days.merge(order_daily_win, on='date', how='left')
        # Fill missing values
        order_daily_win['est_orders_from_orders_per_day_capped'] = order_daily_win['est_orders_from_orders_per_day_capped'].ffill().bfill()

        # Merge the uncapped and capped daily order into the main transaction dataset
        df_transaction_data = df_transaction_data.merge(order_daily, on='date', how='left')
        df_transaction_data = df_transaction_data.merge(order_daily_win, on='date', how='left')

        # Create a per-user spend feature normailising spend for changes in the size of the active user base.
        df_transaction_data['spend_per_user'] = df_transaction_data['total_spend_index'] / df_transaction_data['weekly_active_users_index']

        # Create a per-user orders feature
        df_transaction_data['gross_orders_per_user'] = df_transaction_data['gross_orders_index'] / df_transaction_data['weekly_active_users_index']

        # Create an average-order-value-style feature
        df_transaction_data['aov_index'] = df_transaction_data['total_spend_index'] / df_transaction_data['gross_orders_index']

        return df_transaction_data

    except Exception as e:
        logger.error('Unexpected error occurred while engineering features: %s', e)


def aggregate_data(df_transaction_data: pd.DataFrame, df_reported_data: pd.DataFrame) -> pd.DataFrame:
    '''Takes daily transaction data as input and returns aggregated data as per reported periods'''

    try:
        rows = []

        # Loop through each row in the reported quarterly revenue table
        for i, r in df_reported_data.iterrows():
            # Filter the daily transaction data to the exact dates covered by the current reported period.
            m = df_transaction_data[(df_transaction_data['date'] >= r['start_date']) & (df_transaction_data['date'] <= r['end_date'])].copy()
            
            # Aggregate daily features into quarterly-period features
            rows.append({
                'period': r['period'],
                'start_date': r['start_date'],
                'end_date': r['end_date'],
                'days_in_period': len(m),
                'revenue_index': r['revenue_index'],
                'sum_total_spend_index': m['total_spend_index'].sum(),
                'sum_gross_orders_index': m['gross_orders_index'].sum(),
                'sum_spend_per_user': m['spend_per_user'].sum(),
                'sum_orders_per_user': m['gross_orders_per_user'].sum(),
                'sum_est_orders_from_order_per_day': m['estimated_orders_from_order_per_day'].sum(),
                'sum_est_orders_from_order_per_day_capped': m['est_orders_from_orders_per_day_capped'].sum(),
                'avg_weekly_active_users_index': m['weekly_active_users_index'].mean(),
                'avg_aov_index': m['aov_index'].mean(),
                'avg_est_orders_per_day': m['estimated_orders_from_order_per_day'].mean(),
                'avg_est_orders_per_day_capped': m['est_orders_from_orders_per_day_capped'].mean(),
            })

        # Convert into a DataFrame.
        quaterly_df = pd.DataFrame(rows)

        return quaterly_df

    except Exception as e:
        logger.error('Unexpected error occured while aggregating the data: %s', e)



def main():
    try:
        # Load required data
        df_reported_data = load_data('./data/raw/reported_data.csv')
        df_reported_data['start_date'] = pd.to_datetime(df_reported_data['start_date'])
        df_reported_data['end_date'] = pd.to_datetime(df_reported_data['end_date'])

        df_transaction_data = load_data('./data/raw/transaction_data.csv')
        df_transaction_data['date'] = pd.to_datetime(df_transaction_data['date'])
        
        order_daily = load_data('./data/interim/order_daily.csv')
        order_daily['date'] = pd.to_datetime(order_daily['date'])

        order_daily_win = load_data('./data/interim/order_daily_win.csv')
        order_daily_win['date'] = pd.to_datetime(order_daily_win['date'])

        logger.debug('Files loaded successfully for feature engineering')

        # Engineer features
        df_transaction_data = engineer_features(df_transaction_data, order_daily, order_daily_win)

        # Aggregate daily reported data to quarterly reported periods
        engineered_data = aggregate_data(df_transaction_data, df_reported_data)

        # Save engineered data
        try:
            os.makedirs(os.path.dirname(os.path.join("./data", "processed")), exist_ok=True)
            engineered_data.to_csv('./data/processed/engineered_data.csv', index=False)
            
        except Exception as e:
            logger.error('Some error happened while saving engineered data file: %s', e)

        logger.debug('Feature engineering successfully completed !')

    except Exception as e:
        logger.error('Unexpected error occured while doing feature engineering: %s', e)


if __name__ == '__main__':
    main()