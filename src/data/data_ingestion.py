# Import libraries
import numpy as np
import pandas as pd
import os
import yaml
import logging

# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

# Set console logger
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Set file logger
file_handler = logging.FileHandler('data_ingestion_errors.log')
file_handler.setLevel('ERROR')

# Set format for logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to loggers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

print()
logger.debug('------------------------------------------- DATA INGESTION STARTED -------------------------------------------------------')

# Load data
def load_data(path: str, sheetname: str) -> pd.DataFrame:
    '''Load sheets from an excel file'''
    try:
        df = pd.read_excel(path , sheet_name=sheetname)
        return df
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

# Save data
def save_data(data: pd.DataFrame, name: str, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        data.to_csv(os.path.join(raw_data_path, f"{name}.csv"), index=False)
        logger.debug(f'{name} data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise
    

def main():
    try:
        path = 'data/raw/data_task.xlsx'
        order_numbers_df = load_data(path, 'order_numbers')
        transaction_data_df = load_data(path, 'transaction_data')
        reported_data_df = load_data(path, 'reported_data')

        save_data(order_numbers_df, 'order_numbers', './data')
        save_data(transaction_data_df, 'transaction_data', './data')
        save_data(reported_data_df, 'reported_data', './data')

        logger.debug('Data ingestion successfully completed !')
        
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

