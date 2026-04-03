# Import libraries
import numpy as np
import pandas as pd
import os
import yaml
import logging
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score

# Logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

# Set console logger
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Set file logger
file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

# Set format for logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to loggers
logger.addHandler(console_handler)
logger.addHandler(file_handler)
print()
logger.debug('------------------------------------------- MODEL BUILDING STARTED -------------------------------------------------------')

def load_data(path: str) -> pd.DataFrame:
    '''Loads a file from a given path'''
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.error('Unexpected error occured while loading file: %s', e)
        raise


def train_test_split(df: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
    '''Returns training and testing data'''

    try:
        train = df.iloc[:19].copy()
        test = df.iloc[[19]].copy()
        features.append(target)
        
        return train[features], test[features]
    
    except Exception as e:
        logger.error('Unexpected error occurred while doing train-test splitting during model building phase')
        raise


def save_model(model, path: str) -> None:
    '''Saves the model to a given path'''
    try:
        with open(path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def train_LR_model(engineered_data, selected_features, target) -> LinearRegression:
    '''Trains a Linear regression model on a dataset with given selected features and target'''

    # Train-Test split
    train, test = train_test_split(engineered_data, selected_features, target='revenue_index')

    # Save train and test data
    try:
        os.makedirs(os.path.dirname(os.path.join("./data", "processed")), exist_ok=True)
        train.to_csv('./data/processed/train_data.csv', index=False)
        test.to_csv('./data/processed/test_data.csv', index=False)
            
    except Exception as e:
        logger.error('Some error happened while saving train and test data locally: %s', e)
        raise

    # Make LR model
    final_model = LinearRegression()

    # Train model on all user given features
    final_model.fit(train.iloc[:, :-1], train['revenue_index'])
    logger.debug('Model training completed successfully')

    # Save the model
    save_model(final_model, './models/model.pkl')

    return final_model


def main():
    try:
        # Load data
        engineered_data = load_data('./data/processed/engineered_data.csv')

        # Selected features for training: from experiment 2 in notebooks
        selected_features = ['sum_spend_per_user', 'avg_weekly_active_users_index']

        # Train LR model
        trained_model = train_LR_model(engineered_data, selected_features, target='revenue_index')

        logger.debug('Model building process completed !')
    
    except Exception as e:
        logger.error('Failed to build the model')
        raise


if __name__ == '__main__':
    main()

