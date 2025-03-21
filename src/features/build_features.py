# src/features/build_features.py

import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path

# Import feature creation functions
from .temporal_features import create_temporal_features
from .spatial_features import create_airport_features
from .carrier_features import create_carrier_features
# Import new feature modules
from .cyclical_features import create_cyclical_features
from .interaction_features import create_interaction_features
from .network_features import create_network_features

def convert_data_types(df):
    """Convert data types and clean the input dataframe."""
    df_out = df.copy()
    
    # check if the columns have 'c-' prefix and remove it
    if isinstance(df_out['Month'].iloc[0], str) and 'c-' in df_out['Month'].iloc[0]:
        df_out['Month'] = df_out['Month'].str.replace('c-', '').astype(int)
        df_out['DayofMonth'] = df_out['DayofMonth'].str.replace('c-', '').astype(int)
        df_out['DayOfWeek'] = df_out['DayOfWeek'].str.replace('c-', '').astype(int)
    
    # convert target variable if it's Y/N
    if 'dep_delayed_15min' in df_out.columns and df_out['dep_delayed_15min'].dtype == object:
        df_out['dep_delayed_15min'] = df_out['dep_delayed_15min'].map({'Y': 1, 'N': 0})
    
    return df_out

def build_features(input_filepath, output_filepath, is_train=True, feature_store_path=None):
    """
    Main feature engineering pipeline.
    
    Parameters:
    -----------
    input_filepath : str
        Path to the raw data file
    output_filepath : str
        Path to save the processed data
    is_train : bool, default=True
        Whether this is training data (used for fitting transformations)
    feature_store_path : str, optional
        Path to save/load feature transformations
    """
    # create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # load the data
    print(f"Loading data from {input_filepath}")
    df = pd.read_csv(input_filepath)
    
    # convert data types
    print("Converting data types")
    df = convert_data_types(df)
    
    # create feature store directory if needed
    if feature_store_path is not None:
        os.makedirs(feature_store_path, exist_ok=True)
    
    # load training data for reference if this is test data
    train_data = None
    if not is_train and feature_store_path is not None:
        train_path = os.path.join(feature_store_path, 'train_reference.csv')
        if os.path.exists(train_path):
            print(f"Loading training data reference from {train_path}")
            train_data = pd.read_csv(train_path)
    
    # apply feature transformations - original features
    print("Creating temporal features")
    df = create_temporal_features(df)
    
    print("Creating spatial features")
    df = create_airport_features(df, train_data)
    
    print("Creating carrier features")
    df = create_carrier_features(df, train_data)
    
    # create interaction features from original pipeline
    print("Creating basic interaction features")
    df['hub_to_hub'] = df['origin_is_hub'] * df['dest_is_hub']
    df['peak_weekend'] = df['is_weekend'] * df['is_peak_travel_season']
    
    # NEW ENHANCED FEATURES
    print("Creating cyclical features")
    df = create_cyclical_features(df)
    
    print("Creating advanced interaction features")
    df = create_interaction_features(df)
    
    print("Creating network effect features")
    df = create_network_features(df, train_data)
    
    # save a reference copy of the training data for future transformations
    if is_train and feature_store_path is not None:
        train_ref_path = os.path.join(feature_store_path, 'train_reference.csv')
        print(f"Saving training data reference to {train_ref_path}")
        df.to_csv(train_ref_path, index=False)
    
    # save the processed data
    print(f"Saving processed data to {output_filepath}")
    df.to_csv(output_filepath, index=False)
    
    print("Feature engineering completed!")
    return df

def main():
    """
    Entry point for feature engineering script.
    Example usage: 
        python -m src.features.build_features
    """
    # project base path
    project_dir = Path(__file__).resolve().parents[2]
    
    # define file paths
    raw_data_dir = os.path.join(project_dir, 'data', 'raw')
    processed_data_dir = os.path.join(project_dir, 'data', 'processed')
    feature_store_dir = os.path.join(project_dir, 'models', 'feature_store')
    
    # make sure directories exist
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(feature_store_dir, exist_ok=True)
    
    # process training data
    train_input = os.path.join(raw_data_dir, 'flight_delays_train.csv')
    train_output = os.path.join(processed_data_dir, 'flight_delays_train_features.csv')
    build_features(train_input, train_output, is_train=True, feature_store_path=feature_store_dir)
    
    # process test data
    test_input = os.path.join(raw_data_dir, 'flight_delays_test.csv')
    test_output = os.path.join(processed_data_dir, 'flight_delays_test_features.csv')
    build_features(test_input, test_output, is_train=False, feature_store_path=feature_store_dir)

if __name__ == '__main__':
    main()