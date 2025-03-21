# src/features/test_features.py

import pandas as pd
import os
from pathlib import Path

# Import main feature builder
from .build_features import build_features

def test_feature_engineering():
    """Test the feature engineering pipeline with a small sample"""
    # Project directory
    project_dir = Path(__file__).resolve().parents[2]
    
    # Define paths
    raw_data_dir = os.path.join(project_dir, 'data', 'raw')
    test_dir = os.path.join(project_dir, 'data', 'test')
    os.makedirs(test_dir, exist_ok=True)
    
    # Load a small sample of training data
    train_input = os.path.join(raw_data_dir, 'flight_delays_train.csv')
    df = pd.read_csv(train_input)
    sample = df.sample(1000, random_state=42)  # Take 1000 rows for quick testing
    
    # Save the sample
    sample_path = os.path.join(test_dir, 'sample_train.csv')
    sample.to_csv(sample_path, index=False)
    
    # Process the sample
    output_path = os.path.join(test_dir, 'sample_processed.csv')
    feature_store = os.path.join(test_dir, 'feature_store')
    
    # Run feature engineering
    processed_df = build_features(sample_path, output_path, True, feature_store)
    
    # Print feature summary
    print(f"\nProcessed data shape: {processed_df.shape}")
    print("\nNew features added:")
    
    # Original features in sample
    orig_cols = set(sample.columns)
    # New features
    new_cols = set(processed_df.columns) - orig_cols
    
    # Group features by type for better organization
    cyclical_features = [col for col in new_cols if 'sin' in col or 'cos' in col]
    network_features = [col for col in new_cols if 'hourly' in col or 'congestion' in col or 'connections' in col]
    interaction_features = [col for col in new_cols if 'combo' in col or 'rush' in col or 'winter' in col or 'risk' in col]
    
    print("\nCyclical Features:")
    for col in sorted(cyclical_features):
        print(f"- {col}")
    
    print("\nNetwork Features:")
    for col in sorted(network_features):
        print(f"- {col}")
    
    print("\nInteraction Features:")
    for col in sorted(interaction_features):
        print(f"- {col}")
    
    print("\nOther New Features:")
    other_features = new_cols - set(cyclical_features) - set(network_features) - set(interaction_features)
    for col in sorted(other_features):
        print(f"- {col}")
    
    print("\nFeature engineering test completed successfully!")
    
if __name__ == "__main__":
    test_feature_engineering()