# src/features/network_features.py

import pandas as pd
import numpy as np

def create_network_features(df, train_data=None):
    """
    Create features that capture network effects in flight delays.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with existing features
    train_data : pandas.DataFrame, optional
        Training data for extracting statistics (for test data transform)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with network effect features added
    """
    # Use provided data or itself as reference
    reference_data = train_data if train_data is not None else df
    df_out = df.copy()
    
    # 1. Airport congestion features
    # Calculate number of departures per airport per hour
    if 'dep_hour' in reference_data.columns:
        # Group by origin and hour
        airport_hourly_traffic = reference_data.groupby(['Origin', 'dep_hour']).size().reset_index(name='origin_hourly_flights')
        
        # Create a dictionary for faster lookups
        airport_hour_map = dict(zip(zip(airport_hourly_traffic['Origin'], airport_hourly_traffic['dep_hour']), 
                                 airport_hourly_traffic['origin_hourly_flights']))
        
        # Map to dataframe
        df_out['origin_hourly_flights'] = df_out.apply(
            lambda x: airport_hour_map.get((x['Origin'], x['dep_hour']), 0), axis=1
        )
        
        # Calculate congestion percentile for each airport-hour combination
        df_out['origin_congestion_rank'] = df_out.groupby('Origin')['origin_hourly_flights'].transform(
            lambda x: x.rank(pct=True)
        )
    
    # 2. Hub connectivity
    # Number of destinations served from each origin
    origin_connectivity = reference_data.groupby('Origin')['Dest'].nunique().reset_index(name='origin_num_connections')
    df_out = pd.merge(df_out, origin_connectivity, on='Origin', how='left')
    
    # 3. Carrier load at time of day
    # Number of flights by carrier per hour
    if 'dep_hour' in reference_data.columns:
        carrier_hourly = reference_data.groupby(['UniqueCarrier', 'dep_hour']).size().reset_index(name='carrier_hourly_flights')
        
        # Create mapping
        carrier_hour_map = dict(zip(zip(carrier_hourly['UniqueCarrier'], carrier_hourly['dep_hour']), 
                                   carrier_hourly['carrier_hourly_flights']))
        
        # Apply mapping
        df_out['carrier_hourly_flights'] = df_out.apply(
            lambda x: carrier_hour_map.get((x['UniqueCarrier'], x['dep_hour']), 0), axis=1
        )
    
    # 4. Airport delay propagation effect
    # If we have delay information in the reference data
    if 'dep_delayed_15min' in reference_data.columns:
        # Convert if needed
        if reference_data['dep_delayed_15min'].dtype == object:
            ref_delay = reference_data.copy()
            ref_delay['dep_delayed_15min'] = ref_delay['dep_delayed_15min'].map({'Y': 1, 'N': 0})
        else:
            ref_delay = reference_data
            
        # Calculate recent delay rates by hour at each airport
        airport_hourly_delays = ref_delay.groupby(['Origin', 'dep_hour'])['dep_delayed_15min'].mean().reset_index(name='origin_hour_delay_rate')
        
        # Create mapping
        airport_hour_delay_map = dict(zip(zip(airport_hourly_delays['Origin'], airport_hourly_delays['dep_hour']), 
                                       airport_hourly_delays['origin_hour_delay_rate']))
        
        # Apply mapping
        df_out['origin_hour_delay_rate'] = df_out.apply(
            lambda x: airport_hour_delay_map.get((x['Origin'], x['dep_hour']), np.nan), axis=1
        )
        
        # Fill missing with overall airport delay rate
        if 'origin_delay_rate' in df_out.columns:
            df_out['origin_hour_delay_rate'] = df_out['origin_hour_delay_rate'].fillna(df_out['origin_delay_rate'])
    
    # 5. Route congestion
    route_hourly = reference_data.groupby(['Origin', 'Dest', 'dep_hour']).size().reset_index(name='route_hourly_flights')
    
    # Merge this information
    df_out = pd.merge(
        df_out,
        route_hourly,
        on=['Origin', 'Dest', 'dep_hour'],
        how='left'
    )
    
    # Fill NAs with 0
    df_out['route_hourly_flights'] = df_out['route_hourly_flights'].fillna(0)
    
    # Calculate route congestion rank
    df_out['route_congestion_rank'] = df_out.groupby(['Origin', 'Dest'])['route_hourly_flights'].transform(
        lambda x: x.rank(pct=True) if len(x) > 1 else 0.5
    )
    
    return df_out