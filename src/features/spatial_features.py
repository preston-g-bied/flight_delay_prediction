# src/features/spatial_features.py

import pandas as pd
import numpy as np

def create_airport_features(df, train_data=None):
    """
    Create airport and route-related features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with the raw flight data
    train_data : pandas.DataFrame, optional
        Training data to extract airport statistics (for test data transformation)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with airport features added
    """
    # use the provided dataframe or itself as reference data
    reference_data = train_data if train_data is not None else df
    df_out = df.copy()

    # airport frequency - how busy are these airports?
    origin_counts = reference_data.groupby('Origin').size()
    dest_counts = reference_data.groupby('Dest').size()

    # convert to percentile ranks for better generalization
    origin_ranks = origin_counts.rank(pct=True)
    dest_ranks = dest_counts.rank(pct=True)

    df_out['origin_freq_rank'] = df_out['Origin'].map(origin_ranks)
    df_out['dest_freq_rank'] = df_out['Dest'].map(dest_ranks)

    # airport delay statistics
    if 'dep_delayed_15min' in reference_data.columns:
        if reference_data['dep_delayed_15min'].dtype == object:
            # convert Y/N to 1/0 if not already done
            reference_delay = reference_data.copy()
            reference_delay['dep_delayed_15min'] = reference_delay['dep_delayed_15min'].map({'Y': 1, 'N': 0})
        else:
            reference_delay = reference_data
            
        origin_delay_rates = reference_delay.groupby('Origin')['dep_delayed_15min'].mean()
        dest_delay_rates = reference_delay.groupby('Dest')['dep_delayed_15min'].mean()
        
        df_out['origin_delay_rate'] = df_out['Origin'].map(origin_delay_rates)
        df_out['dest_delay_rate'] = df_out['Dest'].map(dest_delay_rates)

    # hub airport indicators (top 10 by frequency)
    top_origins = origin_counts.nlargest(10).index
    top_dests = dest_counts.nlargest(10).index
    
    df_out['origin_is_hub'] = df_out['Origin'].isin(top_origins).astype(int)
    df_out['dest_is_hub'] = df_out['Dest'].isin(top_dests).astype(int)
    
    # create route features
    df_out['route'] = df_out['Origin'] + '_' + df_out['Dest']
    
    # route statistics
    route_counts = reference_data.groupby('route').size() if 'route' in reference_data.columns else reference_data.groupby(['Origin', 'Dest']).size()
    route_ranks = route_counts.rank(pct=True)
    
    if 'route' in reference_data.columns:
        df_out['route_freq_rank'] = df_out['route'].map(route_ranks)
    else:
        # handle test data that might not have the route column yet
        route_map = {(o, d): r for (o, d), r in zip(route_ranks.index, route_ranks.values)}
        df_out['route_freq_rank'] = df_out.apply(lambda x: route_map.get((x['Origin'], x['Dest']), np.nan), axis=1)
    
    # distance-based features
    df_out['distance_category'] = pd.cut(
        df_out['Distance'],
        bins=[0, 300, 600, 1000, 2000, 5000],
        labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
    )
    
    return df_out