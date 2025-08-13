"""
Data Loading and Preprocessing Module
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_heatwave_data(file_path='data/1972_2024_Heatwave_Daily.xlsx'):
    """Load and preprocess heatwave data"""
    print("📂 Loading heatwave data...")
    data = pd.read_excel(file_path)
    
    # Add basic time features
    data['Year'] = data['timestamp'].dt.year
    data['Month'] = data['timestamp'].dt.month
    data['DayOfYear'] = data['timestamp'].dt.dayofyear
    data['Season'] = data['timestamp'].dt.month % 12 // 3 + 1
    
    # Define heatwave threshold and flag
    threshold = 36
    data['Heatwave'] = data['Dhaka Temperature [2 m elevation corrected]'] > threshold
    
    # Temperature range
    data['Temperature Range'] = (data['Dhaka Temperature [2 m elevation corrected].2'] - 
                               data['Dhaka Temperature [2 m elevation corrected].1'])
    
    print(f"✅ Loaded {len(data):,} heatwave records ({data['timestamp'].min().year}-{data['timestamp'].max().year})")
    return data, threshold

def load_deforestation_data(file_path='data/GFW_Dhaka.csv'):
    """Load and preprocess deforestation data"""
    print("📂 Loading deforestation data...")
    deforestation_data = pd.read_csv(file_path)
    
    # Analyze tree cover loss by year
    tree_loss_by_year = deforestation_data.groupby('Tree_Cover_Loss_Year')['umd_tree_cover_loss__ha'].sum().reset_index()
    tree_loss_by_year = tree_loss_by_year[tree_loss_by_year['Tree_Cover_Loss_Year'] > 0]
    tree_loss_by_year['Year'] = tree_loss_by_year['Tree_Cover_Loss_Year'].astype(int)
    
    print(f"✅ Loaded {len(tree_loss_by_year)} years of deforestation data (Total: {tree_loss_by_year['umd_tree_cover_loss__ha'].sum():.0f} ha)")
    return deforestation_data, tree_loss_by_year

def combine_datasets(data, tree_loss_by_year):
    """Combine climate and deforestation data"""
    print("Combining datasets...")
    
    # Prepare annual temperature data
    annual_temp_stats = data.groupby('Year').agg({
        'Dhaka Temperature [2 m elevation corrected]': ['mean', 'max', 'min', 'std'],
        'Dhaka Relative Humidity [2 m]': 'mean',
        'Dhaka Precipitation Total': 'sum'
    }).round(3)
    
    # Flatten column names
    annual_temp_stats.columns = ['_'.join(col).strip() for col in annual_temp_stats.columns]
    annual_temp_stats = annual_temp_stats.reset_index()
    
    # Merge with deforestation data
    combined_data = pd.merge(annual_temp_stats, 
                            tree_loss_by_year[['Year', 'umd_tree_cover_loss__ha']], 
                            on='Year', how='left')
    
    # Fill missing deforestation values with 0 (years before 2001)
    combined_data['umd_tree_cover_loss__ha'] = combined_data['umd_tree_cover_loss__ha'].fillna(0)
    
    print("Datasets combined successfully!")
    return combined_data, annual_temp_stats

def get_dataset_summary(data):
    """Get comprehensive dataset summary statistics"""
    temp_col = 'Dhaka Temperature [2 m elevation corrected]'
    
    summary = {
        'shape': data.shape,
        'time_range': (data['timestamp'].min(), data['timestamp'].max()),
        'total_days': len(data),
        'total_years': data['timestamp'].dt.year.nunique(),
        'missing_values': data.isnull().sum().sum(),
        'temperature_stats': {
            'mean': data[temp_col].mean(),
            'median': data[temp_col].median(),
            'std': data[temp_col].std(),
            'min': data[temp_col].min(),
            'max': data[temp_col].max(),
            'p95': data[temp_col].quantile(0.95),
            'p99': data[temp_col].quantile(0.99)
        }
    }
    return summary
