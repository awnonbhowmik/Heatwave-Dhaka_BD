"""
Visualization Module
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def plot_temperature_trends(data, annual_temp_stats):
    """Plot temperature trends over time"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Daily temperature plot (recent years)
    recent_data = data[data['Year'] >= 2020]
    axes[0,0].plot(recent_data['timestamp'], recent_data['Dhaka Temperature [2 m elevation corrected]'], 
                   alpha=0.7, linewidth=1)
    axes[0,0].set_title('Daily Temperature (2020-2024)', fontweight='bold')
    axes[0,0].set_xlabel('Date')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Annual temperature trends
    axes[0,1].plot(annual_temp_stats['Year'], annual_temp_stats['Dhaka Temperature [2 m elevation corrected]_mean'], 
                   marker='o', linewidth=2, markersize=4, color='red', label='Mean Temp')
    axes[0,1].plot(annual_temp_stats['Year'], annual_temp_stats['Dhaka Temperature [2 m elevation corrected]_max'], 
                   marker='s', linewidth=1, markersize=3, color='darkred', label='Max Temp')
    axes[0,1].set_title('Annual Temperature Trends (1972-2024)', fontweight='bold')
    axes[0,1].set_xlabel('Year')
    axes[0,1].set_ylabel('Temperature (°C)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Temperature distribution
    axes[1,0].hist(data['Dhaka Temperature [2 m elevation corrected]'], bins=50, alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Temperature Distribution', fontweight='bold')
    axes[1,0].set_xlabel('Temperature (°C)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Seasonal temperature patterns
    season_names = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
    seasonal_data = data.groupby('Season')['Dhaka Temperature [2 m elevation corrected]'].mean()
    axes[1,1].bar([season_names[i] for i in seasonal_data.index], seasonal_data.values, color='orange', alpha=0.7)
    axes[1,1].set_title('Seasonal Temperature Averages', fontweight='bold')
    axes[1,1].set_xlabel('Season')
    axes[1,1].set_ylabel('Average Temperature (°C)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_deforestation_analysis(tree_loss_by_year, combined_data):
    """Plot deforestation analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Tree cover loss over time
    axes[0,0].plot(tree_loss_by_year['Year'], tree_loss_by_year['umd_tree_cover_loss__ha'], 
                   marker='o', linewidth=2, markersize=6, color='darkgreen')
    axes[0,0].set_title('Annual Tree Cover Loss in Dhaka (2001-2023)', fontweight='bold')
    axes[0,0].set_xlabel('Year')
    axes[0,0].set_ylabel('Tree Cover Loss (hectares)')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Deforestation vs Temperature scatter
    valid_data = combined_data[combined_data['umd_tree_cover_loss__ha'] > 0]
    axes[0,1].scatter(valid_data['umd_tree_cover_loss__ha'], 
                      valid_data['Dhaka Temperature [2 m elevation corrected]_mean'],
                      alpha=0.7, s=80, color='purple')
    axes[0,1].set_title('Tree Cover Loss vs Mean Temperature', fontweight='bold')
    axes[0,1].set_xlabel('Tree Cover Loss (hectares)')
    axes[0,1].set_ylabel('Mean Temperature (°C)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(valid_data['umd_tree_cover_loss__ha'], 
                   valid_data['Dhaka Temperature [2 m elevation corrected]_mean'], 1)
    p = np.poly1d(z)
    axes[0,1].plot(valid_data['umd_tree_cover_loss__ha'], 
                   p(valid_data['umd_tree_cover_loss__ha']), "r--", alpha=0.8)
    
    # 3. Cumulative deforestation
    cumulative_loss = tree_loss_by_year['umd_tree_cover_loss__ha'].cumsum()
    axes[1,0].plot(tree_loss_by_year['Year'], cumulative_loss, marker='s', linewidth=2, color='brown')
    axes[1,0].set_title('Cumulative Tree Cover Loss', fontweight='bold')
    axes[1,0].set_xlabel('Year')
    axes[1,0].set_ylabel('Cumulative Loss (hectares)')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Normalized comparison
    scaler = MinMaxScaler()
    norm_data = combined_data[combined_data['Year'] >= 2001].copy()
    norm_data[['temp_norm', 'deforest_norm']] = scaler.fit_transform(
        norm_data[['Dhaka Temperature [2 m elevation corrected]_mean', 'umd_tree_cover_loss__ha']])
    
    axes[1,1].plot(norm_data['Year'], norm_data['temp_norm'], 
                   marker='o', linewidth=2, label='Temperature (normalized)', color='red')
    axes[1,1].plot(norm_data['Year'], norm_data['deforest_norm'], 
                   marker='s', linewidth=2, label='Deforestation (normalized)', color='green')
    axes[1,1].set_title('Normalized Temperature vs Deforestation Trends', fontweight='bold')
    axes[1,1].set_xlabel('Year')
    axes[1,1].set_ylabel('Normalized Values (0-1)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_heatwave_analysis(data, heatwave_summary, threshold=36):
    """Plot heatwave analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Heatwave days per year
    heatwave_days_per_year = data[data['Heatwave']].groupby('Year').size()
    axes[0,0].bar(heatwave_days_per_year.index, heatwave_days_per_year.values, alpha=0.7)
    axes[0,0].set_title('Heatwave Days per Year in Dhaka (1972-2024)', fontweight='bold')
    axes[0,0].set_xlabel('Year')
    axes[0,0].set_ylabel('Number of Heatwave Days')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Heatwave days by month
    heatwave_days_per_month = data[data['Heatwave']].groupby('Month').size()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axes[0,1].bar(range(1, 13), [heatwave_days_per_month.get(i, 0) for i in range(1, 13)], color='orange', alpha=0.7)
    axes[0,1].set_title('Heatwave Days by Month in Dhaka', fontweight='bold')
    axes[0,1].set_xlabel('Month')
    axes[0,1].set_ylabel('Number of Heatwave Days')
    axes[0,1].set_xticks(range(1, 13))
    axes[0,1].set_xticklabels(month_names)
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Rolling average of heatwave days
    rolling_average = heatwave_days_per_year.rolling(window=5).mean()
    axes[1,0].plot(heatwave_days_per_year.index, heatwave_days_per_year.values, label='Heatwave Days', alpha=0.5)
    axes[1,0].plot(rolling_average.index, rolling_average.values, label='5-Year Rolling Average', color='red', linewidth=2)
    axes[1,0].set_title('Heatwave Trends (Rolling Average)', fontweight='bold')
    axes[1,0].set_xlabel('Year')
    axes[1,0].set_ylabel('Number of Heatwave Days')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Heatwave duration histogram
    if len(heatwave_summary) > 0:
        axes[1,1].hist(heatwave_summary['Duration'], bins=20, alpha=0.7, edgecolor='black')
        axes[1,1].set_title('Heatwave Duration Distribution', fontweight='bold')
        axes[1,1].set_xlabel('Duration (days)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(data):
    """Plot correlation matrix of climate variables"""
    selected_columns = [
        'Dhaka Temperature [2 m elevation corrected]',
        'Dhaka Precipitation Total',
        'Dhaka Relative Humidity [2 m]',
        'Dhaka Wind Speed [10 m]',
        'Dhaka Cloud Cover Total',
        'Dhaka Sunshine Duration',
        'Dhaka Mean Sea Level Pressure [MSL]'
    ]
    
    # Filter columns that exist in the data
    available_columns = [col for col in selected_columns if col in data.columns]
    selected_data = data[available_columns].apply(pd.to_numeric, errors='coerce')
    
    correlation_matrix = selected_data.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                linewidths=0.5, vmax=1, vmin=-1)
    plt.title("Correlation Matrix of Climate Variables", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def create_summary_dashboard(data, statistical_results, tree_loss_by_year):
    """Create a summary dashboard with key metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Key metrics
    total_years = data['timestamp'].dt.year.nunique()
    avg_temp = data['Dhaka Temperature [2 m elevation corrected]'].mean()
    total_heatwave_days = data['Heatwave'].sum()
    total_deforestation = tree_loss_by_year['umd_tree_cover_loss__ha'].sum()
    
    # 1. Temperature trend
    annual_data = data.groupby('Year')['Dhaka Temperature [2 m elevation corrected]'].mean()
    axes[0,0].plot(annual_data.index, annual_data.values, linewidth=2, color='red')
    axes[0,0].set_title(f'Temperature Trend\n({total_years} years)', fontweight='bold')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Heatwave frequency
    heatwave_annual = data[data['Heatwave']].groupby('Year').size()
    axes[0,1].bar(heatwave_annual.index, heatwave_annual.values, alpha=0.7, color='orange')
    axes[0,1].set_title(f'Total Heatwave Days\n{total_heatwave_days}', fontweight='bold')
    axes[0,1].set_ylabel('Heatwave Days')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Deforestation
    axes[0,2].bar(tree_loss_by_year['Year'], tree_loss_by_year['umd_tree_cover_loss__ha'], 
                  alpha=0.7, color='green')
    axes[0,2].set_title(f'Deforestation\n{total_deforestation:.0f} hectares lost', fontweight='bold')
    axes[0,2].set_ylabel('Tree Loss (ha)')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Temperature distribution by decade
    data['Decade'] = (data['Year'] // 10) * 10
    decade_temps = data.groupby('Decade')['Dhaka Temperature [2 m elevation corrected]'].mean()
    axes[1,0].bar([f"{int(d)}s" for d in decade_temps.index], decade_temps.values, alpha=0.7)
    axes[1,0].set_title('Average Temperature by Decade', fontweight='bold')
    axes[1,0].set_ylabel('Temperature (°C)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 5. Seasonal patterns
    seasonal_data = data.groupby('Season')['Dhaka Temperature [2 m elevation corrected]'].mean()
    season_names = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
    axes[1,1].bar([season_names[s] for s in seasonal_data.index], seasonal_data.values, alpha=0.7)
    axes[1,1].set_title('Seasonal Temperature Averages', fontweight='bold')
    axes[1,1].set_ylabel('Temperature (°C)')
    
    # 6. Key statistics text
    axes[1,2].axis('off')
    stats_text = f"""
    KEY STATISTICS
    
    Dataset Period: {data['timestamp'].min().strftime('%Y')} - {data['timestamp'].max().strftime('%Y')}
    Total Records: {len(data):,}
    
    Temperature:
    • Average: {avg_temp:.2f}°C
    • Max: {data['Dhaka Temperature [2 m elevation corrected]'].max():.1f}°C
    • Min: {data['Dhaka Temperature [2 m elevation corrected]'].min():.1f}°C
    
    Heatwaves (>36°C):
    • Total Days: {total_heatwave_days:,}
    • Avg per Year: {total_heatwave_days/total_years:.1f}
    
    Deforestation:
    • Total Loss: {total_deforestation:.0f} ha
    • Period: 2001-2023
    """
    
    if 'temperature_trend' in statistical_results:
        temp_trend = statistical_results['temperature_trend']
        stats_text += f"""
    
    Trends:
    • Warming Rate: {temp_trend['slope']:.4f}°C/year
    • 52-year Increase: {temp_trend['total_increase_52years']:.2f}°C
        """
    
    axes[1,2].text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    plt.show()
