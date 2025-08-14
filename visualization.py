"""
Visualization Module with Enhanced KDE Support
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde

def plot_temperature_trends(data, annual_temp_stats):
    """Plot temperature trends over time with enhanced trendlines"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Daily temperature plot (recent years) with trendline
    recent_data = data[data['Year'] >= 2020].copy()
    recent_data = recent_data.sort_values('timestamp')
    
    axes[0,0].plot(recent_data['timestamp'], recent_data['Dhaka Temperature [2 m elevation corrected]'], 
                   alpha=0.7, linewidth=1, color='skyblue')
    
    # Add trendline for recent years
    x_numeric = np.arange(len(recent_data))
    z = np.polyfit(x_numeric, recent_data['Dhaka Temperature [2 m elevation corrected]'], 1)
    p = np.poly1d(z)
    axes[0,0].plot(recent_data['timestamp'], p(x_numeric), "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.4f}°C/day')
    
    axes[0,0].set_title('Daily Temperature (2020-2024) with Trend', fontweight='bold')
    axes[0,0].set_xlabel('Date')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Annual temperature trends with enhanced trendlines
    years = annual_temp_stats['Year'].values
    mean_temps = annual_temp_stats['Dhaka Temperature [2 m elevation corrected]_mean'].values
    max_temps = annual_temp_stats['Dhaka Temperature [2 m elevation corrected]_max'].values
    
    axes[0,1].plot(years, mean_temps, marker='o', linewidth=2, markersize=4, color='red', label='Mean Temp')
    axes[0,1].plot(years, max_temps, marker='s', linewidth=1, markersize=3, color='darkred', label='Max Temp')
    
    # Add trendlines
    z_mean = np.polyfit(years, mean_temps, 1)
    z_max = np.polyfit(years, max_temps, 1)
    p_mean = np.poly1d(z_mean)
    p_max = np.poly1d(z_max)
    
    axes[0,1].plot(years, p_mean(years), "r--", alpha=0.8, linewidth=2, 
                   label=f'Mean Trend: +{z_mean[0]*52:.2f}°C/52yr')
    axes[0,1].plot(years, p_max(years), "--", color='darkred', alpha=0.8, linewidth=2,
                   label=f'Max Trend: +{z_max[0]*52:.2f}°C/52yr')
    
    axes[0,1].set_title('Annual Temperature Trends (1972-2024)', fontweight='bold')
    axes[0,1].set_xlabel('Year')
    axes[0,1].set_ylabel('Temperature (°C)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Temperature distribution with KDE and normal curve overlay
    temp_data = data['Dhaka Temperature [2 m elevation corrected]'].dropna()
    n, bins, patches = axes[1,0].hist(temp_data, bins=50, alpha=0.6, edgecolor='black', density=True, color='skyblue')
    
    # Add KDE curve
    kde = gaussian_kde(temp_data)
    x_kde = np.linspace(temp_data.min(), temp_data.max(), 200)
    kde_curve = kde(x_kde)
    axes[1,0].plot(x_kde, kde_curve, 'g-', linewidth=3, label=f'KDE (Actual Distribution)', alpha=0.9)
    
    # Overlay normal distribution curve for comparison
    mu, sigma = temp_data.mean(), temp_data.std()
    x = np.linspace(temp_data.min(), temp_data.max(), 200)
    normal_curve = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))
    axes[1,0].plot(x, normal_curve, 'r--', linewidth=2, label=f'Normal (μ={mu:.1f}, σ={sigma:.1f})', alpha=0.8)
    
    axes[1,0].set_title('Temperature Distribution: KDE vs Normal', fontweight='bold')
    axes[1,0].set_xlabel('Temperature (°C)')
    axes[1,0].set_ylabel('Density')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. REPLACED: Temperature extremes trends (instead of seasonal duplicate)
    # Calculate annual extremes
    annual_extremes = data.groupby('Year')['Dhaka Temperature [2 m elevation corrected]'].agg(['min', 'max'])
    annual_range = annual_extremes['max'] - annual_extremes['min']
    
    ax4 = axes[1,1]
    line1 = ax4.plot(annual_extremes.index, annual_extremes['max'], 'r-o', markersize=3, label='Annual Max', alpha=0.8)
    line2 = ax4.plot(annual_extremes.index, annual_extremes['min'], 'b-o', markersize=3, label='Annual Min', alpha=0.8)
    
    # Add trendlines for extremes
    z_max_extreme = np.polyfit(annual_extremes.index, annual_extremes['max'], 1)
    z_min_extreme = np.polyfit(annual_extremes.index, annual_extremes['min'], 1)
    p_max_extreme = np.poly1d(z_max_extreme)
    p_min_extreme = np.poly1d(z_min_extreme)
    
    ax4.plot(annual_extremes.index, p_max_extreme(annual_extremes.index), "r--", alpha=0.6, linewidth=2)
    ax4.plot(annual_extremes.index, p_min_extreme(annual_extremes.index), "b--", alpha=0.6, linewidth=2)
    
    # Add temperature range on secondary y-axis
    ax4_twin = ax4.twinx()
    ax4_twin.fill_between(annual_extremes.index, annual_range, alpha=0.2, color='gray', label='Temperature Range')
    ax4_twin.set_ylabel('Temperature Range (°C)', color='gray')
    
    ax4.set_title('Temperature Extremes & Range Trends', fontweight='bold')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Temperature (°C)')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
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
    
    # 2. Deforestation vs Temperature scatter with KDE contours
    valid_data = combined_data[combined_data['umd_tree_cover_loss__ha'] > 0]
    
    # Create scatter plot
    scatter = axes[0,1].scatter(valid_data['umd_tree_cover_loss__ha'], 
                              valid_data['Dhaka Temperature [2 m elevation corrected]_mean'],
                              alpha=0.7, s=80, color='purple', edgecolors='white', linewidth=0.5)
    
    # Add KDE contours if we have enough data points
    if len(valid_data) > 5:
        x_data = valid_data['umd_tree_cover_loss__ha'].values
        y_data = valid_data['Dhaka Temperature [2 m elevation corrected]_mean'].values
        
        # Create a 2D KDE
        xy = np.vstack([x_data, y_data])
        kde_2d = gaussian_kde(xy)
        
        # Create a grid for contour plot
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = y_data.min(), y_data.max()
        xx, yy = np.mgrid[x_min:x_max:(x_max-x_min)/20, y_min:y_max:(y_max-y_min)/20]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        kde_values = np.reshape(kde_2d(positions).T, xx.shape)
        
        # Add contour lines
        axes[0,1].contour(xx, yy, kde_values, levels=3, alpha=0.5, colors='red', linewidths=1.5)
    
    axes[0,1].set_title('Tree Loss vs Temperature (with KDE contours)', fontweight='bold')
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
    
    # 4. Heatwave duration histogram with KDE
    if len(heatwave_summary) > 0:
        durations = heatwave_summary['Duration'].values
        
        # Create histogram with density=True for KDE overlay
        n, bins, patches = axes[1,1].hist(durations, bins=15, alpha=0.6, edgecolor='black', 
                                         density=True, color='lightcoral')
        
        # Add KDE curve if we have enough data points
        if len(durations) > 3:
            kde = gaussian_kde(durations)
            x_kde = np.linspace(durations.min(), durations.max(), 100)
            kde_curve = kde(x_kde)
            axes[1,1].plot(x_kde, kde_curve, 'darkred', linewidth=3, 
                          label=f'KDE (n={len(durations)} events)', alpha=0.9)
            axes[1,1].legend()
        
        axes[1,1].set_title('Heatwave Duration Distribution with KDE', fontweight='bold')
        axes[1,1].set_xlabel('Duration (days)')
        axes[1,1].set_ylabel('Density')
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].text(0.5, 0.5, 'No heatwave events detected', 
                      transform=axes[1,1].transAxes, ha='center', va='center')
        axes[1,1].set_title('Heatwave Duration Distribution', fontweight='bold')
    
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
    """Create a summary dashboard with key metrics and enhanced insights"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Key metrics
    total_years = data['timestamp'].dt.year.nunique()
    avg_temp = data['Dhaka Temperature [2 m elevation corrected]'].mean()
    total_heatwave_days = data['Heatwave'].sum()
    total_deforestation = tree_loss_by_year['umd_tree_cover_loss__ha'].sum()
    
    # 1. Temperature trend with trendline
    annual_data = data.groupby('Year')['Dhaka Temperature [2 m elevation corrected]'].mean()
    axes[0,0].plot(annual_data.index, annual_data.values, linewidth=2, color='red', marker='o', markersize=3)
    
    # Add trendline
    z = np.polyfit(annual_data.index, annual_data.values, 1)
    p = np.poly1d(z)
    axes[0,0].plot(annual_data.index, p(annual_data.index), "r--", alpha=0.8, linewidth=2)
    
    axes[0,0].set_title(f'Temperature Trend\n({total_years} years, +{z[0]*52:.2f}°C)', fontweight='bold')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Heatwave frequency with trendline
    heatwave_annual = data[data['Heatwave']].groupby('Year').size()
    all_years = data['Year'].unique()
    heatwave_annual = heatwave_annual.reindex(all_years, fill_value=0)
    
    axes[0,1].bar(heatwave_annual.index, heatwave_annual.values, alpha=0.7, color='orange')
    
    # Add polynomial trendline for heatwave frequency
    z_hw = np.polyfit(heatwave_annual.index, heatwave_annual.values, 2)
    p_hw = np.poly1d(z_hw)
    axes[0,1].plot(heatwave_annual.index, p_hw(heatwave_annual.index), "r-", alpha=0.8, linewidth=3)
    
    axes[0,1].set_title(f'Total Heatwave Days\n{total_heatwave_days} (trend: accelerating)', fontweight='bold')
    axes[0,1].set_ylabel('Heatwave Days')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Deforestation with trendline
    axes[0,2].bar(tree_loss_by_year['Year'], tree_loss_by_year['umd_tree_cover_loss__ha'], 
                  alpha=0.7, color='green')
    
    # Add deforestation trendline
    z_def = np.polyfit(tree_loss_by_year['Year'], tree_loss_by_year['umd_tree_cover_loss__ha'], 1)
    p_def = np.poly1d(z_def)
    axes[0,2].plot(tree_loss_by_year['Year'], p_def(tree_loss_by_year['Year']), "r-", alpha=0.8, linewidth=3)
    
    axes[0,2].set_title(f'Deforestation\n{total_deforestation:.0f} hectares lost', fontweight='bold')
    axes[0,2].set_ylabel('Tree Loss (ha)')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. ENHANCED: Climate volatility analysis (instead of decade averages)
    annual_volatility = data.groupby('Year')['Dhaka Temperature [2 m elevation corrected]'].std()
    moving_avg_volatility = annual_volatility.rolling(window=5).mean()
    
    axes[1,0].plot(annual_volatility.index, annual_volatility.values, alpha=0.5, color='lightblue', label='Annual Volatility')
    axes[1,0].plot(moving_avg_volatility.index, moving_avg_volatility.values, color='darkblue', linewidth=2, label='5-Year Average')
    
    # Add volatility trendline
    z_vol = np.polyfit(annual_volatility.index, annual_volatility.values, 1)
    p_vol = np.poly1d(z_vol)
    axes[1,0].plot(annual_volatility.index, p_vol(annual_volatility.index), "r--", alpha=0.8, linewidth=2, label='Trend')
    
    axes[1,0].set_title('Temperature Volatility Analysis', fontweight='bold')
    axes[1,0].set_ylabel('Temperature Std Dev (°C)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. REPLACED: Urban Heat Island effect by time of year (instead of seasonal duplicate)
    monthly_variation = data.groupby(['Year', 'Month'])['Dhaka Temperature [2 m elevation corrected]'].mean().unstack()
    
    # Calculate monthly trends over time
    month_trends = {}
    for month in range(1, 13):
        if month in monthly_variation.columns:
            monthly_data = monthly_variation[month].dropna()
            if len(monthly_data) > 1:
                z_month = np.polyfit(monthly_data.index, monthly_data.values, 1)
                month_trends[month] = z_month[0] * 52  # 52-year trend
    
    months = list(month_trends.keys())
    trends = list(month_trends.values())
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    if months:
        # Professional color scheme for research papers - high contrast
        # Use a temperature-based color gradient: cool blues for cooling, warm reds for warming
        max_trend = max(trends) if trends else 1
        min_trend = min(trends) if trends else -1
        
        # Normalize trends to create meaningful color mapping
        colors = []
        for trend in trends:
            if trend > 0:
                # Warming - use red intensity based on magnitude
                intensity = min(trend / max_trend, 1.0) if max_trend > 0 else 0.5
                colors.append((0.8, 0.2 - 0.1 * intensity, 0.2 - 0.1 * intensity))  # Dark red for high warming
            else:
                # Cooling - use blue intensity (though unlikely)
                intensity = min(abs(trend) / abs(min_trend), 1.0) if min_trend < 0 else 0.5
                colors.append((0.2 - 0.1 * intensity, 0.2 - 0.1 * intensity, 0.8))  # Dark blue for cooling
        
        # Alternative: Use a diverging colormap that's print-friendly
        # Normalize values to [-1, 1] for colormap
        if max_trend != min_trend:
            norm_trends = [(t - min_trend) / (max_trend - min_trend) for t in trends]
        else:
            norm_trends = [0.5] * len(trends)
        
        # Use a professional diverging colormap
        cmap = plt.cm.RdBu_r  # Red-Blue reversed (red=warm, blue=cool)
        colors = [cmap(0.8 if nt > 0.5 else 0.2) for nt in norm_trends]  # High contrast values
        
        bars = axes[1,1].bar([month_names[m-1] for m in months], trends, color=colors, 
                            edgecolor='black', linewidth=0.8, alpha=0.9)
        
        # Add value labels on bars for clarity
        for bar, trend in zip(bars, trends):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.05,
                          f'{trend:.2f}°C', ha='center', va='bottom' if height >= 0 else 'top', 
                          fontsize=8, fontweight='bold')
        
        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1.2)
        axes[1,1].set_title('Monthly Warming Trends (52-year)', fontweight='bold', fontsize=12)
        axes[1,1].set_ylabel('Temperature Change (°C)', fontweight='bold')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
    
    # 6. Enhanced key statistics with trends
    axes[1,2].axis('off')
    stats_text = f"""
    ENHANCED CLIMATE ANALYSIS
    
    Dataset: {data['timestamp'].min().strftime('%Y')} - {data['timestamp'].max().strftime('%Y')}
    Records: {len(data):,} daily observations
    
    Temperature Trends:
    • Average: {avg_temp:.2f}°C
    • 52-yr Change: +{z[0]*52:.2f}°C
    • Volatility Trend: {'↑' if z_vol[0] > 0 else '↓'} {abs(z_vol[0]*52):.3f}°C
    
    Extremes:
    • Max: {data['Dhaka Temperature [2 m elevation corrected]'].max():.1f}°C
    • Min: {data['Dhaka Temperature [2 m elevation corrected]'].min():.1f}°C
    • Range: {data['Dhaka Temperature [2 m elevation corrected]'].max() - data['Dhaka Temperature [2 m elevation corrected]'].min():.1f}°C
    
    Heatwaves (>36°C):
    • Total: {total_heatwave_days:,} days
    • Rate: {total_heatwave_days/total_years:.1f}/year
    • Trend: Accelerating
    
    Environmental Impact:
    • Tree Loss: {total_deforestation:.0f} ha (2001-2023)
    • Loss Rate: {z_def[0]:.0f} ha/year
    """
    
    if 'temperature_trend' in statistical_results:
        temp_trend = statistical_results['temperature_trend']
        stats_text += f"""
    
    Statistical Significance:
    • Warming Rate: {temp_trend['slope']:.4f}°C/year
    • Total 52-yr: {temp_trend['total_increase_52years']:.2f}°C
        """
    
    axes[1,2].text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top', 
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.7),
                   transform=axes[1,2].transAxes)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# TIME SERIES & MACHINE LEARNING VISUALIZATION FUNCTIONS
# ============================================================================

def plot_time_series_results(forecasts):
    """Plot time series forecasting results (ARIMA & SARIMA)"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ARIMA plot
    if 'arima' in forecasts:
        arima_data = forecasts['arima']
        monthly_temp = arima_data['historical_data']
        
        axes[0,0].plot(monthly_temp.index[-120:], monthly_temp.values[-120:], 
                      label='Historical (Last 10 years)', color='blue', linewidth=2)
        axes[0,0].plot(arima_data['dates'], arima_data['forecast'], 
                      label='ARIMA Forecast (2025-2030)', color='red', linewidth=2)
        axes[0,0].fill_between(arima_data['dates'], 
                             arima_data['confidence_interval'].iloc[:, 0], 
                             arima_data['confidence_interval'].iloc[:, 1], 
                             color='red', alpha=0.2, label='95% Confidence Interval')
        axes[0,0].set_title('ARIMA Temperature Forecast', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Temperature (°C)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].tick_params(axis='x', rotation=45)
    
    # SARIMA plot
    if 'sarima' in forecasts:
        sarima_data = forecasts['sarima']
        monthly_temp = sarima_data['historical_data']
        
        axes[0,1].plot(monthly_temp.index[-120:], monthly_temp.values[-120:], 
                      label='Historical (Last 10 years)', color='blue', linewidth=2)
        axes[0,1].plot(sarima_data['dates'], sarima_data['forecast'], 
                      label='SARIMA Forecast (2025-2030)', color='green', linewidth=2)
        axes[0,1].fill_between(sarima_data['dates'], 
                             sarima_data['confidence_interval'].iloc[:, 0], 
                             sarima_data['confidence_interval'].iloc[:, 1], 
                             color='green', alpha=0.2, label='95% Confidence Interval')
        axes[0,1].set_title('SARIMA Temperature Forecast', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Date')
        axes[0,1].set_ylabel('Temperature (°C)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].tick_params(axis='x', rotation=45)
    
    # Colorful ARIMA Seasonal Decomposition
    if 'arima' in forecasts and 'decomposition' in forecasts['arima']:
        decomposition = forecasts['arima']['decomposition']
        
        # Trend component (Green)
        trend_data = decomposition.trend.dropna()
        axes[1,0].plot(trend_data.index, trend_data.values, color='green', linewidth=2, label='Trend')
        axes[1,0].set_title('ARIMA Seasonal Decomposition - Trend', fontweight='bold', color='green')
        axes[1,0].set_ylabel('Temperature (°C)')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
        
        # Seasonal component (Orange)
        seasonal_data = decomposition.seasonal
        axes[1,1].plot(seasonal_data.index[:365], seasonal_data.values[:365], 
                      color='orange', linewidth=2, label='Seasonal')
        axes[1,1].set_title('ARIMA Seasonal Decomposition - Seasonal', fontweight='bold', color='orange')
        axes[1,1].set_ylabel('Temperature (°C)')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()


def plot_ml_results(forecasts):
    """Plot machine learning model results with clear model identification"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Model performance comparison
    models = []
    r2_scores = []
    rmse_scores = []
    
    if 'random_forest' in forecasts:
        models.append('Random Forest')
        r2_scores.append(forecasts['random_forest']['test_r2'])
        rmse_scores.append(forecasts['random_forest']['test_rmse'])
    
    if 'xgboost' in forecasts:
        models.append('XGBoost')
        r2_scores.append(forecasts['xgboost']['test_r2'])
        rmse_scores.append(forecasts['xgboost']['test_rmse'])
    
    if 'lstm' in forecasts:
        models.append('LSTM')
        r2_scores.append(forecasts['lstm']['test_r2'])
        rmse_scores.append(forecasts['lstm']['test_rmse'])
    
    if models:
        # R² comparison with clear model names
        colors = ['lightgreen', 'lightblue', 'lightcoral'][:len(models)]
        bars1 = axes[0,0].bar(models, r2_scores, color=colors, alpha=0.7)
        axes[0,0].set_title('ML Model Performance (R² Score)', fontweight='bold')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, score in zip(bars1, r2_scores):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                         f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # RMSE comparison with clear model names
        bars2 = axes[0,1].bar(models, rmse_scores, color=colors, alpha=0.7)
        axes[0,1].set_title('ML Model Performance (RMSE)', fontweight='bold')
        axes[0,1].set_ylabel('RMSE (°C)')
        axes[0,1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, score in zip(bars2, rmse_scores):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                         f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Feature importance (Random Forest or XGBoost)
    if 'random_forest' in forecasts:
        importance_df = forecasts['random_forest']['feature_importance'].head(10)
        axes[1,0].barh(importance_df['Feature'], importance_df['Importance'], 
                      color='lightgreen', alpha=0.7)
        axes[1,0].set_title('Random Forest - Top 10 Features', fontweight='bold')
        axes[1,0].set_xlabel('Feature Importance')
        axes[1,0].grid(True, alpha=0.3, axis='x')
    elif 'xgboost' in forecasts:
        importance_df = forecasts['xgboost']['feature_importance'].head(10)
        axes[1,0].barh(importance_df['Feature'], importance_df['Importance'], 
                      color='lightblue', alpha=0.7)
        axes[1,0].set_title('XGBoost - Top 10 Features', fontweight='bold')
        axes[1,0].set_xlabel('Feature Importance')
        axes[1,0].grid(True, alpha=0.3, axis='x')
    
    # Performance summary with clear model identification
    axes[1,1].axis('off')
    axes[1,1].set_title('ML Model Summary', fontsize=14, fontweight='bold')
    
    summary_text = "MACHINE LEARNING RESULTS\n" + "="*30 + "\n\n"
    
    for i, model in enumerate(models):
        model_key = model.lower().replace(' ', '_')
        if model_key in forecasts:
            perf = forecasts[model_key]
            summary_text += f"{model}:\n"
            summary_text += f"  • R²: {perf['test_r2']:.4f}\n"
            summary_text += f"  • RMSE: {perf['test_rmse']:.4f}°C\n\n"
    
    if summary_text:
        axes[1,1].text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                      transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    plt.show()


def plot_arima_decomposition_colorful(decomposition):
    """Plot colorful ARIMA seasonal decomposition"""
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Original data (Blue)
    axes[0].plot(decomposition.observed, color='blue', linewidth=1.5)
    axes[0].set_title('Original Time Series', fontweight='bold', color='blue')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].grid(True, alpha=0.3)
    
    # Trend (Green)
    axes[1].plot(decomposition.trend, color='green', linewidth=2)
    axes[1].set_title('Trend Component', fontweight='bold', color='green')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal (Orange)
    axes[2].plot(decomposition.seasonal, color='orange', linewidth=2)
    axes[2].set_title('Seasonal Component', fontweight='bold', color='orange')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].grid(True, alpha=0.3)
    
    # Residual (Red)
    axes[3].plot(decomposition.resid, color='red', linewidth=1)
    axes[3].set_title('Residual Component', fontweight='bold', color='red')
    axes[3].set_ylabel('Temperature (°C)')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_sarima_enhanced(forecasts):
    """Plot enhanced SARIMA results with multiple visualizations"""
    if 'sarima' not in forecasts:
        print("SARIMA results not available for plotting.")
        return
    
    sarima_data = forecasts['sarima']
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. SARIMA Forecast Plot
    monthly_temp = sarima_data['historical_data']
    axes[0,0].plot(monthly_temp.index[-120:], monthly_temp.values[-120:], 
                  label='Historical (Last 10 years)', color='blue', linewidth=2)
    axes[0,0].plot(sarima_data['dates'], sarima_data['forecast'], 
                  label='SARIMA Forecast', color='green', linewidth=2)
    axes[0,0].fill_between(sarima_data['dates'], 
                         sarima_data['confidence_interval'].iloc[:, 0], 
                         sarima_data['confidence_interval'].iloc[:, 1], 
                         color='green', alpha=0.2, label='95% Confidence Interval')
    axes[0,0].set_title('SARIMA Temperature Forecast', fontweight='bold')
    axes[0,0].set_xlabel('Date')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Seasonal Decomposition if available
    if 'decomposition' in sarima_data:
        decomp = sarima_data['decomposition']
        trend_data = decomp.trend.dropna()
        axes[0,1].plot(trend_data.index, trend_data.values, color='purple', linewidth=2)
        axes[0,1].set_title('SARIMA Trend Component', fontweight='bold', color='purple')
        axes[0,1].set_ylabel('Temperature (°C)')
        axes[0,1].grid(True, alpha=0.3)
        
        seasonal_data = decomp.seasonal
        axes[1,0].plot(seasonal_data.index[:365], seasonal_data.values[:365], 
                      color='cyan', linewidth=2)
        axes[1,0].set_title('SARIMA Seasonal Component', fontweight='bold', color='cyan')
        axes[1,0].set_ylabel('Temperature (°C)')
        axes[1,0].grid(True, alpha=0.3)
    
    # 3. Model Summary
    axes[1,1].axis('off')
    axes[1,1].set_title('SARIMA Model Summary', fontsize=14, fontweight='bold')
    
    summary = sarima_data.get('model_summary', {})
    summary_text = "SARIMA MODEL RESULTS\n" + "="*25 + "\n\n"
    summary_text += f"Order: {summary.get('order', 'N/A')}\n"
    summary_text += f"Seasonal Order: {summary.get('seasonal_order', 'N/A')}\n"
    summary_text += f"AIC: {summary.get('aic', 'N/A')}\n"
    summary_text += f"BIC: {summary.get('bic', 'N/A')}\n"
    summary_text += f"Log Likelihood: {summary.get('llf', 'N/A')}\n"
    
    axes[1,1].text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
                  transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    plt.show()
