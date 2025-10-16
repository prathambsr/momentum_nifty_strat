"""
Helper functions for Stock Selection Strategy Analysis
Contains all core analysis functions for calculating returns, scoring, and tracking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from pathlib import Path
from config import *

warnings.filterwarnings('ignore')


# ==================== DATA LOADING ====================

def load_data(file_path):
    """
    Load OHLC data and prepare it for analysis
    
    Args:
        file_path: Path to CSV file with OHLC data
        
    Returns:
        DataFrame with cleaned and prepared data
    """
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter out invalid prices if validation is enabled
    if VALIDATE_POSITIVE_PRICES:
        df = df[(df['open'] > 0) & (df['close'] > 0)]
    
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
    df.rename(columns={'symbol': 'ticker'}, inplace=True)
    
    return df


# ==================== PERFORMANCE CALCULATION ====================

def get_first_week_performance(daily_df, year, month):
    """
    Calculate performance for first 7 trading days of a specific month
    
    Args:
        daily_df: DataFrame with daily OHLC data
        year: Year to analyze
        month: Month to analyze
        
    Returns:
        DataFrame with ticker and first_week_return
    """
    month_data = daily_df[(daily_df['date'].dt.year == year) & 
                          (daily_df['date'].dt.month == month)].copy()
    
    if len(month_data) == 0:
        return pd.DataFrame(columns=['ticker', 'first_week_return'])
    
    # Get unique trading dates and take first 7
    trading_days = sorted(month_data['date'].unique())
    if len(trading_days) < FIRST_WEEK_DAYS:
        first_week_days = trading_days
    else:
        first_week_days = trading_days[:FIRST_WEEK_DAYS]
    
    # Get opening price from first day
    first_day_data = month_data[month_data['date'] == first_week_days[0]][['ticker', 'open']]
    first_day_data.columns = ['ticker', 'start_open']
    
    # Get closing price from 7th day (or last available)
    last_day_data = month_data[month_data['date'] == first_week_days[-1]][['ticker', 'close']]
    last_day_data.columns = ['ticker', 'end_close']
    
    # Calculate first week performance
    merged = first_day_data.merge(last_day_data, on='ticker')
    merged['first_week_return'] = ((merged['end_close'] - merged['start_open']) / merged['start_open']) * 100
    
    return merged[['ticker', 'first_week_return']].sort_values('first_week_return', ascending=False)


def get_full_month_performance(daily_df, year, month):
    """
    Calculate performance for entire month (Day 1 to last trading day)
    
    Args:
        daily_df: DataFrame with daily OHLC data
        year: Year to analyze
        month: Month to analyze
        
    Returns:
        DataFrame with ticker and full_month_return
    """
    month_data = daily_df[(daily_df['date'].dt.year == year) & 
                          (daily_df['date'].dt.month == month)].copy()
    
    if len(month_data) == 0:
        return pd.DataFrame(columns=['ticker', 'full_month_return'])
    
    # Get all trading days in the month
    trading_days = sorted(month_data['date'].unique())
    first_day = trading_days[0]
    last_day = trading_days[-1]
    
    # Get opening price from first day
    first_day_data = month_data[month_data['date'] == first_day][['ticker', 'open']]
    first_day_data.columns = ['ticker', 'start_open']
    
    # Get closing price from last day
    last_day_data = month_data[month_data['date'] == last_day][['ticker', 'close']]
    last_day_data.columns = ['ticker', 'end_close']
    
    # Calculate full month performance
    merged = first_day_data.merge(last_day_data, on='ticker')
    merged['full_month_return'] = ((merged['end_close'] - merged['start_open']) / merged['start_open']) * 100
    
    return merged[['ticker', 'full_month_return']].sort_values('full_month_return', ascending=False)


def get_yearly_performance(daily_df, year, start_month, end_month):
    """
    Calculate performance for a yearly period from start_month to end_month
    
    Args:
        daily_df: DataFrame with daily OHLC data
        year: Year to analyze
        start_month: Starting month of period
        end_month: Ending month of period
        
    Returns:
        DataFrame with ticker and yearly_return
    """
    # Filter data for the specified year and month range
    year_data = daily_df[(daily_df['date'].dt.year == year) & 
                         (daily_df['date'].dt.month >= start_month) & 
                         (daily_df['date'].dt.month <= end_month)].copy()
    
    if len(year_data) == 0:
        return pd.DataFrame(columns=['ticker', 'yearly_return'])
    
    # Get all trading days in the period
    trading_days = sorted(year_data['date'].unique())
    first_day = trading_days[0]
    last_day = trading_days[-1]
    
    # Get opening price from first day
    first_day_data = year_data[year_data['date'] == first_day][['ticker', 'open']]
    first_day_data.columns = ['ticker', 'start_open']
    
    # Get closing price from last day
    last_day_data = year_data[year_data['date'] == last_day][['ticker', 'close']]
    last_day_data.columns = ['ticker', 'end_close']
    
    # Calculate yearly performance
    merged = first_day_data.merge(last_day_data, on='ticker')
    merged['yearly_return'] = ((merged['end_close'] - merged['start_open']) / merged['start_open']) * 100
    
    return merged[['ticker', 'yearly_return']].sort_values('yearly_return', ascending=False)


def get_quarterly_performance(daily_df, year, quarter):
    """
    Calculate performance for a specific quarter (Q1=1, Q2=2, Q3=3, Q4=4)
    
    Args:
        daily_df: DataFrame with daily OHLC data
        year: Year to analyze
        quarter: Quarter number (1-4)
        
    Returns:
        DataFrame with ticker and quarterly_return
    """
    start_month, end_month = QUARTERS[quarter]
    
    # Filter data for the specified quarter
    quarter_data = daily_df[(daily_df['date'].dt.year == year) & 
                            (daily_df['date'].dt.month >= start_month) & 
                            (daily_df['date'].dt.month <= end_month)].copy()
    
    if len(quarter_data) == 0:
        return pd.DataFrame(columns=['ticker', 'quarterly_return'])
    
    # Get all trading days in the quarter
    trading_days = sorted(quarter_data['date'].unique())
    first_day = trading_days[0]
    last_day = trading_days[-1]
    
    # Get opening price from first day
    first_day_data = quarter_data[quarter_data['date'] == first_day][['ticker', 'open']]
    first_day_data.columns = ['ticker', 'start_open']
    
    # Get closing price from last day
    last_day_data = quarter_data[quarter_data['date'] == last_day][['ticker', 'close']]
    last_day_data.columns = ['ticker', 'end_close']
    
    # Calculate quarterly performance
    merged = first_day_data.merge(last_day_data, on='ticker')
    merged['quarterly_return'] = ((merged['end_close'] - merged['start_open']) / merged['start_open']) * 100
    
    return merged[['ticker', 'quarterly_return']].sort_values('quarterly_return', ascending=False)


# ==================== SCORING LOGIC ====================

def get_scoring_periods(year, month):
    """
    Determine which periods to use for the 4-component scoring system
    
    Args:
        year: Year to analyze
        month: Month to analyze
        
    Returns:
        Dictionary with 'yearly', 'quarterly', 'monthly', 'weekly' period specifications
    """
    periods = {}
    
    # 1. YEARLY COMPONENT (25%)
    if year == 2017:
        # Special case: All of 2017 uses April 2016 - December 2016
        periods['yearly'] = YEAR_2017_YEARLY_PERIOD.copy()
    else:
        # 2018 onwards: Full previous calendar year
        periods['yearly'] = {'year': year - 1, 'start_month': 1, 'end_month': 12}
    
    # 2. QUARTERLY COMPONENT (25%)
    # Determine which quarter to use based on current month
    if month in [1, 2, 3]:  # Jan-Mar: use previous year Q4
        periods['quarterly'] = {'year': year - 1, 'quarter': 4}
    elif month in [4, 5, 6]:  # Apr-Jun: use current year Q1
        periods['quarterly'] = {'year': year, 'quarter': 1}
    elif month in [7, 8, 9]:  # Jul-Sep: use current year Q2
        periods['quarterly'] = {'year': year, 'quarter': 2}
    else:  # Oct-Dec: use current year Q3
        periods['quarterly'] = {'year': year, 'quarter': 3}
    
    # 3. MONTHLY COMPONENT (25%)
    # Use previous month
    if month == 1:  # January uses December of previous year
        periods['monthly'] = {'year': year - 1, 'month': 12}
    else:
        periods['monthly'] = {'year': year, 'month': month - 1}
    
    # 4. WEEKLY COMPONENT (25%)
    # First week of current month
    periods['weekly'] = {'year': year, 'month': month}
    
    return periods


def calculate_combined_score_for_month(daily_df, year, month):
    """
    Calculate combined score using 4 components (25% each):
    1. Yearly performance
    2. Quarterly performance
    3. Monthly performance (previous month)
    4. Weekly performance (first week of current month)
    
    Args:
        daily_df: DataFrame with daily OHLC data
        year: Year to analyze
        month: Month to analyze
        
    Returns:
        DataFrame with all component scores and combined score
    """
    # Get period specifications
    periods = get_scoring_periods(year, month)
    
    # 1. Calculate yearly performance (25%)
    yearly_perf = get_yearly_performance(
        daily_df, 
        periods['yearly']['year'], 
        periods['yearly']['start_month'], 
        periods['yearly']['end_month']
    )
    
    # 2. Calculate quarterly performance (25%)
    quarterly_perf = get_quarterly_performance(
        daily_df,
        periods['quarterly']['year'],
        periods['quarterly']['quarter']
    )
    
    # 3. Calculate monthly performance (25%)
    monthly_perf = get_full_month_performance(
        daily_df,
        periods['monthly']['year'],
        periods['monthly']['month']
    )
    monthly_perf.columns = ['ticker', 'monthly_return']
    
    # 4. Calculate weekly performance (25%)
    weekly_perf = get_first_week_performance(
        daily_df,
        periods['weekly']['year'],
        periods['weekly']['month']
    )
    weekly_perf.columns = ['ticker', 'weekly_return']
    
    # Merge all components
    combined = yearly_perf.merge(quarterly_perf, on='ticker', how='outer')
    combined = combined.merge(monthly_perf, on='ticker', how='outer')
    combined = combined.merge(weekly_perf, on='ticker', how='outer')
    
    # Fill NaN with 0 (in case a stock doesn't have data for all periods)
    combined = combined.fillna(0)
    
    # Calculate combined score (25% each component)
    combined['combined_score'] = (
        combined['yearly_return'] * WEIGHTS['yearly'] +
        combined['quarterly_return'] * WEIGHTS['quarterly'] +
        combined['monthly_return'] * WEIGHTS['monthly'] +
        combined['weekly_return'] * WEIGHTS['weekly']
    )
    
    # Add component ranks
    combined['yearly_rank'] = combined['yearly_return'].rank(ascending=False, method='min').astype(int)
    combined['quarterly_rank'] = combined['quarterly_return'].rank(ascending=False, method='min').astype(int)
    combined['monthly_rank'] = combined['monthly_return'].rank(ascending=False, method='min').astype(int)
    combined['weekly_rank'] = combined['weekly_return'].rank(ascending=False, method='min').astype(int)
    
    # Sort by combined score
    combined = combined.sort_values('combined_score', ascending=False).reset_index(drop=True)
    
    # Add metadata about periods used
    combined['selection_year'] = year
    combined['selection_month'] = month
    combined['yearly_period'] = f"{periods['yearly']['year']}-{periods['yearly']['start_month']:02d} to {periods['yearly']['year']}-{periods['yearly']['end_month']:02d}"
    combined['quarterly_period'] = f"{periods['quarterly']['year']}-Q{periods['quarterly']['quarter']}"
    monthly_month = periods['monthly']['month']
    combined['monthly_period'] = f"{MONTH_NAMES[monthly_month][:3]}-{str(periods['monthly']['year'])[2:]}"
    combined['weekly_period'] = f"{year}-{month:02d} Week1"
    
    return combined


# ==================== TRADING PERIOD ====================

def get_trading_period_dates(year, month):
    """
    Get start and end dates for trading period
    Trading period: 8th of current month to 7th of next month
    
    Args:
        year: Year
        month: Month
        
    Returns:
        Tuple of (start_date, end_date) as pandas Timestamps
    """
    # Start date: 8th of current month
    start_date = pd.Timestamp(year=year, month=month, day=TRADING_START_DAY)
    
    # End date: 7th of next month
    if month == 12:
        end_date = pd.Timestamp(year=year + 1, month=1, day=TRADING_END_DAY)
    else:
        end_date = pd.Timestamp(year=year, month=month + 1, day=TRADING_END_DAY)
    
    return start_date, end_date


def track_daily_performance(daily_df, tickers, start_date, end_date):
    """
    Track daily cumulative returns for selected stocks from start_date to end_date
    
    Args:
        daily_df: DataFrame with daily OHLC data
        tickers: List of ticker symbols to track
        start_date: Start date for tracking
        end_date: End date for tracking
        
    Returns:
        DataFrame with dates and returns for each ticker
    """
    # Filter data for the date range
    period_data = daily_df[(daily_df['date'] >= start_date) & 
                           (daily_df['date'] <= end_date)].copy()
    
    # Get all trading days in the period
    trading_days = sorted(period_data['date'].unique())
    
    # Calculate returns for each ticker
    results = {'date': trading_days}
    
    for ticker in tickers:
        ticker_data = period_data[period_data['ticker'] == ticker].sort_values('date')
        
        if len(ticker_data) == 0:
            results[ticker] = [np.nan] * len(trading_days)
            continue
        
        # Get starting price (open on first day or closest available)
        start_price = None
        for day in trading_days:
            day_data = ticker_data[ticker_data['date'] == day]
            if len(day_data) > 0:
                start_price = day_data.iloc[0]['open']
                break
        
        if start_price is None:
            results[ticker] = [np.nan] * len(trading_days)
            continue
        
        # Calculate cumulative returns for each day
        daily_returns = []
        for day in trading_days:
            day_data = ticker_data[ticker_data['date'] == day]
            if len(day_data) > 0:
                close_price = day_data.iloc[0]['close']
                ret = ((close_price - start_price) / start_price) * 100
                daily_returns.append(ret)
            else:
                # If no data for this day, use last available return
                daily_returns.append(daily_returns[-1] if daily_returns else 0)
        
        results[ticker] = daily_returns
    
    return pd.DataFrame(results)


# ==================== VISUALIZATION ====================

def create_monthly_chart(tracking_data, year, month, output_file):
    """
    Create performance chart for a single month
    
    Args:
        tracking_data: DataFrame with daily tracking data
        year: Year
        month: Month
        output_file: Path to save chart
    """
    month_name = MONTH_NAMES[month]
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['standard'])
    
    ax.plot(tracking_data['date'], tracking_data['top10_return'],
            label=f'Top 10 (Final: {tracking_data["top10_return"].iloc[-1]:+.2f}%)',
            color=COLORS['top10'], linewidth=3, marker='o', markersize=5, alpha=0.9, zorder=3)
    
    ax.plot(tracking_data['date'], tracking_data['top25_return'],
            label=f'Top 25 (Final: {tracking_data["top25_return"].iloc[-1]:+.2f}%)',
            color=COLORS['top25'], linewidth=3, marker='s', markersize=5, alpha=0.9, zorder=2)
    
    ax.plot(tracking_data['date'], tracking_data['nifty50_return'],
            label=f'Nifty 50 (Final: {tracking_data["nifty50_return"].iloc[-1]:+.2f}%)',
            color=COLORS['nifty50'], linewidth=2.5, marker='D', markersize=4, alpha=0.8, linestyle='--', zorder=1)
    
    # Add zero line
    ax.axhline(0, color=COLORS['zero_line'], linestyle='-', linewidth=1.5, alpha=0.4)
    
    # Styling
    trading_start, trading_end = get_trading_period_dates(year, month)
    ax.set_title(f'Portfolio Performance: {month_name} {year}\nTrading Period: {trading_start.date()} → {trading_end.date()}',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Cumulative Return (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95, shadow=True)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45, ha='right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:+.1f}%'))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI_STANDARD, bbox_inches='tight', facecolor='white')
    plt.close()


def create_cumulative_performance_chart(consolidated_df, output_file):
    """
    Create cumulative performance chart for all periods
    
    Args:
        consolidated_df: DataFrame with consolidated period results
        output_file: Path to save chart
    """
    # Calculate cumulative returns (compounding)
    consolidated_df = consolidated_df.copy()
    consolidated_df['top10_cumulative'] = (1 + consolidated_df['top10_return'] / 100).cumprod() - 1
    consolidated_df['top25_cumulative'] = (1 + consolidated_df['top25_return'] / 100).cumprod() - 1
    consolidated_df['nifty50_cumulative'] = (1 + consolidated_df['nifty50_return'] / 100).cumprod() - 1
    
    # Convert to percentage
    consolidated_df['top10_cumulative'] *= 100
    consolidated_df['top25_cumulative'] *= 100
    consolidated_df['nifty50_cumulative'] *= 100
    
    # Create period index for x-axis
    consolidated_df['period_index'] = range(len(consolidated_df))
    
    # Create the chart
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['large'])
    
    # Plot cumulative returns
    ax.plot(consolidated_df['period_index'], consolidated_df['top10_cumulative'],
            label=f'Top 10 Strategy (Final: {consolidated_df["top10_cumulative"].iloc[-1]:+.1f}%)',
            color=COLORS['top10'], linewidth=3, marker='o', markersize=3, alpha=0.9, zorder=3)
    
    ax.plot(consolidated_df['period_index'], consolidated_df['top25_cumulative'],
            label=f'Top 25 Strategy (Final: {consolidated_df["top25_cumulative"].iloc[-1]:+.1f}%)',
            color=COLORS['top25'], linewidth=3, marker='s', markersize=3, alpha=0.9, zorder=2)
    
    ax.plot(consolidated_df['period_index'], consolidated_df['nifty50_cumulative'],
            label=f'Nifty 50 Baseline (Final: {consolidated_df["nifty50_cumulative"].iloc[-1]:+.1f}%)',
            color=COLORS['nifty50'], linewidth=2.5, marker='D', markersize=2.5, alpha=0.8, linestyle='--', zorder=1)
    
    # Add zero line
    ax.axhline(0, color=COLORS['zero_line'], linestyle='-', linewidth=1.5, alpha=0.4)
    
    # Add year separators
    year_changes = consolidated_df[consolidated_df['month'] == 1]['period_index'].tolist()
    for idx in year_changes[1:]:
        ax.axvline(idx, color=COLORS['grid'], linestyle=':', linewidth=1, alpha=0.3)
    
    # Styling
    ax.set_title('Cumulative Portfolio Performance: January 2017 → September 2025\n4-Component Strategy (25% Yearly + 25% Quarterly + 25% Monthly + 25% Weekly)',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Cumulative Return (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trading Period', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    ax.legend(loc='upper left', fontsize=13, framealpha=0.95, shadow=True)
    
    # X-axis: Show year labels
    year_ticks = consolidated_df.groupby('year')['period_index'].first().tolist()
    year_labels = consolidated_df.groupby('year')['year'].first().tolist()
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels, rotation=0)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:+.0f}%'))
    
    # Add statistics box
    stats_text = (f'Total Periods: {len(consolidated_df)}\n'
                  f'Avg Return/Period:\n'
                  f'  Top 10: {consolidated_df["top10_return"].mean():+.2f}%\n'
                  f'  Top 25: {consolidated_df["top25_return"].mean():+.2f}%\n'
                  f'  Nifty: {consolidated_df["nifty50_return"].mean():+.2f}%\n'
                  f'\n'
                  f'Outperformance:\n'
                  f'  Top 10: {consolidated_df["top10_outperformance"].mean():+.2f}%/period')
    
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                     edgecolor='gray', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI_HIGH, bbox_inches='tight', facecolor='white')
    plt.close()


def create_yearly_comparison_chart(consolidated_df, output_file):
    """
    Create year-over-year comparison chart
    
    Args:
        consolidated_df: DataFrame with consolidated period results
        output_file: Path to save chart
    """
    # Calculate average returns per year
    yearly_avg = consolidated_df.groupby('year')[['top10_return', 'top25_return', 'nifty50_return']].mean().reset_index()
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['standard'])
    
    x = np.arange(len(yearly_avg))
    width = 0.25
    
    bars1 = ax.bar(x - width, yearly_avg['top10_return'], width, label='Top 10',
                   color=COLORS['top10'], edgecolor='black', linewidth=1.2, alpha=0.85)
    bars2 = ax.bar(x, yearly_avg['top25_return'], width, label='Top 25',
                   color=COLORS['top25'], edgecolor='black', linewidth=1.2, alpha=0.85)
    bars3 = ax.bar(x + width, yearly_avg['nifty50_return'], width, label='Nifty 50',
                   color=COLORS['nifty50'], edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:+.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # Add zero line
    ax.axhline(0, color=COLORS['zero_line'], linestyle='-', linewidth=2, alpha=0.6)
    
    # Styling
    ax.set_title('Year-over-Year Average Returns: 2017-2025\n4-Component Strategy Performance by Year',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Average Monthly Return (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(yearly_avg['year'], fontsize=12, fontweight='bold')
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:+.1f}%'))
    
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95, shadow=True)
    
    # Add statistics box
    overall_top10_avg = yearly_avg['top10_return'].mean()
    overall_top25_avg = yearly_avg['top25_return'].mean()
    overall_nifty_avg = yearly_avg['nifty50_return'].mean()
    best_year = yearly_avg.loc[yearly_avg['top10_return'].idxmax(), 'year']
    worst_year = yearly_avg.loc[yearly_avg['top10_return'].idxmin(), 'year']
    
    stats_text = (f'Overall Averages:\n'
                  f'Top 10: {overall_top10_avg:+.2f}%\n'
                  f'Top 25: {overall_top25_avg:+.2f}%\n'
                  f'Nifty: {overall_nifty_avg:+.2f}%\n'
                  f'\n'
                  f'Best Year: {int(best_year)}\n'
                  f'Worst Year: {int(worst_year)}')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                     edgecolor='gray', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI_HIGH, bbox_inches='tight', facecolor='white')
    plt.close()

