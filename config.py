"""
Configuration file for Stock Selection Strategy Analysis
Contains all parameters, paths, and constants used in the analysis
"""

from pathlib import Path

# ==================== FILE PATHS ====================

# Input data
DATA_FILE = "data/nifty50_ohlc_master_complete.csv"

# Output directories
EXCEL_OUTPUT_DIR = Path("excel_data")
MONTHLY_SCORES_DIR = EXCEL_OUTPUT_DIR / "monthly_scores"
MONTHLY_TRACKING_DIR = EXCEL_OUTPUT_DIR / "monthly_tracking"
GRAPHS_OUTPUT_DIR = Path("strat_graphs")
MONTHLY_CHARTS_DIR = GRAPHS_OUTPUT_DIR / "monthly_charts"
RESULTS_DIR = Path("results/analysis")

# Output files
APRIL_2016_PERFORMANCE_CSV = EXCEL_OUTPUT_DIR / "april_2016_performance.csv"
MAY_2016_FIRST_WEEK_CSV = EXCEL_OUTPUT_DIR / "may_2016_first_week_performance.csv"
MAY_2016_COMBINED_SCORES_CSV = EXCEL_OUTPUT_DIR / "may_2016_combined_scores.csv"
COMPREHENSIVE_ALL_STOCKS_CSV = EXCEL_OUTPUT_DIR / "comprehensive_strategy_analysis_all_stocks.csv"
COMPREHENSIVE_2017_2025_CSV = EXCEL_OUTPUT_DIR / "comprehensive_strategy_analysis_2017_2025.csv"
CONSOLIDATED_SUMMARY_CSV = EXCEL_OUTPUT_DIR / "consolidated_all_periods_summary.csv"
YEARLY_SUMMARY_CSV = EXCEL_OUTPUT_DIR / "consolidated_yearly_summary.csv"
DAILY_TRACKING_CSV = EXCEL_OUTPUT_DIR / "daily_tracking_top10_top25_nifty50.csv"

# Graph files
CUMULATIVE_PERFORMANCE_PNG = GRAPHS_OUTPUT_DIR / "consolidated_cumulative_performance.png"
YEARLY_COMPARISON_PNG = GRAPHS_OUTPUT_DIR / "consolidated_yearly_comparison.png"
PORTFOLIO_COMPARISON_PNG = GRAPHS_OUTPUT_DIR / "portfolio_daily_comparison.png"

# ==================== STRATEGY PARAMETERS ====================

# Component weights for scoring (must sum to 1.0)
WEIGHTS = {
    'yearly': 0.25,      # 25% weight on yearly performance
    'quarterly': 0.25,   # 25% weight on quarterly performance
    'monthly': 0.25,     # 25% weight on monthly performance
    'weekly': 0.25       # 25% weight on weekly (first 7 days) performance
}

# Selection criteria
TOP_N_SELECTIONS = {
    'top10': 10,   # Top 10 stocks
    'top25': 25    # Top 25 stocks
}

# Trading period definition
TRADING_START_DAY = 8  # Trading starts on 8th of month
TRADING_END_DAY = 7    # Trading ends on 7th of next month

# First week definition
FIRST_WEEK_DAYS = 7  # First 7 trading days

# ==================== SIMULATION SETTINGS ====================

# April-May 2016 period (original strategy validation)
APRIL_2016_YEAR = 2016
APRIL_2016_MONTH = 4
MAY_2016_YEAR = 2016
MAY_2016_MONTH = 5

# Main simulation period (4-component strategy)
SIMULATION_START_YEAR = 2017
SIMULATION_START_MONTH = 1
SIMULATION_END_YEAR = 2025
SIMULATION_END_MONTH = 9

# Chart generation settings
CREATE_MONTHLY_CHARTS = False  # Set to True to generate all monthly charts
SPECIFIC_MONTHS_TO_CHART = [
    (2017, 1),   # January 2017
    (2018, 1),   # January 2018
    (2020, 3),   # March 2020
    (2024, 12),  # December 2024
    (2025, 9)    # September 2025
]

# ==================== CONSTANTS ====================

# Quarter definitions (month ranges)
QUARTERS = {
    1: (1, 3),   # Q1: Jan-Mar
    2: (4, 6),   # Q2: Apr-Jun
    3: (7, 9),   # Q3: Jul-Sep
    4: (10, 12)  # Q4: Oct-Dec
}

# Month names for output formatting
MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

# Performance categories
PERFORMANCE_CATEGORIES = {
    'excellent': 10.0,    # >= 10% return
    'good': 5.0,          # >= 5% return
    'positive': 0.0,      # >= 0% return
    'slight_loss': -5.0,  # >= -5% return
    'poor': float('-inf') # < -5% return
}

# ==================== VISUALIZATION SETTINGS ====================

# Chart colors
COLORS = {
    'top10': '#2E86AB',    # Blue
    'top25': '#27AE60',    # Green
    'nifty50': '#F39C12',  # Orange
    'zero_line': 'black',
    'grid': 'gray'
}

# Chart sizes
FIGURE_SIZES = {
    'large': (20, 10),
    'standard': (16, 9)
}

# DPI settings
DPI_HIGH = 300
DPI_STANDARD = 200

# ==================== DATA QUALITY SETTINGS ====================

# Minimum data requirements
MIN_TRADING_DAYS = 1  # Minimum trading days required in a period

# Data validation
VALIDATE_POSITIVE_PRICES = True  # Only include rows with open > 0 and close > 0

# ==================== DISPLAY SETTINGS ====================

# Print formatting
SEPARATOR_LENGTH = 100
SEPARATOR_CHAR = "="

# Decimal places for display
DECIMAL_PLACES = {
    'returns': 2,
    'scores': 2,
    'percentages': 1
}

# ==================== SPECIAL CASES ====================

# 2017 special case: Uses April-December 2016 for yearly component
YEAR_2017_YEARLY_PERIOD = {
    'year': 2016,
    'start_month': 4,
    'end_month': 12
}

# ==================== HELPER FUNCTIONS ====================

def create_output_directories():
    """Create all required output directories if they don't exist"""
    EXCEL_OUTPUT_DIR.mkdir(exist_ok=True)
    MONTHLY_SCORES_DIR.mkdir(exist_ok=True)
    MONTHLY_TRACKING_DIR.mkdir(exist_ok=True)
    GRAPHS_OUTPUT_DIR.mkdir(exist_ok=True)
    MONTHLY_CHARTS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

def print_separator(text=""):
    """Print a formatted separator line"""
    if text:
        print(f"\n{SEPARATOR_CHAR * SEPARATOR_LENGTH}")
        print(text)
        print(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    else:
        print(SEPARATOR_CHAR * SEPARATOR_LENGTH)

