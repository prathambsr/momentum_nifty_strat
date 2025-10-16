# Stock Selection Strategy - Organized Codebase

This repository contains a professional, modular implementation of a 4-component stock selection strategy for Nifty 50 stocks.

## Project Structure

```
yearly_strat_new/
├── config.py                          # Configuration parameters and constants
├── helper_functions.py                # Core analysis functions
├── main_script.py                            # Main execution script
├── main.ipynb            # Clean notebook for exploratory analysis
├── results/
│   └── analysis/
│       ├── assumptions.txt            # Strategy assumptions & methodology
│       └── report.txt                 # Performance findings & statistics
├── data/
│   └── nifty50_ohlc_master_complete.csv  # Input data
├── excel_data/                        # Output CSVs
│   ├── monthly_scores/                # Monthly stock scores (105 files)
│   ├── monthly_tracking/              # Daily tracking data (105 files)
│   └── *.csv                          # Consolidated results
├── strat_graphs/                      # Visualizations
│   ├── monthly_charts/                # Individual monthly charts
│   └── *.png                          # Summary charts
└── misc.ipynb # Original notebook (reference only)
```

## Quick Start

### Option 1: Run Complete Analysis

Execute the main script to run the full analysis:

```bash
python main_script.py
```

This will:
- Load data
- Run April-May 2016 validation
- Execute 105-month simulation (Jan 2017 - Sep 2025)
- Generate all CSVs and charts
- Print performance summary

### Option 2: Interactive Analysis

Open the clean notebook for exploratory analysis:

```bash
jupyter notebook main.ipynb
```

The notebook imports from `helper_functions.py` and `config.py` for clean, modular analysis.

## Configuration

Edit `config.py` to modify:
- File paths
- Strategy weights (default: 25% each component)
- Simulation date ranges
- Chart generation settings
- Selection criteria (Top 10, Top 25)

## Core Functions

All analysis functions are in `helper_functions.py`:

**Data Loading:**
- `load_data()` - Load and clean OHLC data

**Performance Calculation:**
- `get_first_week_performance()` - First 7 trading days returns
- `get_full_month_performance()` - Full month returns
- `get_yearly_performance()` - Yearly period returns
- `get_quarterly_performance()` - Quarterly returns

**Scoring & Selection:**
- `get_scoring_periods()` - Determine period logic
- `calculate_combined_score_for_month()` - 4-component scoring

**Trading & Tracking:**
- `get_trading_period_dates()` - Calculate trading dates
- `track_daily_performance()` - Daily return tracking

**Visualization:**
- `create_monthly_chart()` - Individual month charts
- `create_cumulative_performance_chart()` - Cumulative returns
- `create_yearly_comparison_chart()` - Year-over-year bars

## Strategy Methodology

### 4-Component Weighted Average (25% each):

1. **Yearly Performance** - Previous calendar year returns
2. **Quarterly Performance** - Most recent complete quarter
3. **Monthly Performance** - Previous month returns
4. **Weekly Performance** - First 7 trading days of current month

### Trading Period:
- Selection: Based on first 7 trading days
- Trading: Day 8 of month → Day 7 of next month

### Special Cases:
- 2017: Uses April-December 2016 for yearly component
- Quarterly component changes at quarter boundaries

## Documentation

- **`results/analysis/assumptions.txt`** - Detailed methodology, data sources, assumptions, and limitations
- **`results/analysis/report.txt`** - Comprehensive performance analysis with 105 months of results

## Key Results

**Cumulative Returns (Jan 2017 - Sep 2025):**
- Top 10 Strategy: +60.01%
- Top 25 Strategy: +80.79%
- Nifty 50 Baseline: +81.68%

**Win Rate:** 52.4% (Top 10 beats Nifty 50 in 55 out of 105 periods)

**Average Monthly Return:** +0.91% (Top 10) vs +0.47% (Nifty 50)

See `results/analysis/report.txt` for complete analysis.

## Output Files

**Monthly Outputs:**
- `excel_data/monthly_scores/*.csv` - Stock scores for each month
- `excel_data/monthly_tracking/*.csv` - Daily performance tracking

**Consolidated Results:**
- `consolidated_all_periods_summary.csv` - Period-by-period summary
- `consolidated_yearly_summary.csv` - Annual statistics
- `comprehensive_strategy_analysis_2017_2025.csv` - Detailed stock-level data

**Visualizations:**
- `consolidated_cumulative_performance.png` - Main performance chart
- `consolidated_yearly_comparison.png` - Year-over-year comparison
- `monthly_charts/*.png` - Individual monthly charts (optional)

## For Collaboration

### Junior Researchers:
- Start with `main.ipynb` to explore the data
- Modify parameters in `config.py` without touching core logic
- Run `main_script.py` to regenerate results after config changes
- Review `results/analysis/assumptions.txt` to understand methodology

### Senior Researchers:
- Core logic in `helper_functions.py` for review and enhancement
- All functions have docstrings explaining inputs/outputs
- Modular design allows easy testing of individual components
- Add new functions to `helper_functions.py` following existing patterns

## Requirements

```
pandas
numpy
matplotlib
```

Install with:
```bash
pip install pandas numpy matplotlib
```

## Notes

- Original notebook (`misc.ipynb`) preserved as reference
- No changes made to original notebook per user request
- All new code follows modular, maintainable design principles
- Transaction costs and taxes not included in analysis
- Past performance does not guarantee future results

## License

Research and educational purposes only. See disclaimer in `results/analysis/report.txt`.

