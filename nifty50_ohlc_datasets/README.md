# Nifty 50 OHLC Datasets

## Overview
This directory contains historical OHLC (Open, High, Low, Close) data for Nifty 50 constituent stocks. The data is organized intelligently based on constituent changes:
- **Unified files**: For years with no constituent changes (single constituent list)
- **Half-yearly files**: For years with mid-year constituent changes (H1: Jan-Jun, H2: Jul-Dec)

## Dataset Format
Each CSV file follows this standardized format:
```
DATE,SYMBOL,OPEN,HIGH,LOW,CLOSE,VOLUME,ISIN
```

### Column Descriptions
- **DATE**: Trading date in YYYY-MM-DD format
- **SYMBOL**: Stock symbol/ticker
- **OPEN**: Opening price
- **HIGH**: Highest price during the day
- **LOW**: Lowest price during the day
- **CLOSE**: Closing price
- **VOLUME**: Total traded quantity
- **ISIN**: International Securities Identification Number

## Data Coverage

### Available Datasets

#### Unified Files (No constituent changes)
| Year | Records | Trading Days | File |
|------|---------|--------------|------|
| 2016 | 9,300 | 186 | nifty50_ohlc_2016.csv |
| 2021 | 12,050 | 241 | nifty50_ohlc_2021.csv |
| 2023 | 9,820 | 196 | nifty50_ohlc_2023.csv |

#### Split Files (Mid-year constituent changes)
| Period | Records | Trading Days | File |
|--------|---------|--------------|------|
| 2017 H1 | 6,222 | 122 | nifty50_ohlc_2017_H1.csv |
| 2017 H2 | 6,375 | 125 | nifty50_ohlc_2017_H2.csv |
| 2018 H1 | 6,200 | 124 | nifty50_ohlc_2018_H1.csv |
| 2018 H2 | 6,050 | 121 | nifty50_ohlc_2018_H2.csv |
| 2019 H1 | 6,100 | 122 | nifty50_ohlc_2019_H1.csv |
| 2019 H2 | 6,150 | 123 | nifty50_ohlc_2019_H2.csv |
| 2020 H1 | 6,150 | 123 | nifty50_ohlc_2020_H1.csv |
| 2020 H2 | 6,441 | 129 | nifty50_ohlc_2020_H2.csv |
| 2022 H1 | 5,700 | 114 | nifty50_ohlc_2022_H1.csv |
| 2022 H2 | 5,400 | 108 | nifty50_ohlc_2022_H2.csv |
| 2024 H1 | 5,050 | 101 | nifty50_ohlc_2024_H1.csv |
| 2024 H2 | 6,300 | 126 | nifty50_ohlc_2024_H2.csv |
| 2025 H1 | 6,150 | 123 | nifty50_ohlc_2025_H1.csv |
| 2025 H2 | 3,100 | 62 | nifty50_ohlc_2025_H2.csv (partial) |

**Total**: 17 datasets covering 112,558 records

## Data Source
- **Raw Data**: NSE Bhavcopy files from `samco_bhav_data/raw/`
- **Constituents**: Historical Nifty 50 constituent lists (quarterly snapshots)
- **Extraction Date**: October 8, 2025

## Data Quality Notes
1. Only EQ (Equity) series stocks are included
2. Duplicates have been removed (by DATE and SYMBOL)
3. Data is sorted chronologically by DATE and then by SYMBOL
4. The constituent list for each period reflects the actual Nifty 50 composition at that time
5. 2016 data starts from April 1, 2016 (partial year)
6. 2025 H2 is partial data (up to September 2025)

## File Structure Logic

The script intelligently determines whether to create unified or split files:

### Unified Files
Created when constituents remained constant throughout the year:
- **2016**: Used Q1 constituents for entire year
- **2021**: Used Q1 constituents for entire year  
- **2023**: Used Q2 constituents for entire year

### Split Files (H1/H2)
Created when constituents changed mid-year:
- **H1** (First Half): January 1 - June 30 using Q1 constituents
- **H2** (Second Half): July 1 - December 31 using Q2 constituents

## Usage Examples

### Python (pandas)
```python
import pandas as pd

# Load a unified dataset
df_2021 = pd.read_csv('nifty50_ohlc_2021.csv')

# Load half-yearly data
df_2024_h1 = pd.read_csv('nifty50_ohlc_2024_H1.csv')
df_2024_h2 = pd.read_csv('nifty50_ohlc_2024_H2.csv')

# Combine half-yearly data if needed
df_2024_full = pd.concat([df_2024_h1, df_2024_h2])

# Convert date column to datetime
df_2021['DATE'] = pd.to_datetime(df_2021['DATE'])

# Filter for specific symbol
reliance = df_2021[df_2021['SYMBOL'] == 'RELIANCE']

# Calculate daily returns
df_2021['RETURN'] = df_2021.groupby('SYMBOL')['CLOSE'].pct_change()

# Get unique symbols for each period
symbols_2021 = df_2021['SYMBOL'].unique()
print(f"Nifty 50 constituents in 2021: {len(symbols_2021)}")
```

### R
```r
library(readr)
library(dplyr)

# Load a unified dataset
df_2021 <- read_csv('nifty50_ohlc_2021.csv')

# Filter for specific symbol
reliance <- df_2021 %>% filter(SYMBOL == 'RELIANCE')

# Calculate daily returns
df_2021 <- df_2021 %>%
  group_by(SYMBOL) %>%
  mutate(RETURN = (CLOSE - lag(CLOSE)) / lag(CLOSE))
```

## Generation Script

Data was extracted using `create_nifty50_ohlc_datasets.py`, which:
1. Analyzes constituent files to detect years with single vs multiple constituent lists
2. Automatically creates unified files for years with no constituent changes
3. Creates H1/H2 splits for years with mid-year constituent changes
4. Scans bhavcopy files for each period
5. Extracts OHLC data for constituent stocks only
6. Consolidates, deduplicates, and formats the data
7. Saves period-specific datasets

### Key Features
- **Intelligent period detection**: Automatically determines unified vs split based on available constituent files
- **Accurate constituent tracking**: Each period uses the correct Nifty 50 composition for that time
- **Data integrity**: Removes duplicates and validates data consistency
- **Comprehensive coverage**: Processes all available data from 2016 onwards

## Updates

To regenerate or update the datasets:
```bash
python create_nifty50_ohlc_datasets.py
```

The script will:
- Detect all constituent files automatically
- Create appropriate unified or split files
- Generate a summary report
- Overwrite existing files with fresh data

## Directory Contents

```
nifty50_ohlc_datasets/
├── README.md                          (this file)
├── extraction_summary_report.csv      (statistics summary)
├── nifty50_ohlc_2016.csv             (unified)
├── nifty50_ohlc_2017_H1.csv          (split)
├── nifty50_ohlc_2017_H2.csv          (split)
├── nifty50_ohlc_2018_H1.csv          (split)
├── nifty50_ohlc_2018_H2.csv          (split)
├── nifty50_ohlc_2019_H1.csv          (split)
├── nifty50_ohlc_2019_H2.csv          (split)
├── nifty50_ohlc_2020_H1.csv          (split)
├── nifty50_ohlc_2020_H2.csv          (split)
├── nifty50_ohlc_2021.csv             (unified)
├── nifty50_ohlc_2022_H1.csv          (split)
├── nifty50_ohlc_2022_H2.csv          (split)
├── nifty50_ohlc_2023.csv             (unified)
├── nifty50_ohlc_2024_H1.csv          (split)
├── nifty50_ohlc_2024_H2.csv          (split)
├── nifty50_ohlc_2025_H1.csv          (split)
└── nifty50_ohlc_2025_H2.csv          (split - partial)
```

---

**Generated by**: Nifty 50 OHLC Extractor  
**Team**: Quant Research  
**Date**: October 8, 2025  
**Version**: 2.0 (Intelligent Period Detection)
