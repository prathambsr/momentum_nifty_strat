==============================================================================
NIFTY 50 OHLC MASTER FILE - COMPLETE DATASET
==============================================================================

File: nifty50_ohlc_master_complete.csv

This is a comprehensive master file combining ALL Nifty 50 OHLC data from 
all periods (2016-2025) into a single, unified dataset.

==============================================================================
DATASET STATISTICS
==============================================================================

Total Records:             112,308
Date Range:                2016-04-01 to 2025-09-26
Trading Days:              2,243
Unique Symbols (all time): 77
Format:                    DATE,SYMBOL,OPEN,HIGH,LOW,CLOSE,VOLUME,ISIN

==============================================================================
DATA FORMAT
==============================================================================

Column Name    Description
-----------    ----------------------------------------------------------
DATE           Trading date in YYYY-MM-DD format
SYMBOL         Stock symbol/ticker
OPEN           Opening price for the day
HIGH           Highest price during the day
LOW            Lowest price during the day
CLOSE          Closing price for the day
VOLUME         Total traded quantity
ISIN           International Securities Identification Number

==============================================================================
SOURCE FILES COMBINED
==============================================================================

This master file combines data from 17 individual period files:

Unified Files (single constituent list):
  - nifty50_ohlc_2016.csv         (9,300 records)
  - nifty50_ohlc_2021.csv         (12,050 records)
  - nifty50_ohlc_2023.csv         (9,820 records)

Split Files (mid-year constituent changes):
  - nifty50_ohlc_2017_H1.csv      (6,222 records)
  - nifty50_ohlc_2017_H2.csv      (6,375 records)
  - nifty50_ohlc_2018_H1.csv      (6,200 records)
  - nifty50_ohlc_2018_H2.csv      (6,050 records)
  - nifty50_ohlc_2019_H1.csv      (6,100 records)
  - nifty50_ohlc_2019_H2.csv      (6,150 records)
  - nifty50_ohlc_2020_H1.csv      (6,150 records)
  - nifty50_ohlc_2020_H2.csv      (6,441 records)
  - nifty50_ohlc_2022_H1.csv      (5,700 records)
  - nifty50_ohlc_2022_H2.csv      (5,400 records)
  - nifty50_ohlc_2024_H1.csv      (5,050 records)
  - nifty50_ohlc_2024_H2.csv      (6,300 records)
  - nifty50_ohlc_2025_H1.csv      (6,150 records)
  - nifty50_ohlc_2025_H2.csv      (3,100 records - partial)

Duplicates Removed:           196
NaN Records Removed:          54
Final Record Count:           112,308

==============================================================================
DATA QUALITY
==============================================================================

[✓] Sorted by DATE and SYMBOL
[✓] Duplicates removed
[✓] NaN values filtered out
[✓] Only EQ (Equity) series stocks included
[✓] Consistent date format (YYYY-MM-DD)
[✓] All records validated

==============================================================================
RECORDS BY YEAR
==============================================================================

Year    Records    Unique Symbols
------  ---------  ---------------
2016    9,300      50
2017    12,597     54
2018    12,250     53
2019    12,250     51
2020    12,591     51
2021    12,000     50
2022    11,000     51
2023    9,770      50
2024    11,300     51
2025    9,250      52 (partial - up to September)

Note: Total unique symbols is 77 because the Nifty 50 composition changed
over the years. Some stocks were added/removed from the index.

==============================================================================
USAGE EXAMPLES
==============================================================================

Python (pandas):
----------------
import pandas as pd

# Load the master file
df = pd.read_csv('nifty50_ohlc_master_complete.csv')

# Convert date to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Filter for a specific stock
reliance = df[df['SYMBOL'] == 'RELIANCE']

# Filter for a specific year
df_2024 = df[df['DATE'].dt.year == 2024]

# Calculate daily returns
df['RETURN'] = df.groupby('SYMBOL')['CLOSE'].pct_change()

# Get all unique symbols that were ever in Nifty 50
all_symbols = df['SYMBOL'].unique()
print(f"Total unique symbols: {len(all_symbols)}")

R:
---
library(readr)
library(dplyr)

# Load the master file
df <- read_csv('nifty50_ohlc_master_complete.csv')

# Filter for specific stock
reliance <- df %>% filter(SYMBOL == 'RELIANCE')

# Calculate returns
df <- df %>%
  group_by(SYMBOL) %>%
  mutate(RETURN = (CLOSE - lag(CLOSE)) / lag(CLOSE))

==============================================================================
DATA COVERAGE NOTES
==============================================================================

1. Coverage Period:
   - Starts: April 1, 2016
   - Ends: September 26, 2025
   - Total: ~9.5 years of data

2. Constituent Changes:
   - The file includes stocks that were in Nifty 50 during ANY period
   - Some stocks appear only in certain years (added/removed from index)
   - Total of 77 unique symbols across all time periods

3. Partial Data:
   - 2016: Starts from April (not full year)
   - 2025: Only up to September (not full year)

4. Trading Days:
   - Total: 2,243 unique trading days
   - Excludes weekends and holidays
   - May have gaps due to market closures

==============================================================================
REGENERATION
==============================================================================

To regenerate this master file from individual period files:

    python combine_all_ohlc_datasets.py

This will:
1. Read all period files from nifty50_ohlc_datasets/
2. Combine them into a single dataframe
3. Sort by date and symbol
4. Remove duplicates
5. Clean NaN values
6. Save as nifty50_ohlc_master_complete.csv

==============================================================================
FILE INFORMATION
==============================================================================

Generated: October 8, 2025
Script: combine_all_ohlc_datasets.py
Source: NSE Bhavcopy files (samco_bhav_data/raw/)
Format: CSV (UTF-8)
Size: ~112,000+ rows × 8 columns

==============================================================================
CONTACT & SUPPORT
==============================================================================

For questions about:
- Data accuracy: Verify against original NSE bhavcopy files
- Missing data: Check if trading occurred on that date
- Symbol changes: NSE sometimes changes stock symbols/ISINs

==============================================================================

