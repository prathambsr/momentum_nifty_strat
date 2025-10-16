"""
Main execution script for Stock Selection Strategy Analysis
Runs complete simulation from April 2016 through September 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import *
from helper_functions import *

def main():
    """Main execution function"""
    
    # Create output directories
    create_output_directories()
    print_separator("STOCK SELECTION STRATEGY ANALYSIS")
    print(f"Data Source: {DATA_FILE}")
    print(f"Analysis Period: April 2016 - September 2025\n")
    
    # ==================== LOAD DATA ====================
    print_separator("LOADING DATA")
    daily_df = load_data(DATA_FILE)
    print(f"✅ Loaded {len(daily_df):,} rows of data")
    print(f"   Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    print(f"   Unique tickers: {daily_df['ticker'].nunique()}")
    
    # ==================== APRIL 2016 ANALYSIS ====================
    print_separator("APRIL 2016: VALIDATION PERIOD")
    
    # First week performance
    april_first_week = get_first_week_performance(daily_df, APRIL_2016_YEAR, APRIL_2016_MONTH)
    april_top10 = april_first_week.head(TOP_N_SELECTIONS['top10'])['ticker'].tolist()
    april_top25 = april_first_week.head(TOP_N_SELECTIONS['top25'])['ticker'].tolist()
    print(f"✅ First week analysis: {len(april_first_week)} stocks")
    print(f"   Top 10: {', '.join(april_top10[:5])}...")
    
    # Full month performance
    april_full_month = get_full_month_performance(daily_df, APRIL_2016_YEAR, APRIL_2016_MONTH)
    april_results = april_first_week.merge(april_full_month, on='ticker', how='outer')
    april_results.to_csv(APRIL_2016_PERFORMANCE_CSV, index=False)
    print(f"✅ Saved: {APRIL_2016_PERFORMANCE_CSV}")
    
    # ==================== MAY 2016 ANALYSIS ====================
    print_separator("MAY 2016: COMBINED SCORING VALIDATION")
    
    # First week performance
    may_first_week = get_first_week_performance(daily_df, MAY_2016_YEAR, MAY_2016_MONTH)
    may_first_week.to_csv(MAY_2016_FIRST_WEEK_CSV, index=False)
    print(f"✅ May first week analysis: {len(may_first_week)} stocks")
    
    # Combined scoring (50% April full month + 50% May first week)
    combined = april_full_month.merge(may_first_week, on='ticker', how='inner')
    combined['combined_score'] = (combined['full_month_return'] * 0.5) + (combined['first_week_return'] * 0.5)
    combined = combined.sort_values('combined_score', ascending=False).reset_index(drop=True)
    
    combined_top10 = combined.head(TOP_N_SELECTIONS['top10'])['ticker'].tolist()
    combined_top25 = combined.head(TOP_N_SELECTIONS['top25'])['ticker'].tolist()
    
    combined.to_csv(MAY_2016_COMBINED_SCORES_CSV, index=False)
    print(f"✅ Combined scoring: {len(combined)} stocks")
    print(f"   Selected Top 10: {', '.join(combined_top10[:5])}...")
    print(f"✅ Saved: {MAY_2016_COMBINED_SCORES_CSV}")
    
    # ==================== MAIN SIMULATION: 2017-2025 ====================
    print_separator("4-COMPONENT STRATEGY SIMULATION")
    print(f"Period: {MONTH_NAMES[SIMULATION_START_MONTH]} {SIMULATION_START_YEAR} → {MONTH_NAMES[SIMULATION_END_MONTH]} {SIMULATION_END_YEAR}")
    print(f"Weight Distribution: {WEIGHTS['yearly']*100:.0f}% Yearly + {WEIGHTS['quarterly']*100:.0f}% Quarterly + {WEIGHTS['monthly']*100:.0f}% Monthly + {WEIGHTS['weekly']*100:.0f}% Weekly")
    print(f"Trading Period: Day {TRADING_START_DAY} → Day {TRADING_END_DAY} of next month\n")
    
    # Generate list of (year, month) tuples
    simulation_months = []
    for year in range(SIMULATION_START_YEAR, SIMULATION_END_YEAR + 1):
        start_m = SIMULATION_START_MONTH if year == SIMULATION_START_YEAR else 1
        end_m = SIMULATION_END_MONTH if year == SIMULATION_END_YEAR else 12
        for month in range(start_m, end_m + 1):
            simulation_months.append((year, month))
    
    print(f"Total periods to simulate: {len(simulation_months)}")
    print_separator()
    
    # Storage for all period results
    all_period_results = []
    all_comprehensive_data = []
    
    # Run simulation for each month
    for idx, (year, month) in enumerate(simulation_months, 1):
        month_name = MONTH_NAMES[month]
        print(f"\n[{idx}/{len(simulation_months)}] Processing {month_name} {year}...")
        
        try:
            # 1. Calculate combined scores for this month
            combined_scores = calculate_combined_score_for_month(daily_df, year, month)
            
            # 2. Select Top 10 and Top 25
            top10_tickers = combined_scores.head(TOP_N_SELECTIONS['top10'])['ticker'].tolist()
            top25_tickers = combined_scores.head(TOP_N_SELECTIONS['top25'])['ticker'].tolist()
            
            # 3. Get trading period dates
            trading_start, trading_end = get_trading_period_dates(year, month)
            
            # 4. Track daily performance for trading period
            top10_daily = track_daily_performance(daily_df, top10_tickers, trading_start, trading_end)
            top25_daily = track_daily_performance(daily_df, top25_tickers, trading_start, trading_end)
            
            # Get all available tickers for Nifty 50 baseline
            all_tickers = combined_scores['ticker'].tolist()
            nifty50_daily = track_daily_performance(daily_df, all_tickers, trading_start, trading_end)
            
            # Calculate average returns
            if len(top10_daily) > 0:
                top10_daily['avg_return'] = top10_daily[top10_tickers].mean(axis=1)
                top10_final_return = top10_daily['avg_return'].iloc[-1] if len(top10_daily) > 0 else 0
            else:
                top10_final_return = 0
                
            if len(top25_daily) > 0:
                top25_daily['avg_return'] = top25_daily[top25_tickers].mean(axis=1)
                top25_final_return = top25_daily['avg_return'].iloc[-1] if len(top25_daily) > 0 else 0
            else:
                top25_final_return = 0
                
            if len(nifty50_daily) > 0:
                nifty50_daily['avg_return'] = nifty50_daily[[c for c in nifty50_daily.columns if c != 'date']].mean(axis=1)
                nifty50_final_return = nifty50_daily['avg_return'].iloc[-1] if len(nifty50_daily) > 0 else 0
            else:
                nifty50_final_return = 0
            
            # 5. Save monthly outputs
            score_filename = MONTHLY_SCORES_DIR / f'{year}_{month:02d}_{month_name}_scores.csv'
            combined_scores.to_csv(score_filename, index=False)
            
            tracking_filename = MONTHLY_TRACKING_DIR / f'{year}_{month:02d}_{month_name}_tracking.csv'
            tracking_df = top10_daily[['date', 'avg_return']].copy()
            tracking_df.columns = ['date', 'top10_return']
            tracking_df['top25_return'] = top25_daily['avg_return'].values if len(top25_daily) > 0 else 0
            tracking_df['nifty50_return'] = nifty50_daily['avg_return'].values if len(nifty50_daily) > 0 else 0
            tracking_df.to_csv(tracking_filename, index=False)
            
            # 6. Store results for consolidated summary
            period_result = {
                'year': year,
                'month': month,
                'month_name': month_name,
                'trading_start': trading_start,
                'trading_end': trading_end,
                'trading_days': len(top10_daily),
                'top10_return': top10_final_return,
                'top25_return': top25_final_return,
                'nifty50_return': nifty50_final_return,
                'top10_outperformance': top10_final_return - nifty50_final_return,
                'top25_outperformance': top25_final_return - nifty50_final_return
            }
            all_period_results.append(period_result)
            
            # Print summary
            print(f"   ✅ Top 10: {top10_final_return:+.2f}% | Top 25: {top25_final_return:+.2f}% | Nifty: {nifty50_final_return:+.2f}%")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            continue
    
    print_separator("SIMULATION COMPLETE")
    print(f"Processed {len(all_period_results)} periods successfully")
    
    # ==================== CONSOLIDATED SUMMARY ====================
    print_separator("CREATING CONSOLIDATED OUTPUTS")
    
    # Create consolidated DataFrame
    consolidated_df = pd.DataFrame(all_period_results)
    consolidated_df['period'] = consolidated_df['year'].astype(str) + '-' + consolidated_df['month'].astype(str).str.zfill(2) + ' ' + consolidated_df['month_name']
    
    # Reorder columns
    summary_columns = [
        'period', 'year', 'month', 'month_name',
        'trading_start', 'trading_end', 'trading_days',
        'top10_return', 'top25_return', 'nifty50_return',
        'top10_outperformance', 'top25_outperformance'
    ]
    consolidated_df = consolidated_df[summary_columns]
    
    # Save consolidated summary
    consolidated_df.to_csv(CONSOLIDATED_SUMMARY_CSV, index=False)
    print(f"✅ Saved consolidated summary: {CONSOLIDATED_SUMMARY_CSV}")
    
    # ==================== YEARLY SUMMARY ====================
    
    # Group by year and calculate statistics
    yearly_summary = consolidated_df.groupby('year').agg({
        'top10_return': ['mean', 'std', 'min', 'max', 'count'],
        'top25_return': ['mean', 'std', 'min', 'max'],
        'nifty50_return': ['mean', 'std', 'min', 'max'],
        'top10_outperformance': ['mean', 'sum'],
        'top25_outperformance': ['mean', 'sum']
    }).round(2)
    
    # Flatten column names
    yearly_summary.columns = ['_'.join(col).strip() for col in yearly_summary.columns.values]
    yearly_summary = yearly_summary.reset_index()
    
    # Calculate win rates per year
    win_rates = []
    for year in yearly_summary['year']:
        year_data = consolidated_df[consolidated_df['year'] == year]
        top10_win_rate = (year_data['top10_outperformance'] > 0).sum() / len(year_data) * 100
        top25_win_rate = (year_data['top25_outperformance'] > 0).sum() / len(year_data) * 100
        win_rates.append({'year': year, 'top10_win_rate_%': top10_win_rate, 'top25_win_rate_%': top25_win_rate})
    
    win_rates_df = pd.DataFrame(win_rates)
    yearly_summary = yearly_summary.merge(win_rates_df, on='year')
    
    # Save yearly summary
    yearly_summary.to_csv(YEARLY_SUMMARY_CSV, index=False)
    print(f"✅ Saved yearly summary: {YEARLY_SUMMARY_CSV}")
    
    # ==================== VISUALIZATIONS ====================
    print_separator("GENERATING VISUALIZATIONS")
    
    # Cumulative performance chart
    print("Creating cumulative performance chart...")
    create_cumulative_performance_chart(consolidated_df, CUMULATIVE_PERFORMANCE_PNG)
    print(f"✅ Saved: {CUMULATIVE_PERFORMANCE_PNG}")
    
    # Yearly comparison chart
    print("Creating yearly comparison chart...")
    create_yearly_comparison_chart(consolidated_df, YEARLY_COMPARISON_PNG)
    print(f"✅ Saved: {YEARLY_COMPARISON_PNG}")
    
    # Optional: Generate monthly charts
    if CREATE_MONTHLY_CHARTS or SPECIFIC_MONTHS_TO_CHART:
        print_separator("GENERATING MONTHLY CHARTS")
        months_to_chart = SPECIFIC_MONTHS_TO_CHART if SPECIFIC_MONTHS_TO_CHART else simulation_months
        
        for idx, (year, month) in enumerate(months_to_chart, 1):
            try:
                month_name = MONTH_NAMES[month]
                tracking_file = MONTHLY_TRACKING_DIR / f'{year}_{month:02d}_{month_name}_tracking.csv'
                tracking_data = pd.read_csv(tracking_file)
                tracking_data['date'] = pd.to_datetime(tracking_data['date'])
                
                chart_file = MONTHLY_CHARTS_DIR / f'{year}_{month:02d}_{month_name}_performance.png'
                create_monthly_chart(tracking_data, year, month, chart_file)
                print(f"✅ [{idx}/{len(months_to_chart)}] Created chart: {chart_file.name}")
                
            except Exception as e:
                print(f"❌ [{idx}/{len(months_to_chart)}] Error creating chart for {month_name} {year}: {str(e)}")
    
    # ==================== FINAL SUMMARY ====================
    print_separator("ANALYSIS COMPLETE")
    
    print("\n📊 PERFORMANCE SUMMARY:")
    print(f"   Total periods analyzed: {len(consolidated_df)}")
    print(f"   Date range: {consolidated_df['trading_start'].min().date()} to {consolidated_df['trading_end'].max().date()}")
    
    print(f"\n💰 CUMULATIVE RETURNS:")
    top10_cumulative = (1 + consolidated_df['top10_return'] / 100).prod() - 1
    top25_cumulative = (1 + consolidated_df['top25_return'] / 100).prod() - 1
    nifty50_cumulative = (1 + consolidated_df['nifty50_return'] / 100).prod() - 1
    print(f"   Top 10 Strategy:  {top10_cumulative * 100:+.2f}%")
    print(f"   Top 25 Strategy:  {top25_cumulative * 100:+.2f}%")
    print(f"   Nifty 50 Baseline: {nifty50_cumulative * 100:+.2f}%")
    
    print(f"\n📈 AVERAGE RETURNS PER PERIOD:")
    print(f"   Top 10:  {consolidated_df['top10_return'].mean():+.2f}%")
    print(f"   Top 25:  {consolidated_df['top25_return'].mean():+.2f}%")
    print(f"   Nifty 50: {consolidated_df['nifty50_return'].mean():+.2f}%")
    
    print(f"\n🎯 WIN RATE (Beating Nifty 50):")
    top10_wins = (consolidated_df['top10_outperformance'] > 0).sum()
    top25_wins = (consolidated_df['top25_outperformance'] > 0).sum()
    print(f"   Top 10: {top10_wins}/{len(consolidated_df)} ({top10_wins/len(consolidated_df)*100:.1f}%)")
    print(f"   Top 25: {top25_wins}/{len(consolidated_df)} ({top25_wins/len(consolidated_df)*100:.1f}%)")
    
    print(f"\n📁 OUTPUT FILES GENERATED:")
    print(f"   Excel Data: {EXCEL_OUTPUT_DIR}/")
    print(f"   Monthly Scores: {MONTHLY_SCORES_DIR}/")
    print(f"   Monthly Tracking: {MONTHLY_TRACKING_DIR}/")
    print(f"   Graphs: {GRAPHS_OUTPUT_DIR}/")
    
    print_separator("=" * SEPARATOR_LENGTH)
    print("✅ All analysis complete!")
    

if __name__ == "__main__":
    main()

