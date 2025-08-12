#!/usr/bin/env python3
"""
Technical Analysis Script
Reads BTCUSDT_1m.csv and adds RSI 14 indicator
"""

import pandas as pd
import talib
import sys
import os

def add_rsi_to_csv(csv_file: str = "BTCUSDT_1m.csv", rsi_length: int = 14):
    """
    Read CSV file and add RSI column
    
    Args:
        csv_file (str): Path to the CSV file
        rsi_length (int): RSI period length (default: 14)
    """
    try:
        # Check if file exists
        if not os.path.exists(csv_file):
            print(f"❌ Error: File '{csv_file}' not found!")
            print(f"📁 Current directory: {os.getcwd()}")
            print(f"📂 Available files: {[f for f in os.listdir('.') if f.endswith('.csv')]}")
            return None
        
        print(f"📖 Reading CSV file: {csv_file}")
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        print(f"✅ Successfully loaded {len(df)} rows")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Display first few rows
        print("\n📋 First 5 rows:")
        print(df.head())
        
        # Check if we have enough data for RSI calculation
        if len(df) < rsi_length:
            print(f"⚠️  Warning: Only {len(df)} rows available, need at least {rsi_length} for RSI calculation")
            return df
        
        # Add RSI column
        print(f"\n🧮 Calculating RSI with period {rsi_length}...")
        df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_length)
        
        # Display RSI statistics
        rsi_stats = df['rsi'].describe()
        print(f"\n📈 RSI Statistics:")
        print(f"   Mean: {rsi_stats['mean']:.2f}")
        print(f"   Min: {rsi_stats['min']:.2f}")
        print(f"   Max: {rsi_stats['max']:.2f}")
        print(f"   Std: {rsi_stats['std']:.2f}")
        
        # Show rows with RSI values
        print(f"\n📊 Latest 5 rows with RSI:")
        print(df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'rsi']].tail())
        
        # Save the updated DataFrame
        output_file = f"BTCUSDT_1m_with_rsi_{rsi_length}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Saved updated data to: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return None

def analyze_rsi_signals(df: pd.DataFrame):
    """
    Analyze RSI signals for overbought/oversold conditions
    """
    if 'rsi' not in df.columns:
        print("❌ RSI column not found in DataFrame")
        return
    
    print("\n🔍 RSI Signal Analysis:")
    
    # Get latest RSI value
    latest_rsi = df['rsi'].iloc[-1]
    print(f"   Current RSI: {latest_rsi:.2f}")
    
    # Overbought conditions (RSI > 70)
    overbought_count = len(df[df['rsi'] > 70])
    print(f"   Overbought signals (RSI > 70): {overbought_count}")
    
    # Oversold conditions (RSI < 30)
    oversold_count = len(df[df['rsi'] < 30])
    print(f"   Oversold signals (RSI < 30): {oversold_count}")
    
    # Current signal
    if latest_rsi > 70:
        print("   🟡 Current signal: OVERBOUGHT (consider selling)")
    elif latest_rsi < 30:
        print("   🟢 Current signal: OVERSOLD (consider buying)")
    else:
        print("   ⚪ Current signal: NEUTRAL")
    
    # Show recent overbought/oversold periods
    recent_signals = df[df['rsi'].notna()].tail(20)
    overbought_periods = recent_signals[recent_signals['rsi'] > 70]
    oversold_periods = recent_signals[recent_signals['rsi'] < 30]
    
    if len(overbought_periods) > 0:
        print(f"\n   📈 Recent overbought periods (last 20 rows):")
        for _, row in overbought_periods.iterrows():
            print(f"      {row['timestamp']}: RSI = {row['rsi']:.2f}")
    
    if len(oversold_periods) > 0:
        print(f"\n   📉 Recent oversold periods (last 20 rows):")
        for _, row in oversold_periods.iterrows():
            print(f"      {row['timestamp']}: RSI = {row['rsi']:.2f}")

def main():
    """Main function"""
    print("🚀 BTCUSDT Technical Analysis Script")
    print("=" * 50)
    
    # Process the CSV file
    df = add_rsi_to_csv()
    
    if df is not None:
        # Analyze RSI signals
        analyze_rsi_signals(df)
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Total rows processed: {len(df)}")
        print(f"📈 RSI values calculated: {df['rsi'].notna().sum()}")
    else:
        print("❌ Failed to process CSV file")

if __name__ == "__main__":
    main()
