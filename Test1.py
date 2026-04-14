import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# 1. Define Common Chinese Futures
CHINESE_FUTURES = {
    'Gold (沪金)': 'AU0',
    'Silver (沪银)': 'AG0',
    'Crude Oil (原油)': 'SC0',
    'Methanol (甲醇)': 'MA0',
    'Caustic Soda (烧碱)': 'SH0',
    'Glass (玻璃)': 'FG0',
    'Coking Coal (焦煤)': 'JM0',
    'Coke (焦炭)': 'J0'
}

# 2. Strict TDX Strategy Implementation
def apply_strict_tdx_strategy(df):
    df['Center_of_Gravity'] = (
        df['Close'] + 0.618 * df['Close'].shift(1) + 0.382 * df['Close'].shift(1) + 
        0.236 * df['Close'].shift(3) + 0.146 * df['Close'].shift(4)
    ) / 2.382
    
    def linear_slope(y):
        if len(y) < 22: return np.nan
        x = np.arange(1, len(y) + 1)
        return np.polyfit(x, y, 1)[0]
    
    df['Slope_22'] = df['Close'].rolling(window=22).apply(linear_slope, raw=True)
    df['Trading_Line_Base'] = (df['Slope_22'] * 20) + df['Close']
    df['Trading_Line'] = df['Trading_Line_Base'].ewm(span=55, adjust=False).mean()
    df['Golden_Line'] = np.where(df['Center_of_Gravity'] >= df['Trading_Line'], df['Trading_Line'], np.nan)
    df['Empty_Line'] = np.where(df['Center_of_Gravity'] < df['Trading_Line'], df['Trading_Line'], np.nan)
    df['Radar_Line'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['Radar_Up'] = np.where(df['Radar_Line'] > df['Radar_Line'].shift(1), df['Radar_Line'], np.nan)
    df['Radar_Down'] = np.where(df['Radar_Line'] < df['Radar_Line'].shift(1), df['Radar_Line'], np.nan)
    return df

# 3. Build the All-in-One Dashboard
def build_dashboard():
    end_date = datetime.today()
    start_date = end_date - relativedelta(years=3)
    
    fig = go.Figure()
    dropdown_buttons = []
    
    # We have 5 lines/traces per commodity
    traces_per_asset = 5 
    
    print("\nStarting data fetch. This will take about 10-15 seconds to pull all 8 assets...")
    
    # Loop through all commodities
    for i, (name, symbol) in enumerate(CHINESE_FUTURES.items()):
        print(f"Fetching {name}...")
        try:
            df = ak.futures_zh_daily_sina(symbol=symbol)
            column_mapping = {
                "日期": "Date", "date": "Date", "开盘价": "Open", "open": "Open",
                "收盘价": "Close", "close": "Close", "最高价": "High", "high": "High",
                "最低价": "Low", "low": "Low", "成交量": "Volume", "volume": "Volume"
            }
            df = df.rename(columns=column_mapping)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df = df[df.index >= pd.to_datetime(start_date)]
            
            df = apply_strict_tdx_strategy(df)
            
            # Visibility logic: Only the first asset is visible by default
            is_visible = (i == 0)
            
            # Add the 5 traces for this asset
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=f'{name} Price', increasing_line_color='red', decreasing_line_color='green', visible=is_visible))
            fig.add_trace(go.Scatter(x=df.index, y=df['Golden_Line'], line=dict(color='red', width=2), name='【黄金线】 Bullish', visible=is_visible))
            fig.add_trace(go.Scatter(x=df.index, y=df['Empty_Line'], line=dict(color='cyan', width=2), name='【空仓线】 Bearish', visible=is_visible))
            fig.add_trace(go.Scatter(x=df.index, y=df['Radar_Up'], line=dict(color='orange', width=1), name='【金】 Radar Up', visible=is_visible))
            fig.add_trace(go.Scatter(x=df.index, y=df['Radar_Down'], line=dict(color='blue', width=1), name='【空】 Radar Down', visible=is_visible))

            # Create the visibility array for the dropdown button
            # Example: If we are on the 2nd asset, make traces 5-9 True, and all others False
            visibility = [False] * (len(CHINESE_FUTURES) * traces_per_asset)
            for j in range(traces_per_asset):
                visibility[(i * traces_per_asset) + j] = True
                
            # Create the button for this asset
            dropdown_buttons.append(
                dict(label=name, method="update", args=[{"visible": visibility}, {"title": f"{name} - TDX Strategy Overlay"}])
            )
            
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")

    # Configure Layout with Web Interface elements
    fig.update_layout(
        title=f'{list(CHINESE_FUTURES.keys())[0]} - TDX Strategy Overlay',
        yaxis_title='Price (CNY)',
        template='plotly_dark',
        height=800,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        
        # 1. The Dropdown Selector (Top Left)
        updatemenus=[dict(
            active=0,
            buttons=dropdown_buttons,
            x=0.0,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )],
        
        # 2. The Time Span Quick Selectors & Range Slider (Bottom)
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    print("\nDashboard built! Opening in browser...")
    
    # Save to an actual HTML file and open it
    html_filename = "TDX_Futures_Dashboard.html"
    fig.write_html(html_filename)
    fig.show()
    print(f"File saved locally as: {html_filename}")

if __name__ == "__main__":
    build_dashboard()