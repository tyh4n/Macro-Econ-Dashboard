import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

ASSETS = {
    'Global Gold (COMEX)': ('INTL_FUTURE', 'GC'),
    'Global Oil (WTI)': ('INTL_FUTURE', 'CL'),
    'S&P 500 (SPY ETF)': ('US_STOCK', 'SPY'),
    'Gold (沪金)': ('CN_FUTURE', 'AU0'),
    'Silver (沪银)': ('CN_FUTURE', 'AG0'),
    'Crude Oil (原油)': ('CN_FUTURE', 'SC0'),
    'Methanol (甲醇)': ('CN_FUTURE', 'MA0'),
    'Caustic Soda (烧碱)': ('CN_FUTURE', 'SH0'),
    'Glass (玻璃)': ('CN_FUTURE', 'FG0'),
    'Coking Coal (焦煤)': ('CN_FUTURE', 'JM0'),
    'Coke (焦炭)': ('CN_FUTURE', 'J0')
}

# --- STRATEGY FUNCTIONS ---
def apply_strict_tdx_strategy(df):
    df['Center_of_Gravity'] = (df['Close'] + 0.618 * df['Close'].shift(1) + 0.382 * df['Close'].shift(1) + 0.236 * df['Close'].shift(3) + 0.146 * df['Close'].shift(4)) / 2.382
    def linear_slope(y):
        if len(y) < 22: return np.nan
        x = np.arange(1, len(y) + 1)
        return np.polyfit(x, y, 1)[0]
    df['Slope_22'] = df['Close'].rolling(window=22).apply(linear_slope, raw=True)
    df['Trading_Line'] = ((df['Slope_22'] * 20) + df['Close']).ewm(span=55, adjust=False).mean()
    df['Golden_Line'] = np.where(df['Center_of_Gravity'] >= df['Trading_Line'], df['Trading_Line'], np.nan)
    df['Empty_Line'] = np.where(df['Center_of_Gravity'] < df['Trading_Line'], df['Trading_Line'], np.nan)
    df['Radar_Line'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['Radar_Up'] = np.where(df['Radar_Line'] > df['Radar_Line'].shift(1), df['Radar_Line'], np.nan)
    df['Radar_Down'] = np.where(df['Radar_Line'] < df['Radar_Line'].shift(1), df['Radar_Line'], np.nan)
    return df

def apply_td_sequential(df):
    df['Setup_Up'] = 0
    df['Setup_Down'] = 0
    up_cond = df['Close'] > df['Close'].shift(4)
    down_cond = df['Close'] < df['Close'].shift(4)
    sup, sdown = 0, 0
    df['Countdown_Up'] = np.nan
    df['Countdown_Down'] = np.nan
    cup, cdown = 0, 0
    sup_act, sdown_act = False, False
    
    for i in range(len(df)):
        if up_cond.iloc[i]: sup += 1; sdown = 0
        elif down_cond.iloc[i]: sdown += 1; sup = 0
        else: sup, sdown = 0, 0
        df.iloc[i, df.columns.get_loc('Setup_Up')] = sup
        df.iloc[i, df.columns.get_loc('Setup_Down')] = sdown
        if sup == 9: sup_act, sdown_act = True, False; cup = 0 
        if sdown == 9: sdown_act, sup_act = True, False; cdown = 0 
        if i >= 2:
            if sup_act and df['Close'].iloc[i] >= df['High'].iloc[i-2]:
                cup += 1; df.iloc[i, df.columns.get_loc('Countdown_Up')] = cup
                if cup == 13: sup_act = False
            if sdown_act and df['Close'].iloc[i] <= df['Low'].iloc[i-2]:
                cdown += 1; df.iloc[i, df.columns.get_loc('Countdown_Down')] = cdown
                if cdown == 13: sdown_act = False
    return df

def apply_bollinger_bands(df, window=20, num_sd=2):
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    df['BB_Std'] = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * num_sd)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * num_sd)
    return df

def apply_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# --- ALERT SYSTEM LOGIC ---
def evaluate_alerts(df, asset_name):
    alerts = []
    # Get the last two days of data to detect crossovers
    today = df.iloc[-1]
    yesterday = df.iloc[-2]

    # 1. Trend Change (Golden Line / Empty Line Crossover)
    if pd.isna(yesterday['Golden_Line']) and not pd.isna(today['Golden_Line']):
        alerts.append(f"🟢 {asset_name}: Trend turned BULLISH (Golden Line crossover).")
    elif pd.isna(yesterday['Empty_Line']) and not pd.isna(today['Empty_Line']):
        alerts.append(f"🔴 {asset_name}: Trend turned BEARISH (Empty Line crossover).")

    # 2. Radar Momentum Change
    if pd.isna(yesterday['Radar_Up']) and not pd.isna(today['Radar_Up']):
        alerts.append(f"📈 {asset_name}: Short-term Radar turned UP.")
    elif pd.isna(yesterday['Radar_Down']) and not pd.isna(today['Radar_Down']):
        alerts.append(f"📉 {asset_name}: Short-term Radar turned DOWN.")

    # 3. Bollinger Band Breakouts
    if today['Close'] > today['BB_Upper']:
        alerts.append(f"⚠️ {asset_name}: Price broke ABOVE Upper Bollinger Band ({today['Close']:.2f}).")
    elif today['Close'] < today['BB_Lower']:
        alerts.append(f"⚠️ {asset_name}: Price broke BELOW Lower Bollinger Band ({today['Close']:.2f}).")

    # 4. RSI Extremes
    if today['RSI'] >= 70:
        alerts.append(f"🔥 {asset_name}: RSI is Overbought at {today['RSI']:.1f}.")
    elif today['RSI'] <= 30:
        alerts.append(f"🧊 {asset_name}: RSI is Oversold at {today['RSI']:.1f}.")

    # 5. TD Sequential Exhaustion
    if today['Countdown_Up'] == 13:
        alerts.append(f"🚨 {asset_name}: TD Sequential Exhaustion 13 (POTENTIAL TOP/SELL).")
    elif today['Countdown_Down'] == 13:
        alerts.append(f"🚨 {asset_name}: TD Sequential Exhaustion 13 (POTENTIAL BOTTOM/BUY).")

    return alerts

def send_email_alert(alert_messages):
    if not alert_messages:
        print("No new alerts triggered today.")
        return

    # Pull credentials from GitHub Secrets
    sender_email = os.environ.get("EMAIL_SENDER")
    sender_password = os.environ.get("EMAIL_PASSWORD")
    receiver_email = os.environ.get("EMAIL_RECEIVER")

    if not sender_email or not sender_password or not receiver_email:
        print("Email credentials not found in environment. Skipping email dispatch.")
        return

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f"📊 Trading Bot Alerts: {len(alert_messages)} Triggers Detected"

    body = "Here are your technical analysis triggers for today:\n\n"
    body += "\n".join(alert_messages)
    body += "\n\nView Dashboard: [Insert Your GitHub Pages URL Here]"

    msg.attach(MIMEText(body, 'plain'))

    try:
        # Assuming Gmail SMTP. Change to smtp.mail.yahoo.com or others if needed
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("✅ Alert email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# --- DASHBOARD BUILDER ---
def build_dashboard():
    start_date = datetime.today() - relativedelta(years=3)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    dropdown_buttons = []
    traces_per_asset = 13 
    
    all_triggered_alerts = [] # List to hold all alerts for the email

    print(f"\nStarting data fetch for {len(ASSETS)} assets...")
    
    for i, (name, (asset_type, symbol)) in enumerate(ASSETS.items()):
        print(f"Fetching {name}...")
        try:
            if asset_type == 'CN_FUTURE': df = ak.futures_zh_daily_sina(symbol=symbol)
            elif asset_type == 'INTL_FUTURE': df = ak.futures_foreign_hist(symbol=symbol)
            elif asset_type == 'US_STOCK': df = ak.stock_us_daily(symbol=symbol)
                
            column_mapping = {"日期": "Date", "date": "Date", "开盘价": "Open", "open": "Open", "收盘价": "Close", "close": "Close", "最高价": "High", "high": "High", "最低价": "Low", "low": "Low", "成交量": "Volume", "volume": "Volume"}
            df = df.rename(columns=column_mapping)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df = df[df.index >= pd.to_datetime(start_date)]
            
            df = apply_strict_tdx_strategy(df)
            df = apply_td_sequential(df)
            df = apply_bollinger_bands(df)
            df = apply_rsi(df)
            
            # Check for alerts and add to master list
            asset_alerts = evaluate_alerts(df, name)
            all_triggered_alerts.extend(asset_alerts)
            
            is_visible = (i == 0)
            
            # --- CHARTS ---
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=f'{name} Price', increasing_line_color='green', decreasing_line_color='red', visible=is_visible), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Golden_Line'], line=dict(color='red', width=2), name='📈 Trend (Bull/Bear)', legendgroup='trend', showlegend=True, visible=is_visible), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Empty_Line'], line=dict(color='cyan', width=2), name='Bearish', legendgroup='trend', showlegend=False, visible=is_visible), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Radar_Up'], line=dict(color='orange', width=1), name='🎯 Radar (Up/Down)', legendgroup='radar', showlegend=True, visible=is_visible), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Radar_Down'], line=dict(color='blue', width=1), name='Radar Down', legendgroup='radar', showlegend=False, visible=is_visible), row=1, col=1)

            sup9, sdown9 = df[df['Setup_Up'] == 9], df[df['Setup_Down'] == 9]
            cup13, cdown13 = df[df['Countdown_Up'] == 13], df[df['Countdown_Down'] == 13]
            fig.add_trace(go.Scatter(x=sup9.index, y=sup9['High'] * 1.015, mode='markers+text', text='9', textposition='middle center', textfont=dict(color='white', size=11, family='Arial Black'), marker=dict(color='blue', symbol='square', size=16), name='🔢 TD Sequential (9/13)', legendgroup='td', showlegend=True, visible=is_visible), row=1, col=1)
            fig.add_trace(go.Scatter(x=sdown9.index, y=sdown9['Low'] * 0.985, mode='markers+text', text='9', textposition='middle center', textfont=dict(color='white', size=11, family='Arial Black'), marker=dict(color='blue', symbol='square', size=16), name='Setup 9 Buy', legendgroup='td', showlegend=False, visible=is_visible), row=1, col=1)
            fig.add_trace(go.Scatter(x=cup13.index, y=cup13['High'] * 1.025, mode='markers+text', text='13', textposition='middle center', textfont=dict(color='white', size=11, family='Arial Black'), marker=dict(color='red', symbol='square', size=18), name='Exhaustion 13 Sell', legendgroup='td', showlegend=False, visible=is_visible), row=1, col=1)
            fig.add_trace(go.Scatter(x=cdown13.index, y=cdown13['Low'] * 0.975, mode='markers+text', text='13', textposition='middle center', textfont=dict(color='white', size=11, family='Arial Black'), marker=dict(color='red', symbol='square', size=18), name='Exhaustion 13 Buy', legendgroup='td', showlegend=False, visible=is_visible), row=1, col=1)

            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dash'), name='🌊 Bollinger Bands', legendgroup='bb', showlegend=True, visible=is_visible), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], line=dict(color='rgba(150, 150, 150, 0.5)', width=1), name='BB Middle', legendgroup='bb', showlegend=False, visible=is_visible), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dash'), name='BB Lower', legendgroup='bb', showlegend=False, fill='tonexty', fillcolor='rgba(150, 150, 150, 0.05)', visible=is_visible), row=1, col=1)

            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#B388FF', width=2), name='RSI (14)', showlegend=False, visible=is_visible), row=2, col=1)

            visibility = [False] * (len(ASSETS) * traces_per_asset)
            for j in range(traces_per_asset): visibility[(i * traces_per_asset) + j] = True
            dropdown_buttons.append(dict(label=name, method="update", args=[{"visible": visibility}, {"title": f"{name} - Strategy Overlay"}]))
            
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")

    # Process and send emails
    send_email_alert(all_triggered_alerts)

    fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)

    fig.update_layout(
        title=f'{list(ASSETS.keys())[0]} - Strategy Overlay',
        template='plotly_dark',
        height=850,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)"),
        updatemenus=[dict(active=0, buttons=dropdown_buttons, x=0.0, xanchor="left", y=1.10, yanchor="top")],
    )

    # --- PLOTLY AUTO-SCALE FIX ---
    # Setting autorange=True and fixedrange=False ensures the Y-axis adapts
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider=dict(visible=False))
    fig.update_yaxes(title_text="Price", autorange=True, fixedrange=False, row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], fixedrange=False, row=2, col=1)

    fig.write_html("index.html")
    print("\nDashboard built and saved as index.html!")

if __name__ == "__main__":
    build_dashboard()