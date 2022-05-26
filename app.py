# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:14:06 2022

@author: anare
"""

# IMPORTS
# -------------------------------------------------------------------------------------
from numpy import array
import dash
from dash import Dash, html, callback_context, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly
import plotly.express as px
#import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import yfinance as yf
import numpy as np
import pandas_ta as ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from itertools import cycle


# STYLING
# ---------------------------------------------------------------------------------------
# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#3f308a",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


# Options available in the dropdown to select the coins
coin_options = ['BITCOIN', 'CARDANO', 'COSMOS', 'AVALANCHE',
                'AXIE INFINITY', 'ETHEREUM', 'CHAINLINK', 'POLYGON', 'SOLANA', 'TERRA']


dropdown_coins = dcc.Dropdown(id='coins_selection',
                              options=[{'label': x, 'value': x}
                                       for x in coin_options],
                              searchable=False,
                              clearable=False,
                              value='BITCOIN'
                              )

sidebar = html.Div(
    [
        html.H2("CRYPTO \n DASH $" , className="box_title"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Overall", href="/", active="exact"),
                dbc.NavLink("Price Evolution", href="/page-1", active="exact"),
                dbc.NavLink("Momentum", href="/page-2", active="exact"),
                dbc.NavLink("Trend", href="/page-3", active="exact"),
                dbc.NavLink("Price Prediction",
                            href="/page-4", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)


# TEMPLATE
# ------------------------------------------------------------------------------------------


#create function to generate the datasets of each coin
def generate_dataset(coin):
    #get the data
#    data = yf.download(tickers=coin, period = '1y', interval = '1d')
    data = yf.Ticker(coin).history(period='1y')[['Close', 'Open', 'High', 'Volume', 'Low']]
    data['change_close'] = data['Close'].pct_change()
    data['change_close'] = data['change_close']*100
    return data

#Creating the datasets
BITCOIN = generate_dataset(coin = 'BTC-USD')
CARDANO = generate_dataset(coin = 'ADA-USD')
COSMOS = generate_dataset(coin = 'ATOM-USD')
AVALANCHE = generate_dataset(coin = 'AVAX-USD')
AXIE = generate_dataset(coin = 'AXS-USD')
ETHEREUM = generate_dataset(coin = 'ETH-USD')
CHAINLINK = generate_dataset(coin = 'LINK-USD')
POLYGON = generate_dataset(coin = 'MATIC-USD')
SOLANA = generate_dataset(coin = 'SOL-USD')
TERRA = generate_dataset(coin = 'LUNA1-USD')


#Creating the datasets
BITCOIN_mc = 562533040128
CARDANO_mc = 17193611264
COSMOS_mc = 3059195392
AVALANCHE_mc = 7515231744
AXIE_mc = 1248404864
ETHEREUM_mc = 235987320832
CHAINLINK_mc = 3281126400
POLYGON_mc = 5028843520
SOLANA_mc = 47190794240
TERRA_mc= 1213128064


# initialize data of lists.
ovr = {'Crypto':['Bitcoin', 'Cardano', 'Cosmos', 'Avalanche', 'Axie Infinity', 'Ethereum', 'Chainlink', 'Polygon', 'Solana', 'Terra'],
       'Type': ['Biggest Cryptocurrency', 'Others', 'Others', 'Others', 'Others', '2nd Biggest Cryptocurrency', 'Others', 'Others', 'Others', 'Others'],
        'Market_Cap':[BITCOIN_mc, CARDANO_mc, COSMOS_mc, AVALANCHE_mc, AXIE_mc, ETHEREUM_mc, CHAINLINK_mc, POLYGON_mc, SOLANA_mc, TERRA_mc],
          'Current_Price': [BITCOIN['Close'][-1], CARDANO['Close'][-1], COSMOS['Close'][-1], AVALANCHE['Close'][-1],
      AXIE['Close'][-1], ETHEREUM['Close'][-1], CHAINLINK['Close'][-1], POLYGON['Close'][-1], SOLANA['Close'][-1], TERRA['Close'][-1]],
      '% Change': [BITCOIN['change_close'][-1], CARDANO['change_close'][-1], COSMOS['change_close'][-1], AVALANCHE['change_close'][-1],
                     AXIE['change_close'][-1], ETHEREUM['change_close'][-1], CHAINLINK['change_close'][-1], 
                     POLYGON['change_close'][-1], SOLANA['change_close'][-1], TERRA['change_close'][-1]]}

  
# Create DataFrame
ovr = pd.DataFrame(ovr)


# Create Dic for coins
coin_options = {
    'BITCOIN': BITCOIN, 'CARDANO': CARDANO, 'COSMOS': COSMOS, 'AVALANCHE': AVALANCHE, 'AXIE INFINITY': AXIE, 'ETHEREUM': ETHEREUM,
    'CHAINLINK': CHAINLINK, 'POLYGON': POLYGON, 'SOLANA': SOLANA, 'TERRA': TERRA
}


# Dataset with only the close prices of the coins
progress = [BITCOIN['Close'], CARDANO['Close'], COSMOS['Close'], AVALANCHE['Close'],
            AXIE['Close'], ETHEREUM['Close'], CHAINLINK['Close'], POLYGON['Close'], SOLANA['Close'], TERRA['Close']]

progress = pd.concat(progress, axis=1)

progress.columns = ['BITCOIN', 'CARDANO', 'COSMOS',
                    'AVALANCHE', 'AXIE', 'ETHEREUM', 'CHAINLINK', 'POLYGON', 'SOLANA', 'TERRA']


# Dataset with all open prices
open_price = [BITCOIN['Open'], CARDANO['Open'], COSMOS['Open'], AVALANCHE['Open'],
              AXIE['Open'], ETHEREUM['Open'], CHAINLINK['Open'], POLYGON['Open'], SOLANA['Open'], TERRA['Open']]
open_price = pd.concat(open_price, axis=1)
open_price.columns = ['BITCOIN', 'CARDANO', 'COSMOS',
                      'AVALANCHE', 'AXIE INFINITY', 'ETHEREUM', 'CHAINLINK', 'POLYGON', 'SOLANA', 'TERRA']

# Dataset with all close prices
close_price = [BITCOIN['Close'], CARDANO['Close'], COSMOS['Close'], AVALANCHE['Close'],
               AXIE['Close'], ETHEREUM['Close'], CHAINLINK['Close'], POLYGON['Close'], SOLANA['Close'], TERRA['Close']]
close_price = pd.concat(close_price, axis=1)
close_price.columns = ['BITCOIN', 'CARDANO', 'COSMOS',
                       'AVALANCHE', 'AXIE INFINITY', 'ETHEREUM', 'CHAINLINK', 'POLYGON', 'SOLANA', 'TERRA']

# Dataset with all high prices
high_price = [BITCOIN['High'], CARDANO['High'], COSMOS['High'], AVALANCHE['High'],
              AXIE['High'], ETHEREUM['High'], CHAINLINK['High'], POLYGON['High'], SOLANA['High'], TERRA['High']]
high_price = pd.concat(high_price, axis=1)
high_price.columns = ['BITCOIN', 'CARDANO', 'COSMOS',
                      'AVALANCHE', 'AXIE INFINITY', 'ETHEREUM', 'CHAINLINK', 'POLYGON', 'SOLANA', 'TERRA']

# Dataset with all low prices
low_price = [BITCOIN['Low'], CARDANO['Low'], COSMOS['Low'], AVALANCHE['Low'],
             AXIE['Low'], ETHEREUM['Low'], CHAINLINK['Low'], POLYGON['Low'], SOLANA['Low'], TERRA['Low']]
low_price = pd.concat(low_price, axis=1)
low_price.columns = ['BITCOIN', 'CARDANO', 'COSMOS',
                     'AVALANCHE', 'AXIE INFINITY', 'ETHEREUM', 'CHAINLINK', 'POLYGON', 'SOLANA', 'TERRA']


#OVERALL
fig_overall = px.treemap(ovr, 
                 path=[px.Constant("Cryptocurrency Market"), 'Type', 'Crypto'], 
                 values='Market_Cap',
                 color_continuous_scale='dense_r',
                 color='% Change',
                 color_continuous_midpoint=0
                 )

fig_overall.update_layout(title_text='Market Cap', title_x=0.5)
fig_overall.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig_overall.data[0]['textfont']['size'] = 15




# 1st GRAPH
# declare figure
fig_progress = go.Figure()

# Candlestick
fig_progress.add_trace(go.Candlestick(x=BITCOIN.index,
                                      open=open_price['BITCOIN'],
                                      high=high_price['BITCOIN'],
                                      low=low_price['BITCOIN'],
                                      close=close_price['BITCOIN'], name='market data'))

# Add titles
fig_progress.update_layout(
    title='Bitcoin live share price evolution',
    yaxis_title='Bitcoin Price (kUS Dollars)',
    xaxis_title='Date',
    font_size=15, 
    font_color='black',
    plot_bgcolor='white')

# X-Axes
fig_progress.update_xaxes(
    rangeslider_visible=False,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15 Days", step="day", stepmode="backward"),
            dict(count=1, label="1 Month", step="month", stepmode="backward"),
            dict(count=3, label="3 Months", step="month", stepmode="backward"),
            dict(count=6, label="6 Months", step="month", stepmode="backward"),
            dict(step="all", label = '1 Year')
        ])
    )
)


# RSI GRAPH

df_rsi = pd.DataFrame()

df_rsi['close'] = close_price[["BITCOIN"]]  # creating a dataframe to plot rsi

# Calculate Price Differences
df_rsi['diff'] = df_rsi.diff(1)

# Calculate Avg. Gains/Losses
df_rsi['gain'] = df_rsi['diff'].clip(lower=0).round(2)
df_rsi['loss'] = df_rsi['diff'].clip(upper=0).abs().round(2)

# Definning the window length
window_length = 14
# Get initial Averages
df_rsi['avg_gain'] = df_rsi['gain'].rolling(
    window=window_length, min_periods=window_length).mean()[:window_length+1]
df_rsi['avg_loss'] = df_rsi['loss'].rolling(
    window=window_length, min_periods=window_length).mean()[:window_length+1]


# Get WMS averages
# Average Gains
for i, row in enumerate(df_rsi['avg_gain'].iloc[window_length+1:]):
    df_rsi['avg_gain'].iloc[i + window_length + 1] =\
        (df_rsi['avg_gain'].iloc[i + window_length] *
         (window_length - 1) +
         df_rsi['gain'].iloc[i + window_length + 1])\
        / window_length
# Average Losses
for i, row in enumerate(df_rsi['avg_loss'].iloc[window_length+1:]):
    df_rsi['avg_loss'].iloc[i + window_length + 1] =\
        (df_rsi['avg_loss'].iloc[i + window_length] *
         (window_length - 1) +
         df_rsi['loss'].iloc[i + window_length + 1])\
        / window_length

# Calculate RS Values
df_rsi['rs'] = df_rsi['avg_gain'] / df_rsi['avg_loss']

# Calculate RSI
df_rsi['rsi'] = 100 - (100 / (1.0 + df_rsi['rs']))

# ading open, high and low values to the dataset
df_rsi['open'] = open_price['BITCOIN'].values
df_rsi['high'] = high_price['BITCOIN'].values
df_rsi['low'] = low_price['BITCOIN'].values

# Create Figure
fig_rsi = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_width=[0.25, 0.75])

# Create Candlestick chart for price data
fig_rsi.add_trace(go.Candlestick(
    x=df_rsi.index,
    open=df_rsi['open'],
    high=df_rsi['high'],
    low=df_rsi['low'],
    close=df_rsi['close'],
    increasing_line_color='green',
    decreasing_line_color='red',
    showlegend=False), row=1, col=1)

# Make RSI Plot
fig_rsi.add_trace(go.Scatter(
    x=df_rsi.index,
    y=df_rsi['rsi'],
    line=dict(color='green', width=2),
    showlegend=False,
), row=2, col=1
)

# Add upper/lower bounds
fig_rsi.update_yaxes(range=[-10, 110], row=2, col=1)
fig_rsi.add_hline(y=0, col=1, row=2, line_color="#666", line_width=2)
fig_rsi.add_hline(y=100, col=1, row=2, line_color="#666", line_width=2)

# Add overbought/overSOLANAd
fig_rsi.add_hline(y=30, col=1, row=2, line_color='#336699',
                  line_width=2, line_dash='dash')
fig_rsi.add_hline(y=70, col=1, row=2, line_color='#336699',
                  line_width=2, line_dash='dash')

# Customize font, colors, hide range slider
layout = go.Layout(
    plot_bgcolor='white',
    title_text='Relative Strength Index - RSI',
    # Font Families
    font_size=15, 
    font_color='black',
    xaxis=dict(
        rangeslider=dict(
            visible=False
        )
    )
)
# update and display
fig_rsi.update_layout(layout)


#TREND - MACD

df_trend = BITCOIN.copy()

# Get the 26-day EMA of the closing price
k = df_trend['Close'].ewm(span=12, adjust=False, min_periods=12).mean()

# Get the 12-day EMA of the closing price
d = df_trend['Close'].ewm(span=26, adjust=False, min_periods=26).mean()

# Subtract the 26-day EMA from the 12-Day EMA to get the MACD
macd = k - d

# Get the 9-Day EMA of the MACD for the Trigger line
macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()

# Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
macd_h = macd - macd_s

# Add all of our new values for the MACD to the dataframe
df_trend['macd'] = df_trend.index.map(macd)
df_trend['macd_h'] = df_trend.index.map(macd_h)
df_trend['macd_s'] = df_trend.index.map(macd_s)

# calculate MACD values
df_trend.ta.macd(close='close', fast=12, slow=26, append=True)
# Force lowercase (optional)
df_trend.columns = [x.lower() for x in df_trend.columns]
# Construct a 2 x 1 Plotly figure
fig_trend = make_subplots(rows=2, cols=1)
# price Line
fig_trend.append_trace(
    go.Scatter(
        x=df_trend.index,
        y=df_trend['open'],
        line=dict(color='green', width=1),
        name='open',
        # showlegend=False,
        legendgroup='1',
    ), row=1, col=1
)
# Candlestick chart for pricing
fig_trend.append_trace(
    go.Candlestick(
        x=df_trend.index,
        open=df_trend['open'],
        high=df_trend['high'],
        low=df_trend['low'],
        close=df_trend['close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False
    ), row=1, col=1
)
# Fast Signal (%k)
fig_trend.append_trace(
    go.Scatter(
        x=df_trend.index,
        y=df_trend['macd_12_26_9'],
        line=dict(color='green', width=2),
        name='macd',
        # showlegend=False,
        legendgroup='2',
    ), row=2, col=1
)
# Slow signal (%d)
fig_trend.append_trace(
    go.Scatter(
        x=df_trend.index,
        y=df_trend['macds_12_26_9'],
        line=dict(color='red', width=2),
        # showlegend=False,
        legendgroup='2',
        name='signal'
    ), row=2, col=1
)
# Colorize the histogram values
colors = np.where(df_trend['macdh_12_26_9'] < 0, 'red', 'green')
# Plot the histogram
fig_trend.append_trace(
    go.Bar(
        x=df_trend.index,
        y=df_trend['macdh_12_26_9'],
        name='histogram',
        marker_color=colors,
    ), row=2, col=1
)
# Layout
layout = go.Layout(
    plot_bgcolor='white',
    title_text='Moving Average Convergence Divergence - MACD',
    
    # Font Families
    font_size=15, 
    font_color='black',
    xaxis=dict(
        rangeslider=dict(
            visible=False
        )
    )
)
# Update options
fig_trend.update_layout(layout)


# PRICE PREDICTION

closedf = pd.DataFrame()
closedf['Date'] = close_price.index
closedf.set_index('Date')

closedf['close'] = close_price['BITCOIN'].values
close_stock = closedf.copy()

# deleting date column and normalizing using MinMax Scaler

del closedf['Date']
scaler = MinMaxScaler(feature_range=(0, 1))
closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))


# we keep the training set as 70% and 30% testing set

training_size = int(len(closedf)*0.70)
test_size = len(closedf)-training_size
train_data, test_data = closedf[0:training_size,
                                :], closedf[training_size:len(closedf), :1]

# convert an array of values into a dataset matrix


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]  # i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

predictions = model_lr.predict(X_test)

train_predict = model_lr.predict(X_train)
test_predict = model_lr.predict(X_test)

train_predict = train_predict.reshape(-1, 1)
test_predict = test_predict.reshape(-1, 1)

# RMSE
rmse_lr = round(math.sqrt(mean_squared_error(y_test, test_predict)), 4)

# Transform back to original form

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

# shift train predictions for plotting

look_back = time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# shift test predictions for plotting
testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2) +
                1:len(closedf)-1, :] = test_predict

names = cycle(['Original close price',
              'Train predicted close price', 'Test predicted close price'])


plotdf = pd.DataFrame({'date': close_stock['Date'],
                       'original_close': close_stock['close'],
                       'train_predicted_close': trainPredictPlot.reshape(1, -1)[0].tolist(),
                       'test_predicted_close': testPredictPlot.reshape(1, -1)[0].tolist()})

fig_pred = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close'], plotdf['train_predicted_close'],
                                                plotdf['test_predicted_close']],
                   labels={'value': 'Close Price', 'date': 'Date'})
fig_pred.update_layout(title_text='Original Close Price VS Predicted Close Price',
                       plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig_pred.for_each_trace(lambda t:  t.update(name=next(names)))

fig_pred.update_xaxes(showgrid=False)
fig_pred.update_yaxes(showgrid=False)

x_input = test_data[len(test_data)-time_step:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()


lst_output = []
n_steps = time_step
i = 0
pred_days = 1
while(i < pred_days):

    if(len(temp_input) > time_step):

        x_input = np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1, -1)

        yhat = model_lr.predict(x_input)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat.tolist())
        temp_input = temp_input[1:]

        lst_output.extend(yhat.tolist())
        i = i+1

    else:
        yhat = model_lr.predict(x_input)

        temp_input.extend(yhat.tolist())
        lst_output.extend(yhat.tolist())

        i = i+1

last_days = np.arange(1, time_step+1)
day_pred = np.arange(time_step+1, time_step+pred_days+1)

model_lr = closedf.tolist()
model_lr.extend((np.array(lst_output).reshape(-1, 1)).tolist())
model_lr = scaler.inverse_transform(model_lr).reshape(1, -1).tolist()[0]


# card to display rmse
card_content_rmse = [
    dbc.CardHeader("ROOT-MEAN-SQUARED DEVIATION"),
    dbc.CardBody(
        [
            html.H5(rmse_lr, className="card-title"),
            #html.P(
            #    round((BITCOIN['change_close'][-1]), 3).astype(str) + '%',
            #    className="card-text",
            #),
        ], id='rmse_btc'
    ),
]

card_content_lr = [
    dbc.CardHeader("NEXT 24H PRICE PREDICTION ($)"),
    dbc.CardBody(
        [
            html.H5(round(model_lr[-1], 2), className="card-title"),
            #html.P(
            #    round((BITCOIN['change_close'][-1]), 3).astype(str) + '%',
            #    className="card-text",
            #),
        ], id='lr_btc'
    ),
]

row_1_pred = dbc.Row(
    [
        dbc.Col(dbc.Card(card_content_lr, color="primary", outline=True)),
        dbc.Col(dbc.Card(card_content_rmse, color="primary", outline=True)),
                
    ],
    className="mb-4", style={'width':'45%',"text-align": "center", 'font-size': '20px'},
)

cards_price_pred = html.Div([row_1_pred])


# card to use on Price Prediction - Price predicted
card_content1 = [
    dbc.CardHeader("BITCOIN ($)"),
    dbc.CardBody(
        [
            html.H5(round((BITCOIN['Close'][-1]), 2), className="card-title", style={"text-align": "center"}),
            html.P(
                round((BITCOIN['change_close'][-1]), 3).astype(str) + '%',
                className="card-text", style={"text-align": "center", 'color': 'purple'}
            ),
        ], id='bitcoin_price'
    ),
]

card_content2 = [
    dbc.CardHeader("ETHEREUM ($)"),
    dbc.CardBody(
        [
            html.H5(round((ETHEREUM['Close'][-1]), 2), className="card-title", style={"text-align": "center"}),
            html.P(
                round((ETHEREUM['change_close'][-1]), 3).astype(str) + '%',
                className="card-text", style={"text-align": "center", 'color': 'purple'}
            ),
        ], id='ethereum_price'
    ),
]
card_content3 = [
    dbc.CardHeader("CARDANO ($)"),
    dbc.CardBody(
        [
            html.H5(round((CARDANO['Close'][-1]), 2), className="card-title", style={"text-align": "center"}),
            html.P(
                round((CARDANO['change_close'][-1]), 3).astype(str) + '%',
                className="card-text", style={"text-align": "center", 'color': 'purple'}
            ),
        ], id='cardano_price'
    ),
]
card_content4 = [
    dbc.CardHeader("COSMOS ($)"),
    dbc.CardBody(
        [
            html.H5(round((COSMOS['Close'][-1]), 2), className="card-title", style={"text-align": "center"}),
            html.P(
                round((COSMOS['change_close'][-1]), 3).astype(str) + '%',
                className="card-text", style={"text-align": "center", 'color': 'purple'}
            ),
        ], id='cosmos_price'
    ),
]
card_content5 = [
    dbc.CardHeader("AVALANCHE ($)"),
    dbc.CardBody(
        [
            html.H5(round((AVALANCHE['Close'][-1]), 2), className="card-title", style={"text-align": "center"}),
            html.P(
                round((AVALANCHE['change_close'][-1]), 3).astype(str) + '%',
                className="card-text", style={"text-align": "center", 'color': 'purple'}
            ),
        ], id='avalanche_price'
    ),
]
card_content6 = [
    dbc.CardHeader("AXIE INFINITY ($)"),
    dbc.CardBody(
        [
            html.H5(round((AXIE['Close'][-1]), 2), className="card-title", style={"text-align": "center"}),
            html.P(
                round((AXIE['change_close'][-1]), 3).astype(str) + '%',
                className="card-text", style={"text-align": "center", 'color': 'purple'}
            ),
        ], id='axie_price'
    ),
]
card_content7 = [
    dbc.CardHeader("CHAINLINK ($)"),
    dbc.CardBody(
        [
            html.H5(round((CHAINLINK['Close'][-1]), 2), className="card-title", style={"text-align": "center"}),
            html.P(
                round((CHAINLINK['change_close'][-1]), 3).astype(str) + '%',
                className="card-text", style={"text-align": "center", 'color': 'purple'}
            ),
        ], id='CHAINLINK_price'
    ),
]

card_content8 = [
    dbc.CardHeader("POLYGON ($)"),
    dbc.CardBody(
        [
            html.H5(round((POLYGON['Close'][-1]), 2), className="card-title", style={"text-align": "center"}),
            html.P(
                round((POLYGON['change_close'][-1]), 3).astype(str) + '%',
                className="card-text", style={"text-align": "center", 'color': 'purple'}
            ),
        ], id='polygon_price'
    ),
]

card_content9 = [
    dbc.CardHeader("SOLANA ($)"),
    dbc.CardBody(
        [
            html.H5(round((SOLANA['Close'][-1]), 2), className="card-title", style={"text-align": "center"}),
            html.P(
                round((SOLANA['change_close'][-1]), 3).astype(str) + '%',
                className="card-text", style={"text-align": "center", 'color': 'purple'}
            ),
        ], id='solana_price'
    ),
]

card_content10 = [
    dbc.CardHeader("TERRA ($)"),
    dbc.CardBody(
        [
            html.H5(round((TERRA['Close'][-1]), 2), className="card-title", style={"text-align": "center"}),
            html.P(
                round((TERRA['change_close'][-1]), 3).astype(str) + '%',
                className="card-text", style={"text-align": "center", 'color': 'purple'}
            ),
        ], id='terra_price'
    ),
]


row_1 = dbc.Row(
    [
        dbc.Col(dbc.Card(card_content1, color="primary", outline=True)),
        dbc.Col(dbc.Card(card_content2, color="primary", outline=True)),
        dbc.Col(dbc.Card(card_content3, color="primary", outline=True)),
        dbc.Col(dbc.Card(card_content9, color="primary", outline=True)),
        dbc.Col(dbc.Card(card_content5, color="primary", outline=True)),
        
        
    ],
    className="mb-4",
)


row_2 = dbc.Row(
    [
     dbc.Col(dbc.Card(card_content4, color="primary", outline=True)),
     dbc.Col(dbc.Card(card_content6, color="primary", outline=True)),   
     dbc.Col(dbc.Card(card_content7, color="primary", outline=True)),
        dbc.Col(dbc.Card(card_content8, color="primary", outline=True)),
        dbc.Col(dbc.Card(card_content10, color="primary", outline=True)),
    ]
)

cards_overall = html.Div([row_1, row_2])


# THE APP ITSELF
# -------------------------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[
                dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

server = app.server

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    dbc.Row([
        dbc.Col(width=4),
        dbc.Col(width=4),
        dbc.Col(width=2),
        dbc.Col(dropdown_coins, width=2),
        dbc.Col(width=4),
        dbc.Col(width=8)
    ], style={'background-color': '#3f308a', 'padding': '15px'}),

    content,



])


# CALLBACKS
# ------------------------------------------------------------------------------------------
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return [
            # html.H1('Overall',
            #        style={'textAlign':'center'}),
            html.Br(),
            cards_overall,
            html.Br(),
            html.Br(),
            dcc.Graph(id='overall_treemap', figure = fig_overall, className='box')
        ]
    elif pathname == "/page-1":
        return [
            # html.H1('Price Evolution',
            #        style={'textAlign':'center'}),
            dcc.Graph(id='progress_graph', className='box')
             
        ]
    elif pathname == "/page-2":
        return [
            # html.H1('Momentum',
            #        style={'textAlign':'center'}),
            dcc.Graph(id='rsi_graph', className='box')
        ]
    elif pathname == "/page-3":
        return [
            # html.H1('Trend',
            #        style={'textAlign':'center'}),
            dcc.Graph(id='macd_graph', className='box')
        ]
    elif pathname == "/page-4":
        return [
            # html.H1('Price Prediction',
            #        style={'textAlign':'center'}),
            
            cards_price_pred,
            dcc.Graph(id='predicted_graph', className='box'),

        ]
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


@app.callback(
    Output('lr_btc', 'children'),
    Input('coins_selection', 'value')

)
# returns the price predicted
def price_prediction(coin):

    closedf = pd.DataFrame()
    closedf['Date'] = close_price.index
    closedf.set_index('Date')

    closedf['close'] = close_price[coin].values
    close_stock = closedf.copy()

    # deleting date column and normalizing using MinMax Scaler

    del closedf['Date']
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

    # we keep the training set as 70% and 30% testing set

    training_size = int(len(closedf)*0.70)
    test_size = len(closedf)-training_size
    train_data, test_data = closedf[0:training_size,
                                    :], closedf[training_size:len(closedf), :1]

    # convert an array of values into a dataset matrix

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]  # i=0, 0,1,2,3-----99   100
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    from sklearn.linear_model import LinearRegression
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    predictions = model_lr.predict(X_test)

    train_predict = model_lr.predict(X_train)
    test_predict = model_lr.predict(X_test)

    train_predict = train_predict.reshape(-1, 1)
    test_predict = test_predict.reshape(-1, 1)

    # RMSE
    rmse_lr = math.sqrt(mean_squared_error(y_test, test_predict))

    rmse_value = []

    rmse_value.append(rmse_lr)

    # Transform back to original form

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

    # shift train predictions for plotting

    look_back = time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2) +
                    1:len(closedf)-1, :] = test_predict

    names = cycle(['Original close price',
                  'Train predicted close price', 'Test predicted close price'])

    plotdf = pd.DataFrame({'date': close_stock['Date'],
                           'original_close': close_stock['close'],
                          'train_predicted_close': trainPredictPlot.reshape(1, -1)[0].tolist(),
                           'test_predicted_close': testPredictPlot.reshape(1, -1)[0].tolist()})

    fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close'], plotdf['train_predicted_close'],
                                               plotdf['test_predicted_close']],
                  labels={'value': 'Close Price', 'date': 'Date'})
    fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name=next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    # fig.show()

    x_input = test_data[len(test_data)-time_step:].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    from numpy import array

    lst_output = []
    n_steps = time_step
    i = 0
    pred_days = 1
    while(i < pred_days):

        if(len(temp_input) > time_step):

            x_input = np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1, -1)

            yhat = model_lr.predict(x_input)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat.tolist())
            temp_input = temp_input[1:]

            lst_output.extend(yhat.tolist())
            i = i+1

        else:
            yhat = model_lr.predict(x_input)

            temp_input.extend(yhat.tolist())
            lst_output.extend(yhat.tolist())

            i = i+1

    last_days = np.arange(1, time_step+1)
    day_pred = np.arange(time_step+1, time_step+pred_days+1)

    model_lr = closedf.tolist()
    model_lr.extend((np.array(lst_output).reshape(-1, 1)).tolist())
    model_lr = scaler.inverse_transform(model_lr).reshape(1, -1).tolist()[0]

    return round(model_lr[-1], 2)


@app.callback(
    Output('predicted_graph', 'figure'),
    Input('coins_selection', 'value')

)
# returns the price_prediction graph
def price_prediction_graph(coin):

    closedf = pd.DataFrame()
    closedf['Date'] = close_price.index
    closedf.set_index('Date')

    closedf['close'] = close_price[coin].values
    close_stock = closedf.copy()

    # deleting date column and normalizing using MinMax Scaler

    del closedf['Date']
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

    # we keep the training set as 70% and 30% testing set

    training_size = int(len(closedf)*0.70)
    test_size = len(closedf)-training_size
    train_data, test_data = closedf[0:training_size,
                                    :], closedf[training_size:len(closedf), :1]

    # convert an array of values into a dataset matrix

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]  # i=0, 0,1,2,3-----99   100
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    from sklearn.linear_model import LinearRegression
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    predictions = model_lr.predict(X_test)

    train_predict = model_lr.predict(X_train)
    test_predict = model_lr.predict(X_test)

    train_predict = train_predict.reshape(-1, 1)
    test_predict = test_predict.reshape(-1, 1)

    # Transform back to original form

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

    # shift train predictions for plotting

    look_back = time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2) +
                    1:len(closedf)-1, :] = test_predict

    names = cycle(['Original close price',
                  'Train predicted close price', 'Test predicted close price'])

    plotdf = pd.DataFrame({'date': close_stock['Date'],
                           'original_close': close_stock['close'],
                          'train_predicted_close': trainPredictPlot.reshape(1, -1)[0].tolist(),
                           'test_predicted_close': testPredictPlot.reshape(1, -1)[0].tolist()})

    fig_pred = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close'], plotdf['train_predicted_close'],
                                                    plotdf['test_predicted_close']],
                       labels={'value': 'Close Price', 'date': 'Date'})
    fig_pred.update_layout(title_text='Original Close Price VS Predicted Close Price',
                           plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig_pred.for_each_trace(lambda t:  t.update(name=next(names)))

    fig_pred.update_xaxes(showgrid=False)
    fig_pred.update_yaxes(showgrid=False)

    return fig_pred


@app.callback(
    Output('rmse_btc', 'children'),

    Input('coins_selection', 'value')

)
# returns the rmse
def price_prediction_rmse(coin):

    closedf = pd.DataFrame()
    closedf['Date'] = close_price.index
    closedf.set_index('Date')

    closedf['close'] = close_price[coin].values
    close_stock = closedf.copy()

    # deleting date column and normalizing using MinMax Scaler

    del closedf['Date']
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

    # we keep the training set as 70% and 30% testing set

    training_size = int(len(closedf)*0.70)
    test_size = len(closedf)-training_size
    train_data, test_data = closedf[0:training_size,
                                    :], closedf[training_size:len(closedf), :1]

    # convert an array of values into a dataset matrix

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]  # i=0, 0,1,2,3-----99   100
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    from sklearn.linear_model import LinearRegression
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    predictions = model_lr.predict(X_test)

    train_predict = model_lr.predict(X_train)
    test_predict = model_lr.predict(X_test)

    train_predict = train_predict.reshape(-1, 1)
    test_predict = test_predict.reshape(-1, 1)

    # RMSE
    rmse_lr = round(math.sqrt(mean_squared_error(y_test, test_predict)), 4)

    return rmse_lr


@app.callback(
    Output('progress_graph', 'figure'),
    Input('coins_selection', 'value')

)
def coin_price_progress(coin):
    # declare figure
    fig_progress = go.Figure()

    # Candlestick
    fig_progress.add_trace(go.Candlestick(x=progress.index,
                                          open=open_price[coin].values,
                                          high=high_price[coin].values,
                                          low=low_price[coin].values,
                                          close=close_price[coin].values, name='market data'))

    # Add titles
    fig_progress.update_layout(
        title_text='Live Share Price Evolution', 
        font_size=15, 
        font_color='black',
        yaxis_title='Price (kUS Dollars)',
        xaxis_title='Date',
        plot_bgcolor='white')

    # X-Axes
    fig_progress.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=15, label="15 Days", step="day", stepmode="backward"),
                dict(count=1, label="1 Month", step="month", stepmode="backward"),
                dict(count=3, label="3 Months", step="month", stepmode="backward"),
                dict(count=6, label="6 Months", step="month", stepmode="backward"),
                dict(step="all", label = '1 Year')
            ])
        )
    )

    # Show
    return fig_progress


@app.callback(
    Output('rsi_graph', 'figure'),
    Input('coins_selection', 'value')
)
def rsi_plot(coin):

    df_rsi = pd.DataFrame()
    df_rsi['close'] = close_price[[coin]]  # creating a dataframe to plot rsi

    # Calculate Price Differences
    df_rsi['diff'] = df_rsi.diff(1)

    # Calculate Avg. Gains/Losses
    df_rsi['gain'] = df_rsi['diff'].clip(lower=0).round(2)
    df_rsi['loss'] = df_rsi['diff'].clip(upper=0).abs().round(2)

    # Definning the window length
    window_length = 14
    # Get initial Averages
    df_rsi['avg_gain'] = df_rsi['gain'].rolling(
        window=window_length, min_periods=window_length).mean()[:window_length+1]
    df_rsi['avg_loss'] = df_rsi['loss'].rolling(
        window=window_length, min_periods=window_length).mean()[:window_length+1]

    # Get WMS averages
    # Average Gains
    for i, row in enumerate(df_rsi['avg_gain'].iloc[window_length+1:]):
        df_rsi['avg_gain'].iloc[i + window_length + 1] =\
            (df_rsi['avg_gain'].iloc[i + window_length] *
             (window_length - 1) +
             df_rsi['gain'].iloc[i + window_length + 1])\
            / window_length
    # Average Losses
    for i, row in enumerate(df_rsi['avg_loss'].iloc[window_length+1:]):
        df_rsi['avg_loss'].iloc[i + window_length + 1] =\
            (df_rsi['avg_loss'].iloc[i + window_length] *
             (window_length - 1) +
             df_rsi['loss'].iloc[i + window_length + 1])\
            / window_length

    # Calculate RS Values
    df_rsi['rs'] = df_rsi['avg_gain'] / df_rsi['avg_loss']

    # Calculate RSI
    df_rsi['rsi'] = 100 - (100 / (1.0 + df_rsi['rs']))

    # ading open, high and low values to the dataset
    df_rsi['open'] = open_price[coin].values
    df_rsi['high'] = high_price[coin].values
    df_rsi['low'] = low_price[coin].values

    # Create Figure
    fig_rsi = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_width=[0.25, 0.75])

    # Create Candlestick chart for price data
    fig_rsi.add_trace(go.Candlestick(
        x=df_rsi.index,
        open=df_rsi['open'],
        high=df_rsi['high'],
        low=df_rsi['low'],
        close=df_rsi['close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False), row=1, col=1)

    # Make RSI Plot
    fig_rsi.add_trace(go.Scatter(
        x=df_rsi.index,
        y=df_rsi['rsi'],
        line=dict(color='green', width=2),
        showlegend=False,
    ), row=2, col=1
    )

    # Add upper/lower bounds
    fig_rsi.update_yaxes(range=[-10, 110], row=2, col=1)
    fig_rsi.add_hline(y=0, col=1, row=2, line_color="#666", line_width=2)
    fig_rsi.add_hline(y=100, col=1, row=2, line_color="#666", line_width=2)

    # Add overbought/overSOLANAd
    fig_rsi.add_hline(y=30, col=1, row=2, line_color='#336699',
                      line_width=2, line_dash='dash')
    fig_rsi.add_hline(y=70, col=1, row=2, line_color='#336699',
                      line_width=2, line_dash='dash')

    # Customize font, colors, hide range slider
    layout = go.Layout(
        plot_bgcolor='white',
        title_text='Relative Strength Index - RSI',
        # Font Families
        font_size=15, 
        font_color='black',
        yaxis_title='Price (kUS Dollars)',
        xaxis_title='Date',
        xaxis=dict(
            rangeslider=dict(
                visible=False
            )
        )
    )
    # update and display
    fig_rsi.update_layout(layout)
    return fig_rsi


@app.callback(
    Output('macd_graph', 'figure'),
    Input('coins_selection', 'value')
)
def macd_trend(coin):

    df_trend = pd.DataFrame()
    df_trend['Date'] = close_price.index
    df_trend.set_index('Date')

    df_trend['close'] = close_price[coin].values
    df_trend['open'] = open_price[coin].values
    df_trend['high'] = high_price[coin].values
    df_trend['low'] = low_price[coin].values

    # Get the 26-day EMA of the closing price
    k = df_trend['close'].ewm(span=12, adjust=False, min_periods=12).mean()

    # Get the 12-day EMA of the closing price
    d = df_trend['close'].ewm(span=26, adjust=False, min_periods=26).mean()

    # Subtract the 26-day EMA from the 12-Day EMA to get the MACD
    macd = k - d

    # Get the 9-Day EMA of the MACD for the Trigger line
    macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()

    # Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
    macd_h = macd - macd_s

    # Add all of our new values for the MACD to the dataframe
    df_trend['macd'] = df_trend.index.map(macd)
    df_trend['macd_h'] = df_trend.index.map(macd_h)
    df_trend['macd_s'] = df_trend.index.map(macd_s)

    # calculate MACD values
    df_trend.ta.macd(close='close', fast=12, slow=26, append=True)
    # Force lowercase (optional)
    df_trend.columns = [x.lower() for x in df_trend.columns]
    # Construct a 2 x 1 Plotly figure
    fig_trend = make_subplots(rows=2, cols=1)
    # price Line
    fig_trend.append_trace(
        go.Scatter(
            x=df_trend.index,
            y=df_trend['open'],
            line=dict(color='green', width=1),
            name='open',
            # showlegend=False,
            legendgroup='1',
        ), row=1, col=1
    )
    # Candlestick chart for pricing
    fig_trend.append_trace(
        go.Candlestick(
            x=df_trend.index,
            open=df_trend['open'],
            high=df_trend['high'],
            low=df_trend['low'],
            close=df_trend['close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            showlegend=False
        ), row=1, col=1
    )
    # Fast Signal (%k)
    fig_trend.append_trace(
        go.Scatter(
            x=df_trend.index,
            y=df_trend['macd_12_26_9'],
            line=dict(color='green', width=2),
            name='macd',
            # showlegend=False,
            legendgroup='2',
        ), row=2, col=1
    )
    # Slow signal (%d)
    fig_trend.append_trace(
        go.Scatter(
            x=df_trend.index,
            y=df_trend['macds_12_26_9'],
            line=dict(color='red', width=2),
            # showlegend=False,
            legendgroup='2',
            name='signal'
        ), row=2, col=1
    )
    # Colorize the histogram values
    colors = np.where(df_trend['macdh_12_26_9'] < 0, 'red', 'green')
    # Plot the histogram
    fig_trend.append_trace(
        go.Bar(
            x=df_trend.index,
            y=df_trend['macdh_12_26_9'],
            name='histogram',
            marker_color=colors,
        ), row=2, col=1
    )
    # Layout
    layout = go.Layout(
        plot_bgcolor='white',
        title='Moving Average Convergence Divergence - MACD',
        
        # Font Families
        font_size=15, 
        font_color='black',
        xaxis=dict(
            rangeslider=dict(
                visible=False
            )
        )
    )
    # Update options
    fig_trend.update_layout(layout)

    # Show
    return fig_trend


# RUNNING THE APP
# -----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
