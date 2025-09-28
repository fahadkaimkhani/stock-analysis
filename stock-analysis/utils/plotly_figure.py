import plotly.graph_objects as go
import dateutil
import pandas_ta as pta
import datetime


def plotly_table(dataframe):
    headerColor = '#007BFF'  # Blue header background
    rowEvenColor = '#f8fafd'
    rowOddColor = '#e1efff'

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Metric</b>", "<b>Value</b>"],
            line_color='white',
            fill_color=headerColor,
            align='left',
            font=dict(color='white', size=14),
            height=35
        ),
        cells=dict(
            values=[
                ["<b>" + str(i) + "</b>" for i in dataframe.index],
                [str(i) for i in dataframe.iloc[:, 0]]
            ],
            fill_color=[[rowOddColor, rowEvenColor]*((len(dataframe)+1)//2)],
            align='left',
            line_color='white',
            font=dict(color='black', size=14),
            height=30
        )
    )])

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=250)
    return fig



def filter_data(dataframe, num_period):
    # Handle integer num_period as "days"
    if isinstance(num_period, int):
        date = dataframe.index[-1] - dateutil.relativedelta.relativedelta(days=num_period)

    elif num_period == '1mo':
        date = dataframe.index[-1] - dateutil.relativedelta.relativedelta(months=1)
    elif num_period == '5d':
        date = dataframe.index[-1] - dateutil.relativedelta.relativedelta(days=5)
    elif num_period == '6mo':
        date = dataframe.index[-1] - dateutil.relativedelta.relativedelta(months=6)
    elif num_period == '1y':
        date = dataframe.index[-1] - dateutil.relativedelta.relativedelta(years=1)
    elif num_period == '5y':
        date = dataframe.index[-1] - dateutil.relativedelta.relativedelta(years=5)
    elif num_period == 'ytd':
        date = datetime.datetime(dataframe.index[-1].year, 1, 1)
    else:
        date = dataframe.index[0]

    df = dataframe.reset_index()
    if 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)

    return df[df['Date'] > date]




def close_chart(dataframe, num_period =False):
    if num_period:
        dataframe = filter_data(dataframe,num_period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Open'], 
                            mode='lines', 
                            name='Open', line = dict ( width=2,color = '#5ab7ff')))
    fig.add_trace(go.Scatter(x=dataframe ['Date'], y=dataframe ['Close'],
                            mode='lines',
                            name='Close', line = dict(width=2,color = 'black')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['High'],
                            mode='lines', name='High', line = dict(width=2,color = '#0078ff')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Low'],
                            mode='lines', name='Low', line = dict(width=2,color = 'red')))
    fig.update_xaxes (rangeslider_visible=True)
    fig.update_layout (height = 500, margin=dict(l=0, r=20, t=20, b=0), plot_bgcolor = 'white', paper_bgcolor = '#e1efff', legend=dict(
    yanchor="top",
    xanchor="right"
    ))
    return fig


def candlestick(dataframe, num_period):
    dataframe = filter_data(dataframe,num_period)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=dataframe['Date'],
            open=dataframe['Open'], high=dataframe['High'], 
            low=dataframe['Low'], close=dataframe['Close']))

    fig.update_layout(showlegend = False,height = 500, margin=dict(l=0, r=20, t=20, b=0), plot_bgcolor = 'white', paper_bgcolor = '#e1efff')
    return fig


def RSI(dataframe, num_period):
    dataframe['RSI'] = pta.rsi(dataframe['Close'])
    dataframe = filter_data(dataframe,num_period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe.RSI, name = 'RSI', marker_color='orange', line = dict( width=2,color = 'orange'),
    ))
    fig.add_trace(go.Scatter(

    x=dataframe['Date'],
    y=[70]*len(dataframe), name = 'Overbought', marker_color='red', line = dict(width=2,color = 'red', dash='dash'),
    ))

    fig.add_trace(go.Scatter(
    x=dataframe['Date'],
    y=[30]*len(dataframe),fill='tonexty', name = 'Oversold', marker_color='#79da84', line = dict(width=2,color = '#79da84', dash='dash')
    ))

    fig.update_layout(yaxis_range=[0,100],
        height=200, plot_bgcolor = 'white', paper_bgcolor = '#e1efff',margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h",
    yanchor="top",
    y=1.02,
    xanchor="right",
    x=1
    )
    )
    return fig


def Moving_average(dataframe, num_period):
    dataframe['SMA_50'] = pta.sma(dataframe['Close'], 50)
    dataframe = filter_data(dataframe, num_period)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Close'],
                             mode='lines', name='Close', line=dict(width=2, color='black')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['SMA_50'],
                             mode='lines', name='SMA 50', line=dict(width=2, color='purple')))

    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(height=500, margin=dict(l=0, r=20, t=20, b=0),
                      plot_bgcolor='white', paper_bgcolor='#eeeeff',
                      legend=dict(yanchor="top", xanchor='center'))
    return fig


 
def MACD(dataframe, num_period):
    # Calculate MACD only once
    macd_data = pta.macd(dataframe['Close'])
    macd = macd_data.iloc[:, 0]
    macd_signal = macd_data.iloc[:, 1]
    macd_hist = macd_data.iloc[:, 2]

    # Add to dataframe
    dataframe['MACD'] = macd
    dataframe['MACD Signal'] = macd_signal
    dataframe['MACD Hist'] = macd_hist

    # Filter
    dataframe = filter_data(dataframe, num_period)

    # Color for histogram bars
    c = ['red' if cl < 0 else 'green' for cl in dataframe['MACD Hist']]

    # Create plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe['MACD'],
        name='MACD',
        line=dict(width=2, color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe['MACD Signal'],
        name='MACD Signal',
        line=dict(width=2, color='red', dash='dash')
    ))

    fig.add_trace(go.Bar(
        x=dataframe['Date'],
        y=dataframe['MACD Hist'],
        name='MACD Histogram',
        marker_color=c
    ))

    fig.update_layout(
        height=200,
        plot_bgcolor='white',
        paper_bgcolor='#e1efff',
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig
