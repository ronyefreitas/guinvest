# !pip install streamlit
# !pip install yfinance
# !pip install plotly
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import datetime as dt
import pandas as pd
import numpy as np
import pmdarima as pm
import requests
from bs4 import BeautifulSoup


today = dt.date.today()
tomorrow = today + dt.timedelta(days=1)
yesterday = today - dt.timedelta(days=1)

@st.cache(persist=True)  # avoid realoadig data constantly
def load_tickers_volume():
    """ This function returns a pandas DataFrame with the B3 most active stocks
    using as source the webpage
    https://www.ibovx.com.br/maiores-volume-bovespa.aspx"""

    agent = {"User-Agent": "Mozilla/5.0"}
    page = requests.get('https://www.ibovx.com.br/maiores-volume-bovespa.aspx',
                        headers=agent)
    # Getting the page content.
    soup = BeautifulSoup(page.content, "html.parser")
    # In order to find what you want from the webpage, you can open it in a
    # chrome web browser, open the inspect element tool and figure out the id
    # and/or tags of the desired data.
    table = soup.find_all("table")  # Getting the table
    table = table[0]
    # Looking into the table and retrieving only the tickers
    tickers_list = []
    volumes_list = []
    for row in table.find_all('tr')[1:]:
        try:
            tickers_list.append(row.find_all('a')[0].get_text())
            volumes_list.append(row.find_all('td')[4].get_text())
        except IndexError:
            pass

    df = pd.DataFrame({'Ticker': tickers_list, 'Volume': volumes_list})
    # in order to use the yfinance library to retrieve Stocks info directly
    # from Yahoo! Finance it is necessary to include '.SA' at the end of each
    # ticker, because that is the format required by yfinance.
    df['y_ticker'] = df['Ticker'].apply(lambda x: str(x) + '.SA')

    return df

@st.cache(persist=True)  # avoid realoadig data constantly
def load_tickers():
    """ This function returns a pandas DataFrame with all the tickers available
    for the Brazilian stocks market using as source the webpage
    https://www.fundamentus.com.br/resultado.php"""

    agent = {"User-Agent": "Mozilla/5.0"}
    page = requests.get('https://www.fundamentus.com.br/resultado.php',
                        headers=agent)
    # Getting the page content.
    soup = BeautifulSoup(page.content, "html.parser")
    # In order to find what you want from the webpage, you can open it in a
    # chrome web browser, open the inspect element tool and figure out the id
    # and/or tags of the desired data.
    soup.find_all(id="resultado")
    table_fundamentus = soup.find_all(id="resultado")[0]  # Getting the table
    tickers_list = []
    # Looking into the table and retrieving only the tickers
    for row in table_fundamentus.find_all('tr')[1:]:
        tickers_list.append(row.find(class_='tips').find('a').get_text())

    df = pd.DataFrame({'ticker': tickers_list})
    # in order to use the yfinance library to retrieve Stocks info directly
    # from Yahoo! Finance it is necessary to include '.SA' at the end of each
    # ticker, because that is the format required by yfinance.
    df['y_ticker'] = df['ticker'].apply(lambda x: str(x) + '.SA')

    return df

@st.cache(persist=True, suppress_st_warning=True)
def load_historical_data(tickers, start_date, end_date, df_type='prices'):
    """ This function returns a pandas DataFrame with the closing prices of the
    given tickers between a start date and and date (not included).
    When df_type='ret' option is selected, the function returns the percent
    change related to the previous closing price as well."""
    yf_tickers = []
    for ticker in tickers:
        # Creating a list of yfinance tikers objects
        yf_tickers.append([yf.Ticker(ticker), ticker])
    # Formarting date to match yfinance library requirements
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    df_all = pd.DataFrame()

    prices_ret_cols = []
    prices_cols = []
    for yf_ticker, str_ticker in yf_tickers:
        df_history = yf_ticker.history(start=start_date, end=end_date)
        # Sorting the data in order to show the latest valus first
        df_history.sort_index(ascending=False, inplace=True)
        df_all[str_ticker + ' Close'] = df_history['Close']

        # Calculating the return values based on previous closing price.
        df_all[str_ticker + ' Close D-1'] = (df_all[str_ticker
                                                    + ' Close'].shift(-1))

        df_all[str_ticker + ' Return (%)'] = ((df_all[str_ticker + ' Close']
                                                 / df_all[str_ticker +
                                                          ' Close D-1'] - 1)
                                              * 100)

        prices_cols.append(str_ticker + ' Close')

        prices_ret_cols.append(str_ticker + ' Close')
        prices_ret_cols.append(str_ticker + ' Return (%)')

    df_all.index = df_all.index.strftime('%Y-%m-%d')

    # Return a DataFrame with closing prices and percent return or with
    # closing prices only.
    if df_type == 'prices':
        return df_all[prices_cols]
    elif df_type == 'ret':
        return df_all[prices_ret_cols]
    else:
        return 0

st.cache(persist=True)
def make_chart(df_prices):

    traces = []
    for stock in df_prices.columns:
        traces.append(
            go.Scatter(
                x=df_prices.index,
                y=df_prices[stock],
                name=stock.split()[0],
                )
            )

    fig = dict(data=traces)

    # fig.update_layout(height=4000)
    st.plotly_chart(fig)

st.cache(persist=True)
def make_table(df_prices):

    table = ff.create_table(self.df_prices_ret_.round(decimals=2),
                            index=True)
    st.plotly_chart(table)

st.cache(persist=True)
def make_df_pred_ARIMA(df_prices, steps):
    """ This function creates the DataFrame which has the tickers, closing
    prices and predicted values based on ARIMA model for a selected number of
    steps ahead."""

    # Sort values from the oldest values to the newest ones in order to fit
    # the model.
    df_true_values = df_prices.sort_index(ascending=True)
    forecasts = pd.DataFrame(index=[f'(ARIMA) D + {i + 1}' for i in range(steps)])
    for col in df_true_values.columns:
        # Find ARIMA's p, d, q values automatically and fit the model.
        model = pm.auto_arima(df_true_values[col])
        forecasts[col] = model.predict(steps)
    # Dataframe with the last 5 real closing prices and the predicted values.
    df_results = pd.concat([df_true_values.iloc[-5:], forecasts])
    df_results.columns = [f'{col}'.split('.')[0] for col in df_results.columns]
    st.dataframe(df_results.round(decimals=4), height=2500)

st.cache(persist=True)
def make_df_compare(df_prices, periods=10):
    """ This function creates a DataFrame and a Plot that helps the user to
    evaluate the model performance, comparing the actual data with the value
    that the model would have predicted for the given date based only on the
    previous datapoints."""
    df_prices = df_prices.sort_index(ascending=True)
    preds = []
    i = periods + 1
    while (i >= 1):
        # The model only takes into acount the past values and then predicts
        # one step ahead.
        model = pm.auto_arima(df_prices.iloc[: - i])
        # st.dataframe(df_prices.iloc[: - i])
        preds.append(model.predict(1)[0])
        i = i - 1

    # The first prediction is omitted because it is related to the previous day
    df_preds = pd.DataFrame({'Predicted': preds[1:]},
                            index=df_prices.iloc[-periods:].index)

    # Putting the real and predicted values side to side.
    df_compare = pd.concat([df_prices.iloc[-periods:], df_preds],
                           axis=1, join='inner')
    # The real percent return.
    df_compare['Real Ret %'] = (((df_compare[df_compare.columns[0]]
                                 - df_compare[df_compare.columns[0]].shift(1))
                                / df_compare[df_compare.columns[0]].shift(1))
                                * 100)
    # The predicted percent return.
    df_compare['Pred Ret %'] = (((df_compare[df_compare.columns[1]]
                                 - df_compare[df_compare.columns[0]].shift(1))
                                / df_compare[df_compare.columns[1]].shift(1))
                                * 100)
    # Plotting the data.
    make_chart(df_compare[['Real Ret %', 'Pred Ret %']])
    st.dataframe(df_compare.round(decimals=4), height=2500)


# STREAMLIT DASHBOARD #
st.title("Guinvest")
st.sidebar.title("My Stocks")

df_tickers = load_tickers_volume()
# Filling in a multiselect box with all the tickers available.
tickers = st.sidebar.multiselect(label="Tickers",
                                 options=df_tickers['y_ticker'].values,
                                 key=0)

# A data imputer for the start and end date which will define the timespan
# which the model will consider in its fitting. The default start date is one
# year before today.
start_date = st.sidebar.date_input("From",
                                   value=today - dt.timedelta(days=365),
                                   max_value=today)
end_date = st.sidebar.date_input("Until",
                                 value=today,
                                 max_value=today) + dt.timedelta(days=1)

# A selectbox to choose the number of steps (days) ahead the model will predict
steps_options = np.arange(1, 11, 1)
steps_options = list(steps_options)
steps = st.sidebar.selectbox(label='Periods to forecast',
                             options=steps_options)

# st.dataframe(load_tickers_volume())
try:
    # load the data based on chosen tickers and show them.
    df_prices = load_historical_data(tickers, start_date, end_date,
                                     df_type='prices')
    make_chart(df_prices)
    make_df_pred_ARIMA(df_prices, int(steps))
except AttributeError:
    st.write('Please, select at least one valid ticker.')
except ValueError:
    st.write('There is a ticker which does not have values. Please, remove it\
             from selection.')
except TypeError:
    st.write('Select how many days ahead you want to predict.')

# The models are evaluated one ticker at a time. The user can choose among the
# previously selected ones.
st.sidebar.title('Evaluating models')
ticker_eval = st.sidebar.selectbox(label='Select ticker',
                                   options=tickers)

# A selectbox that enables the user to chdf_prices.iloc[: - i])oose the model evaluation timespan.
eval_options = np.arange(6, 31, 1)
eval_options = list(eval_options)
eval_periods = st.sidebar.selectbox(label='Evaluation period (days)',
                                    options=eval_options)

try:
    # Make the evaluation plots and table.
    make_df_compare(df_prices[ticker_eval + ' Close'], periods=eval_periods)
except NameError:
    pass
