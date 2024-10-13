import pandas as pd
import numpy as np
import os
import os
import json

#fixed variable
DATA_PATH = r".\Data"

def save_dict_json(filename, ticker_stock: dict, path=DATA_PATH):
    present_path = os.getcwd()
    os.chdir(path)
    
    with open(f"{filename}.json","w") as f:
        json.dump(ticker_stock,f)
        f.close()
    os.chdir(present_path)
    print("File saved")

def consolidated_df(stocks,tickers):
    final_df = pd.DataFrame()
    for stock in stocks:
        ticker = tickers.get(stock)
        df = get_stock_data(ticker)
        final_df[stock] = df['daily_return']
    return final_df

def get_stock_data(filename: str,path=DATA_PATH):
    file = os.path.join(path,filename+".csv")

    try:
        df = pd.read_csv(file, parse_dates=['Date'], index_col='Date')
        return df
        
    except FileNotFoundError:
        print(f"{filename} not found")


def get_ticker_name(filename='ticker_stockName.json',path=DATA_PATH):
    file = os.path.join(path,filename)

    try:
        with open(file, 'r') as tickers:
            ticker_dict = json.load(tickers)
            tickers.close()
    except Exception as e:
        print(e)

    return ticker_dict


def save_df_to_csv(df,ticker,path=DATA_PATH):
    full_file_path = os.path.join(path, f"{ticker}.csv")
    
    try:
        # Check if the file already exists to modify permissions.
        if os.path.exists(full_file_path):
            os.chmod(full_file_path, 0o666)  # Set file permissions to be readable and writable.
        else:
            print(f"File not found, creating new file: {ticker}")
        
        # Save the DataFrame to CSV.
        df.to_csv(full_file_path, index=True)
        print(f"{ticker} Data saved successfully")
    
    except PermissionError:
        print("Permission denied: You don't have the necessary permissions to modify this file.")
    
    except Exception as e:
        print(f"An error occurred: {e}")


def calculate_daily_return(df):
    df['daily_return'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
    df.dropna(inplace=True)
    return df

def calculate_roi(df,invested_amount = None, syear=None, eyear=None):
    if ((syear != None) and (eyear !=None)):
        new_df = df[(df.index.year >= syear) & (df.index.year <=eyear)]
    else:
        new_df = df.copy()

    cum_return = df['daily_return'].add(1).cumprod().iloc[-1]

    if invested_amount != None:
        present_value = invested_amount * cum_return
        return present_value
    else:
        return cum_return - 1
    
def calculate_mean(df, syear=None, eyear=None):
    if ((syear != None) and (eyear !=None)):
        new_df = df[(df.index.year >= syear) & (df.index.year <=eyear)]
    else:
        new_df = df.copy()

    mean_return = df['daily_return'].mean()
    return mean_return

def calculate_stdv(df, syear=None, eyear=None):
    if ((syear != None) and (eyear !=None)):
        new_df = df[(df.index.year >= syear) & (df.index.year <=eyear)]
    else:
        new_df = df.copy()

    std_retun = df['daily_return'].std()
    return std_retun


def calculate_cv(df, syear=None, eyear=None):
    if ((syear != None) and (eyear !=None)):
        new_df = df[(df.index.year >= syear) & (df.index.year <=eyear)]
    else:
        new_df = df.copy()

    mean_return = calculate_mean(df, syear=None, eyear=None)
    std_return = calculate_stdv(df, syear=None, eyear=None)

    return std_return/mean_return

def portfolio_std(weights, cov_matrix):
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(variance)

# Function to calculate expected annualized return
def expected_return(weights, consolidated_df):
    return np.sum(consolidated_df.mean() * weights) * 252  # assuming daily returns, 252 trading days in a year

# Function to calculate Sharpe ratio
def sharp_ratio(weights, consolidated_df, cov_matrix, risk_free_rate):
    portfolio_ret = expected_return(weights, consolidated_df)
    portfolio_risk = portfolio_std(weights, cov_matrix)
    return (portfolio_ret - risk_free_rate) / portfolio_risk

# Function to minimize negative Sharpe ratio
def negative_sharp_ratio(weights, consolidated_df, cov_matrix, risk_free_rate):
    return -sharp_ratio(weights, consolidated_df, cov_matrix, risk_free_rate)


