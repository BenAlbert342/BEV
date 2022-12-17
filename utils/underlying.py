"""Contains a set of methods for analyzing stock price data. Currently only works for closing prices."""

import pandas as pd
from utils.viz import plot_time_series
from math import sqrt

class ETF():
    """
    A class for analyzing the time series of index prices.
    """

    def __init__(self,prices,sym):
        """
        Initalizes the index price class.

        Args:
            prices (pd.Series): A time series of index prices. 
            sym (str): The index symbol.
        """
        self._p = prices
        self._sym = sym

    def get_returns(self,time_interval='daily',prices=None):
        """
        Calculates the precentage change in price.

        Args:
            time_interval (str): Time period to calcuate returns. 
                Should be 'daily', 'monthly', or 'annual'.
            prices (pd.Series): A time series of prices. If None
                will use the initalized series.

        Output: returns (pd.Series): Percentage change in price.
        """
        if time_interval=="daily":
            returns = self._calc_daily_retunrs(prices) # need to add monthly, and yearly returns

        return 100*returns

    def get_realized_volatility(self,n=30,returns=None,annualize=True):
        """
        Calculates the rolling standard deviation in price returns.

        Args:
            n (int): Number of days to include in calculation.
            returns (pd.Series): Time series of returns.
            annualize (bool): Annualize the realized volatility.

        Output:
            real_vol (pd.Series): Realized volatility.
        """
        if returns is None:
            returns = self.get_returns()
        if annualize:
            real_vol = returns.rolling(n).std()*sqrt(252)

        return real_vol

    def get_cumulative_returns(self,start_date=None,end_date=None,returns=None):
        """
        Calculates the cumulative returns from a index price time series.

        Args:
            start_date (DateTime): First date to use.
            end_date (DateTime): Last date to use.
            returns (pd.Series): Time series of returns.

        Output:
            cum_ret (pd.Series): Cumulative returns from index price time series.
        """
        if returns is None:
            returns = self.get_returns()
        if start_date is not None:
            returns = returns[pd.to_datetime(returns.index)>=start_date]
        if end_date is not None:
            returns = returns[pd.to_datetime(returns.index)<=end_date]
        returns = returns+1
        cum_ret = returns.cumprod()-1

        return cum_ret

    def analyze_index_prices(self,start_date=None,end_date=None,n=30):
        """
        Analyzes the time series of index prices. 

        Args:
            start_date (DateTime): First date to use.
            end_date (DateTime): : Last date to use.
            n (int): Number of days to include in calculation.

        Output:
            results (pd.DataFrame): A df with the index price, daily returns,
                realized volatility, and cumulative returns.
        """
        prices = self._p
        if start_date is not None:
            prices = prices[pd.to_datetime(prices.index)>=start_date]
        if end_date is not None:
            prices = prices[pd.to_datetime(prices.index)<=end_date]
        returns = self.get_returns(prices=prices)
        real_vol = self.get_realized_volatility(n=n,returns=returns)
        cum_ret = self.get_cumulative_returns(returns=returns)
        results = pd.concat([prices,returns,real_vol,cum_ret],axis=1)
        results.columns = ["Price","Return","RealizedVolatility","CumulativeReturn"]

        return results

    def _create_plot(self,ts,y_axes_name,start_date,end_date,title,trendline,window):
        """
        Creates a time series plot.

        Args:
            ts (pd.Series): The time series to plot.
            y_axes_name (str): Name of y-axis.
            start_date (datetime): First date to include in plot.
            end_date (datetime): Last date to include in plot.
            title (str): title of plot.
            trendline (bool): Include trend line.
            window (int): window to use for rolling average trend line.

        Output:
            fig (px.Scatter): The time series plot.
        """
        # Should have a filter ts method...
        if start_date is not None:
            ts = ts[pd.to_datetime(ts.index)>=start_date]
        if end_date is not None:
            ts = ts[pd.to_datetime(ts.index)<=end_date]

        fig = plot_time_series(
            ts = ts,
            y_axes_name=y_axes_name,
            series_name=self._sym,
            trendline=trendline,
            window=window,
            plot_title=title
        )

        return fig

    def plot_price(self,start_date=None,end_date=None,title=None,trendline=True,window=10):
        """
        Creates a time series plot of the closing stock price.

        Args:
            start_date (datetime): First date to include in plot.
            end_date (datetime): Last date to include in plot.
            title (str): title of plot.
            trendline (bool): Include trend line.
            window (int): window to use for rolling average trend line.

        Output:
            fig (px.Scatter): The time series plot.
        """
        price_ts = self._p
        fig = self._create_plot(
            ts=price_ts,
            y_axes_name="Price",
            start_date=start_date,
            end_date=end_date,
            title=title,
            trendline=trendline,
            window=window
        )
        return fig

    def plot_returns(self,start_date=None,end_date=None,title=None,trendline=True,window=10):
        """
        Creates a time series plot of the close to close stock returns.

        Args:
            n (int): Number of days to include in calculation.
            annualize (bool): Annualize the realized volatility.
            start_date (datetime): First date to include in plot.
            end_date (datetime): Last date to include in plot.
            title (str): title of plot.
            trendline (bool): Include trend line.
            window (int): window to use for rolling average trend line.

        Output:
            fig (px.Scatter): The time series plot.
        """
        returns = self.get_returns()
        fig = self._create_plot(
            ts=returns,
            y_axes_name="Returns",
            start_date=start_date,
            end_date=end_date,
            title=title,
            trendline=trendline,
            window=window
        )
        return fig

    def plot_realized_volatility(self,annualize=True,n=30,start_date=None,end_date=None,title=None,trendline=True,window=10):
        """
        Creates a time series plot of the realized volatility.

        Args:
            start_date (datetime): First date to include in plot.
            end_date (datetime): Last date to include in plot.
            title (str): title of plot.
            trendline (bool): Include trend line.
            window (int): window to use for rolling average trend line.

        Output:
            fig (px.Scatter): The time series plot.
        """
        vol = self.get_realized_volatility(
            n=n,
            annualize=annualize
        )
        fig = self._create_plot(
            ts=vol,
            y_axes_name="Volatility",
            start_date=start_date,
            end_date=end_date,
            title=title,
            trendline=trendline,
            window=window
        )
        return fig

    def _calc_daily_retunrs(self,prices=None):
        """
        Calculates the daily percentage change in prices.

        Args:
            prices (pd.Series): A time series of prices. If None
                will use the initalized series.
        """
        if prices is None:
            prices = self._p
        return prices.diff()/prices.shift(1)

    @property
    def closing_prices(self):
        """Returns the time series of index prices."""
        return self._p

    @property
    def symbol(self):
        """Returns the index ticker."""
        return self._sym