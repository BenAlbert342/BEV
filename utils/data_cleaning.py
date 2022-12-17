"""This module contains a set of method for cleaning option data."""
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from scipy.interpolate import SmoothBivariateSpline
from utils.option_math import (
    calc_forward_price,
    calc_lks
)
from utils.opions.european import (
    EuropeanCall,
    EuropeanPut
)

class DataCleaner():
    """A class for cleaning option data."""

    def __init__(
            self,
            underlying,
            minDTE=0,
            maxDTE=35,
            minMoneyness=-0.7,
            maxMoneyness=0.7,
            cols=["QUOTE_DATE","UNDERLYING_LAST","FORWARD","EXPIRE_DATE","DTE","STRIKE","C_IV","C_DELTA","C_BID","C_ASK","C_PRICE","P_IV","P_DELTA","P_BID","P_ASK","P_PRICE","INTERPOLATED"],
            calandar="NYSE",
            r=0
        ):
        """
        Initalizes the data cleaning class.

        Args:
            underlying (pd.DataFrame): A time series of underlying prices.
                Date column should be QUOTE_DATE, and price collumn should be
                UNDERLYING_LAST.
            minDTE (int): The minimum number of days till experation
                to include in the data.
            maxDTE (int): The maximum number of days till experation
                to include in the data.
            minMoneyness (float): The minimum value of the moneyness that 
                options will be included in the data. Filtering ocurs on
                the first day of data.
            maxMoneyness (float): The maximum value of the moneyness that 
                options will be included in the data.
            cols (list): A list of columns to keep in the data.
            calandar (str): The calandar you would like to use to calcuate 
                dte.
            r (float): The risk free rate.
        """
        self._s = underlying
        self._minDTE = minDTE
        self._maxDTE = maxDTE
        self._minMoneyness = minMoneyness
        self._maxMoneyness = maxMoneyness
        self._cols = cols
        self._calandar = mcal.get_calendar(calandar)
        self._r = r

    def clean_data(self,x,expiration):
        """
        Cleans up messy option data. 

        Args:
            x (pd.DataFrame): Option data.
            expiration (str): The date that the option expires. 
        
        output: 
            x_clean (pd.DataFrame): Clean data.
        """
        self._x = x.copy()
        self._x.drop(["DTE","UNDERLYING_LAST","EXPIRE_DATE"],axis=1,inplace=True)
        self._expiry = expiration
        self._calc_dte()
        self._filter_by_dte()
        self._dates = self._x.QUOTE_DATE.unique()
        self._calc_forward()
        self._filter_by_moneyness()
        self._strikes = self._x.STRIKE.unique()
        self._calc_mark_price()
        # If there are holes in the option time series 
        if len(self._strikes)*len(self._dates)!=self._x.shape[0]:
            # Add the missing dates
            self._fill_in_missing_dates()
        self._interpolate()
        self._calc_iv()


        self._x["EXPIRE_DATE"] = expiration
        self._x = self._x[self._cols]

        return self._x

    def _calc_dte(self):
        """
        Calculates the number of trading days until the option expires. 
        """
        self._x.QUOTE_DATE = pd.to_datetime(self._x.QUOTE_DATE) # Convert date to datetime
        start_date = self._x.QUOTE_DATE.min() # Get first day of data
        # Look up business days on calandar
        biz_days = self._calandar.schedule(start_date=start_date, end_date=self._expiry).index
        num_trading_days = len(biz_days) # Get the number of trading days 
        calandar = pd.DataFrame({
            "QUOTE_DATE": biz_days,
            "DTE": [num_trading_days-(i+1) for i in range(num_trading_days)]
        })
        self._x = calandar.merge( # merege with data and add missing days 
            self._x, how="left", left_on="QUOTE_DATE", right_on="QUOTE_DATE"
        )
        self._x["tte"] = self._x.DTE/252

    def _calc_forward(self):
        """Calcuates the forward price and the option's moneyness."""
        underlying_ts = pd.DataFrame(self._dates,columns=["QUOTE_DATE"]).\
            merge(self._s,how="left",left_on=["QUOTE_DATE"],right_on=["QUOTE_DATE"])
        if underlying_ts.UNDERLYING_LAST.isna().any():
            underlying_ts.UNDERLYING_LAST = underlying_ts.UNDERLYING_LAST.interpolate()
        self._x = self._x.merge(
            underlying_ts, how="left", left_on=["QUOTE_DATE"], right_on=["QUOTE_DATE"]
        )
        self._x["FORWARD"] = calc_forward_price(
            s=self._x.UNDERLYING_LAST,
            r=self._r,
            q=0,
            t=self._x.tte
        )
        self._x["lks"] = calc_lks(f=self._x.FORWARD,k=self._x.STRIKE)

    def _calc_mark_price(self):
        """
        Calculates the mid point between the bid and the ask for calls
        and puts.
        """
        self._x["C_PRICE"] = (self._x.C_BID + self._x.C_ASK)/2
        self._x["P_PRICE"] = (self._x.P_BID + self._x.P_ASK)/2

    def _calc_iv(self):
        """
        Calculates the iv and delta. 
        """
        self._x.C_IV.fillna(0.5,inplace=True)
        self._x.P_IV.fillna(0.5,inplace=True)

        c_iv = []
        c_delta = []
        p_iv = []
        p_delta = []

        for indx, row in self._x.iterrows():
            call = EuropeanCall(
                k=row.STRIKE,s=row.UNDERLYING_LAST,expiration="",t=(row.DTE+1e-4),price=row.C_PRICE,iv0=row.C_IV
            )
            c_iv.append(call.volatility)
            c_delta.append(call.delta)

            put = EuropeanPut(
                k=row.STRIKE,s=row.UNDERLYING_LAST,expiration="",t=(row.DTE+1e-4),price=row.P_PRICE,iv0=row.P_IV
            )
            p_iv.append(put.volatility)
            p_delta.append(put.delta)

        self._x.C_IV = c_iv
        self._x.C_DELTA = c_delta
        self._x.P_IV = p_iv
        self._x.P_DELTA = p_delta

    def _filter_by_dte(self):
        """
        Removes observations that are greater than or equal to the maxDTE
        of less than or equal to the minDTE.
        """
        self._x = self._x[
            (
                self._x.DTE>=self._minDTE
            )&(
                self._x.DTE<=self._maxDTE
            )
        ].reset_index(drop=True)

    def _filter_by_moneyness(self):
        """Filteres the data by option moneyness on the first day of data."""
        start_date = self._x.QUOTE_DATE.min()
        strikes = self._x[self._x.QUOTE_DATE==start_date]
        strikes = self._x.STRIKE[
            (
                self._x.lks>=self.minMoneyness
            )&(
                self._x.lks<=self.maxMoneyness
            )
        ].to_list()
        self._x = self._x[self._x.STRIKE.isin(strikes)]

    def _interpolate(self):
        """Fills in the implicitly missing dates and interpolates the missing data."""
        train = self._x.dropna()
        spline = SmoothBivariateSpline(
            x=train.lks.to_numpy(),
            y=train.tte.to_numpy(),
            z=np.log(train.C_PRICE.to_numpy())
        )
        self._x["C_PRICE_interp"] = np.exp(spline.ev(xi=self._x.lks,yi=self._x.tte))

        spline = SmoothBivariateSpline(
            x=train.lks.to_numpy(),
            y=train.tte.to_numpy(),
            z=np.log(train.P_PRICE.to_numpy())
        )
        self._x["P_PRICE_interp"] = np.exp(spline.ev(xi=self._x.lks,yi=self._x.tte))

        self._x["INTERPOLATED"] = (self._x.C_PRICE.isna()) | (self._x.P_PRICE.isna())
        self._x.C_PRICE = self._x.apply(
            lambda x: x.C_PRICE_interp if np.isnan(x.C_PRICE) else x.C_PRICE,
            axis=1
        )
        self._x.P_PRICE = self._x.apply(
            lambda x: x.P_PRICE_interp if np.isnan(x.P_PRICE) else x.P_PRICE,
            axis=1
        )

    def _fill_in_missing_dates(self):
        """
        Fills in the implicitly missing dates in option time series.
        """
        complete_df = pd.DataFrame(
            [(k,date) for k in self._strikes for date in self._dates],
            columns=["STRIKE","QUOTE_DATE"]
        )
        underlying_table = self._x[["QUOTE_DATE","UNDERLYING_LAST","FORWARD","DTE"]].drop_duplicates()
        self._x.drop(["UNDERLYING_LAST","FORWARD","DTE"],axis=1,inplace=True)
        complete_df = complete_df.merge(
            underlying_table,how="left",left_on="QUOTE_DATE",right_on="QUOTE_DATE"
        )
        self._x = complete_df.merge(
            self._x, how="left", left_on=["STRIKE","QUOTE_DATE"], right_on=["STRIKE","QUOTE_DATE"]
        )
        self._x["lks"] = calc_lks(f=self._x.FORWARD,k=self._x.STRIKE)
        self._x["tte"] = self._x.DTE/252

    @property
    def minDTE(self):
        """Returns the minium days till experiation."""
        return self._minDTE

    @property
    def maxDTE(self):
        """Returns the maximum days till experiation."""
        return self._maxDTE

    @property
    def cols(self):
        """Returns the selected columns."""
        return self._cols
    
    @property
    def calandar(self):
        """Returns the trading calandar."""
        return self._calandar

    @property
    def minMoneyness(self):
        """Returns the minimum moneyness value."""
        return self._minMoneyness

    @property
    def maxMoneyness(self):
        """Returns the maximum moneyness value."""
        return self._maxMoneyness
    
    @property
    def riskFreeRate(self):
        """Returns the risk free rate."""
        return self._r