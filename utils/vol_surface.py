"""
This moduel contains a set of method for fitting and visualizing a implied volatility surface. 
"""

from scipy.interpolate import SmoothBivariateSpline
import pandas as pd
import numpy as np
from utils.viz import (
    plot_iv_simle,
    plot_iv_surface,
    plot_iv_surface_tile
)
from utils.option_math import (
    calc_atm_iv,
    calc_forward_price,
    calc_lks,
    calc_normK
)

pd.options.mode.chained_assignment = None

class VolatilitySurface():
    """
    A class for analyzing a chain of options. Converts strikes to moneyness measurement,
    interpolates missing data, and plots the volatility surface. 
    """

    def __init__(self,moneyness,interpolate=True,calc_iv=False,option=None):
        """
        Initalizes the volatility surface class. 

        Args:
            moneyness (str): The measure of moneyness to be used. The class will 
                accept 'strike', 'lks', or 'normalized_strike'.
            interpolate (bool): If True will interpolate the missing data using 
                a bivariate cubic smoothing spline. The independent varibles are
                moneyness and the square root of the time till experation.
            calc_iv (bool): If true will calcuate the implied volatility 
                from option quotes. 
            option (Option): The type of option. This class will be used to 
                calcuate the implied volatility if calc_iv is True.
        """
        self._moneyness = moneyness
        self._interp = interpolate
        self._calc_iv = calc_iv
        self._option = option
        self._moneyness = moneyness
        self._data = None

    def fit_surface(self,opt_chain,r=0):
        """
        Fits a volatility surface to the option data.
        
        Args:
            opt_chain (pd.DataFrame): A days worth of option data.
            r (float): Risk free rate.
        """
        self._data = opt_chain.copy()
        # Calcuate Time Till Experation 
        self._data["tte"] = self._data['DTE'] / 252
        self._data["stte"] = np.sqrt(self._data.tte)
        # Calculate the Forward price
        self._data["forward_price"] = calc_forward_price(
            s=self._data.UNDERLYING_LAST,
            r=r,
            q=0,
            t=self._data.tte
        )
        if self._calc_iv: # If call and put IV is not already calculated 
            # Calculate the IV
            self._calculate_implied_vol(r=r)
        else: # Otherwise
             # Get teh otm IV   
            self._get_otm_iv() # May move out of this class

        self._calc_moneyness(x=opt_chain)
        if self._interp:
            self._interpolate_missing_iv()

    def plot_smile(self,expiration,trendline=None):
        """
        Plots the implied volatility smile. 

        Args:
            expiration (str): The expiration date of the smile 
                you would like to plot. 
            trendline  (str): The trendline you would like to 
                include in the plot.

        Output: 
            fig (px.Figure): The volatility smile figure. 
        """
        fig = plot_iv_simle(
            opt_chain=self._data,
            expiration=expiration,
            x_axes_name=self._moneyness,
            trendline=trendline
        )
        return fig

    def plot_surface(self,title=None):
        """
        Plots the implied volatility surface as a 3d scatter plot.

        Args: 
            title (str): The title of the figure.

        Output: 
            fig (px.Figure): The 3d scatter plot.
        """
        fig = plot_iv_surface(
            opt_chain=self._data,
            iv_col="IV",
            moneyness_col="Moneyness",
            tte_col="stte",
            x_axes_name=self._moneyness,
            y_axes_name="Sqrt of tte",
            title=title
        )
        return fig

    def plot_surface_tile(self,dropna=True,title=None):
        """
        Creates a tile plot of the IV surface. 

        Args: 
            dropna (bool): Remove NA values from plot.
            title (str): The title of the figure. 

        Output:
            fig (px.Figure): The tile plot. 
        """
        fig = plot_iv_surface_tile(
            opt_chian=self._data,
            strike_col="STRIKE",
            dte_col="DTE",
            iv_col="IV",
            normalize=True,
            x_axes_title="ln(Strike/Underlying)",
            y_axes_title="sqrt(time)",
            title=title,
            dropna=dropna
        )
        return fig

    ### Methods Used to Calculate the Implied Vol ###
    def _calculate_implied_vol(self,r):
        """
        Calcuates the implied volatility for a chain of out of the money options.
        The results are saved as an IV column in self._data.

        Args:
            r (float): Risk free rate.
        """
        iv = []
        for indx, row in self._data.iterrows():
            if row.STRIKE >= row.forward_price:
                right="call"
            else:
                right="put"
            try:
                _ = self._get_iv(
                    k=row.STRIKE,
                    s=row.UNDERLYING_LAST,
                    expiration=row.EXPIRE_DATE,
                    t=row.DTE,
                    right=right,
                    price=row.PRICE,
                    r=r 
                )
            except:
                _ = np.NaN
            iv.append(_)
        self._data["IV"] = iv

    def _get_iv(self,k,s,expiration,t,price,r,right):
        """
        Fits an option and returns the implied volatility. 

        Args:
            k (float): Strike price.
            s (float): underlying price.
            expiration (str): expiration date.
            t (float): Time till expiration.
            price (float): Option price. 
            r (float): Risk free rate.
            right (str): 'Call' or 'Put'.
        
        Output: 
            iv (float): Implied volatility. 
        """
        opt = self._option(
            k=k,
            s=s,
            expiration=expiration,
            t=t,
            right=right,
            price=price,
            r=r,
            iv0=0.1
        )
        iv = opt.volatility
        
        return iv

    def _get_otm_iv(self):
        """
        Calcautes the out of the money (OTM) IV. Results are saved 
        as an IV column on self._data.
        """
        self._data["IV"] = self._data.apply(lambda x: x['C_IV'] if x['STRIKE']>=x['forward_price'] else x['P_IV'],axis=1)
    
    ### Methods Used to Calculate the Moneyness of an Option. ###
    def _calc_moneyness(self, x):
        """
        Calcuates the moneyness of an option and adds the results as a
        'Moneyness' column on self._data.

        Args:
            x (pd.DataFrame): A chain of option data.
        
        Output:
            x (pd.DataFrame): A chain of option data with moneyness measurment.
        """
        if self._moneyness=="strike":
            self._calc_strike_moneyness()
        elif self._moneyness=="lks":
            self._calc_lks()
        elif self._moneyness=="normalized_strike":
            self._calc_norm_strike()
        else:
            print(f"Error: {self._moneyness} not supported moneyness must be 'strike', 'lks', or 'normalized_strike'")

        return x

    def _calc_strike_moneyness(self):
        """Sets the moneyness column equal to the strike."""
        self._data["Moneyness"] = self._data["STRIKE"]

    def _calc_lks(self):
        """Calculates the log of the strike divided by the forward."""
        self._data["Moneyness"] = calc_lks(
            f=self._data.forward_price,
            k=self._data.STRIKE
        )

    def _calc_norm_strike(self):
        """Calculates the normalized strike and adds it to dataframe."""
        atm_iv = calc_atm_iv(
            k=self._data.STRIKE,
            f=self._data.forward_price,
            iv=self._data.IV,
            expiration=self._data.EXPIRE_DATE
        )
        atm_iv = atm_iv.to_frame().reset_index().rename(columns={"Exp": "EXPIRE_DATE", 0: "ATM_IV"})
        self._data = self._data.merge(
            atm_iv, how="left", left_on=["EXPIRE_DATE"], right_on=["EXPIRE_DATE"]
        )
        self._data["Moneyness"] = calc_normK(
            f=self._data.forward_price,
            k=self._data.STRIKE,
            sigma=self._data.ATM_IV,
            t=self._data.tte
        )
    ### Methods Used to Interpolate Missing Data ###
    def _interpolate_missing_iv(self):
        """
        Interpolates the missing data using a bivariate cubic smoothing spline. 
        """
        train = self._data[["stte","IV","Moneyness"]].dropna()
        spline = SmoothBivariateSpline(
            x=train.Moneyness.to_numpy(),
            y=train.stte.to_numpy(),
            z=train.IV.to_numpy()
        )
        self._data.IV = self._data.apply(
            lambda x: spline.ev(xi=x.Moneyness,yi=x.stte) if np.isnan(x.IV) else x.IV,
            axis=1
        ).astype(float)
    ### Properties ###
    @property
    def calc_iv(self):
        """Returns if the implied volatility was calcuated."""
        return self._calc_iv

    @property
    def moneyness(self):
        """Returns the moneyness measure for options chain."""
        return self._moneyness

    @property
    def options_data(self):
        """Returns the options data."""
        return self._data

    @property 
    def interpolate(self):
        """Returns if the missing data was interpolated."""
        return self._interp