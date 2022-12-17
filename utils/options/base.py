"""This module contains the base option class that all other options should use."""

from abc import abstractmethod
from scipy.optimize import minimize
import numpy as np

class Option(): 
    """
    A base class for pricing options. 
    """

    def __init__(self,k,s,expiration,t,price=None,vol=None,div=0,r=0,iv0=0.1,detailed=False):
        """
        Initalizes the options class.

        Args:
            k (float): The strike price.
            s (float): Underly price.
            expiration (datetime): The expiration date.
            t (pd.DataFrame): Time till experation.
            price (float): Price of option. If None will calcuate 
                the option price.
            vol (float): volatility of underlying. If None will calcuate the 
                implied volatility.
            duv (float): The dividend.
            r (float): The interest rate
            iv0 (folat): The starting iv values
            detailed (bool): Prints summary of optimization results.
            right (str): 'call' or 'put'.
        """

        self._k = k
        self._expiration = expiration
        self._s = s
        self._price = price
        self._t = t
        self._tte = t/252
        self._vol = vol
        self._div = div
        self._r = r

        if self._price is None:
#            print("Calculating Option Price.")
            self._price = self._calc_price()
        elif self._vol is None:
#            print("Calculating Implied Volatility.")
            self._vol = self._calc_iv(
                iv0=iv0,
                detailed=detailed
            )
        
#        print("Calcuating Greeks")
        self._calc_greeks()

    @abstractmethod
    def _calc_price(self,iv):
        """Abstract method for calcuting the price of an option."""

    @abstractmethod
    def _calc_delta(self):
        """Calculates the delta of a option."""

    @abstractmethod
    def _calc_gamma(self):
        """Calculates the gamma of a option."""

    @abstractmethod
    def _calc_vega(self):
        """Calculates the Vega of a option."""

    @abstractmethod
    def _calc_theta(self):
        """Calculates the theta of a option."""
    
    @abstractmethod
    def _calc_rho(self):
        """Calculates the rho of a call option."""

    def _calc_greeks(self):
        """Calcuates the option's greeks."""
        self._delta = self._calc_delta()
        self._gamma = self._calc_gamma()
        self._vega = self._calc_vega()
        self._theta = self._calc_theta()
        self._rho = self._calc_rho()

    def _iv_objective(self,iv):
        """
        The objective function that is minimized to find the implied volatility for 
        a option.

        Args: 
            iv (float): Implied volatility.

        Output: 
            se (float): The squared error.
        """
        premium = self._calc_price(vol=iv)
        se = np.abs(self._price-premium)*1000
        return se

    def _calc_iv(self,iv0=1,detailed=False):
        """
        Calcuates the implied volatility (iv) for a option.

        Args:
            iv0 (folat): The starting iv values
            detailed (bool): Prints summary of optimization results.

        Output:
            iv (float): The implied volatility. 
        """
#        print(f"Implied Vol Starting Value: {iv0}")
        iv = minimize(
            fun=self._iv_objective,
            x0=iv0,
            bounds=((0,None),)
        )
        if detailed:
            print(iv)
        return iv.x[0]

    def get_option_data(self):
        """Returns all option data as a dictionary."""
        data = {
            "Strike": self._k,
            "Underlying": self._s,
            "DTE": self._t,
            "Experation": self._expiration,
            "Volatility": self._vol,
            "Price": self._price,
            "delta": self._delta,
            "gamma": self._gamma,
            "vega": self._vega,
            "theta": self._theta,
            "rho": self._rho
        }
        return data

    @property
    def strike(self):
        """Returns the strike price."""
        return self._k

    @property 
    def expiration_date(self):
        """Returns the option expiration date."""
        return self._expiration
    
    @property
    def underlying(self):
        """Returns the underly stock."""
        return self._s

    @property
    def price(self):
        """Returns the option premium."""
        return self._price

    @property
    def volatility(self):
        """Returns the volatility of the underlying asset."""
        return self._vol

    @property 
    def days_til_experation(self):
        """Returns the number of trading days until the option expires."""
        return self._t

    @property
    def interest_rate(self):
        """Returns the interest rate."""
        return self._r

    @property
    def dividends(self):
        """Returns the dividends."""
        return self._div

    @property
    def delta(self):
        """Returns the option delta."""
        return self._delta
    
    @property
    def gamma(self):
        """Returns the option gamma."""
        return self._gamma

    @property
    def vega(self):
        """Returns the option vega."""
        return self._vega
        
    @property
    def theta(self):
        """Returns the option theta."""
        return self._theta

    @property
    def rho(self):
        """Returns the option rho."""
        return self._rho
