"""This module contains a set of methods for pricing European calls and puts."""

from math import log, sqrt, exp
from scipy.stats import norm
#from abc import abstractmethod
from utils.opions.base import Option
from utils.option_math import (
    european_calc_d1,
    european_calc_d2
)

class EuropeanCall(Option):
    """
    A class for calculating the price, implied volatility, and greeks for a
    European Call options. 
    """
    
    def _calc_price(self,vol=None):
        """
        Calculates the premium of a Eurpean Call from the Black Scholes Merton model.
        
        Args: 
            vol (float): Implied volatility of option.

        Outputs:
            premium (float): Price of European Call from BSM.
        """
        if vol is None:
            vol = self._vol
        d1 = european_calc_d1(s=self._s,k=self._k,tte=self._tte,vol=vol,r=self._r)
        d2 = european_calc_d2(s=self._s,k=self._k,tte=self._tte,vol=vol,r=self._r)
        premium = self._s*norm.cdf(d1)-self._k*exp(-self._r*self._tte)*norm.cdf(d2)
        return premium 

    def _calc_delta(self):
        """
        Calculates the delta of a call option.
        """
        d1 = european_calc_d1(s=self._s,k=self._k,tte=self._tte,vol=self._vol,r=self._r)
        delta = norm.cdf(d1)
        return delta

    def _calc_gamma(self):
        """Calculates the gamma of a call option."""
        d1 = european_calc_d1(s=self._s,k=self._k,tte=self._tte,vol=self._vol,r=self._r)
        gamma = norm.pdf(d1) / (self._s * self._vol * sqrt(self._tte))
        return gamma

    def _calc_vega(self):
        """Calculates the Vega of a call option."""
        d1 = european_calc_d1(s=self._s,k=self._k,tte=self._tte,vol=self._vol,r=self._r)
        vega = (self._s*norm.pdf(d1)*sqrt(self._tte))
        return vega

    def _calc_theta(self):
        """Calculates the theta of a call option."""
        d1 = european_calc_d1(s=self._s,k=self._k,tte=self._tte,vol=self._vol,r=self._r)
        d2 = european_calc_d2(s=self._s,k=self._k,tte=self._tte,vol=self._vol,r=self._r)
        first_term = -(self._s*norm.pdf(d1)*self._vol)/(2*sqrt(self._tte))
        second_term = self._r*self._k*exp(-self._r*self._tte)*norm.cdf(d2)
        theta = (first_term - second_term)
        return theta
    
    def _calc_rho(self):
        """Calculates the rho of a call option."""
        d2 = european_calc_d2(s=self._s,k=self._k,tte=self._tte,vol=self._vol,r=self._r)
        rho = (self._k*self._tte*exp(-self._r*self._tte)*norm.cdf(d2))
        return rho

class EuropeanPut(Option):
    """
    A class for calculating the price, implied volatility, and greeks for a
    European Put options. 
    """

    def _calc_price(self,vol=None):
        """
        Calculates the premium of a Eurpean Put from the Black Scholes Merton model.
        
        Args: 
            vol (float): Implied volatility of option.

        Outputs:
            premium (float): Price of European Put from BSM.
        """
        if vol is None:
            vol = self._vol
        d1 = european_calc_d1(s=self._s,k=self._k,tte=self._tte,vol=vol,r=self._r)
        d2 = european_calc_d2(s=self._s,k=self._k,tte=self._tte,vol=vol,r=self._r)
        premium = self._k * exp(-self._r*self._tte) * norm.cdf(-d2) - self._s * norm.cdf(-d1)
        return premium
        
    def _calc_delta(self):
        "Calculates the options delta."
        d1 = european_calc_d1(s=self._s,k=self._k,tte=self._tte,vol=self._vol,r=self._r)
        delta = -norm.cdf(-d1)
        return delta

    def _calc_gamma(self):
        "Calculates the option gamma"
        d1 = european_calc_d1(s=self._s,k=self._k,tte=self._tte,vol=self._vol,r=self._r)
        gamma = norm.pdf(d1)/(self._s*self._vol*sqrt(self._tte))
        return gamma

    def _calc_vega(self):
        "Calculates the option vega."
        d1 = european_calc_d1(s=self._s,k=self._k,tte=self._tte,vol=self._vol,r=self._r)
        vega = (self._s*norm.pdf(d1)*sqrt(self._tte))
        return vega

    def _calc_theta(self):
        """Calculates the option theta."""
        # 0.01*(-(S*norm.pdf(d1(S,K,T,r,sigma))*sigma)/(2*sqrt(T)) + r*K*exp(-r*T)*norm.cdf(-d2(S,K,T,r,sigma)))
        d1 = european_calc_d1(s=self._s,k=self._k,tte=self._tte,vol=self._vol,r=self._r)
        d2 = european_calc_d2(s=self._s,k=self._k,tte=self._tte,vol=self._vol,r=self._r)
        first_term = -(self._s*norm.pdf(d1)*self._vol)/(2*sqrt(self._tte))
        second_term = self._r*self._k*exp(-self._r*self._tte)*norm.cdf(-d2)
        theta = (first_term + second_term) 
        return theta

    def _calc_rho(self):
        """Calculates the option rho."""
        # 0.01*(-K*T*exp(-r*T)*norm.cdf(-d2(S,K,T,r,sigma)))
        d2 = european_calc_d2(s=self._s,k=self._k,tte=self._tte,vol=self._vol,r=self._r)
        rho = (-self._k*self._tte*exp(-self._r*self._tte)*norm.cdf(-d2))
        return rho

class EuropeanOption():
    """
    A class for calculating the price, implied volatility, and greeks for a
    European options. 
    """

    def __init__(self,k,s,expiration,t,right,price=None,vol=None,div=0,r=0,iv0=0.1,detailed=False):
        """
        Initailizes the option as a European 'call' or 'put'.
        
        Args:
        """
        if right=="call":
            self.instance = EuropeanCall(k=k,s=s,expiration=expiration,t=t,price=price,vol=vol,div=div,r=r,iv0=iv0,detailed=detailed) 
        else:
            self.instance = EuropeanPut(k=k,s=s,expiration=expiration,t=t,price=price,vol=vol,div=div,r=r,iv0=iv0,detailed=detailed)

    # called when an attribute is not found:
    def __getattr__(self, name):
        # assume it is implemented by self.instance
        return self.instance.__getattribute__(name)

    @property
    def right(self):
        """Returns the type of option (i.e. call or put)."""
        return self._right