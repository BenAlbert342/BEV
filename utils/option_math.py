"""
A collection of functions to preform basic options related calcuations.
"""
from math import (
    e,
    log,
    sqrt
)
import numpy as np
import pandas as pd

def european_calc_d1(s,k,tte,vol,r=0):
    """
    Calculates the d1 from the Black Scholes Mertan (BSM) formul for a European option
        
    Args: 
        s (float): underlying price.
        k (float): strike price.
        tte (float): time till experation measured as fraction of a trading year.
        vol (float): Implied volatility of option.
        r (float): Interest rate.

    Outputs:
        d1 (float): BSM d1 value. 
    """
    d1 = (log(s/k)+(r+vol**2/2.)*tte)/(vol*sqrt(tte))
    return d1

def european_calc_d2(s,k,tte,vol,r):
    """
    Calculates the d2 from the Black Scholes Mertan formul for a European option.
                
    Args: 
        s (float): underlying price.
        k (float): strike price.
        tte (float): time till experation measured as fraction of a trading year.
        vol (float): Implied volatility of option.
        r (float): Interest rate.

    Outputs:
        d2 (float): BSM d2 value.
    """
    d1 = european_calc_d1(s=s,k=k,tte=tte,vol=vol,r=r)
    d2 = d1-vol*sqrt(tte)
    return d2

def calc_forward_price(s,r,q,t):
    """
    Calculates the forwards price.

    Args:
        s (float): spot price.
        r (float): Risk free rate.
        q (float): dividend yeild.
        t (float): Time measured in fraction of a year.

    Output:
        f (float): Forward price. 
    """
    f = s * (e ** ((r-q)*t))
    return f

def calc_lks(f,k):
    """
    Calculates the log of the strike divided by the forward price.

    Args:
        f (float): Forward price.
        k (float): Strike price.
    Output: 
        lks (float): The natural log of the strike divided by the
            forward price.
    """
    lks = np.log(k/f)
    return(lks)

def calc_normK(f,k,sigma,t):
    """
    Calculates the normalized strike. 

    Args:
        f (float): Forward price.
        k (float): Strike price.
        sigma (float): Volatility.
        t (float): time till expiration measure in fration of a year.
    
    Output:
        normK (float): Normalized strike.
    """
    normK = (f-k)/(sigma*np.sqrt(t))
    return normK

def calc_atm_iv(k,f,iv,expiration):
    """
    Estimates the at the money (ATM) implied volatility (IV)
    by using piece-wise linear interpolation. Please note that 
    this function will NOT anualize the volatility. 

    Args:
        k (np.array): Vector of strike prices. 
        f (np.array): Vector of forward prices. 
        iv (Np.array): Vector of out of the money IV.
        expiration (np.array): Vector of expiration dates.

    Output:
        atm_iv (pd.Series): A vector of ATM IV, where the index 
            is the expiration date.
    """
    dist = k-f
    sign_dist = np.sign(dist)
    dist_abs = np.abs(dist)
    df_full = pd.DataFrame({
        "k": k,
        "Exp": expiration,
        "iv": iv,
        "dist_sign": sign_dist,
        "dist_abs": dist_abs,
        "dist": dist
    })
    df_atm = df_full[["Exp","dist_abs","dist_sign"]].groupby(["Exp","dist_sign"]).min().reset_index()
    df_atm = df_atm.merge(
        df_full[["Exp","dist_abs","iv","dist"]], how="left",left_on=["Exp","dist_abs"],right_on=["Exp","dist_abs"]
    )
    atm_iv = df_atm.groupby("Exp").apply(
        lambda x: np.interp(x=0,xp=x["dist"],fp=x["iv"])
    )
    
    return atm_iv

## Could Use to Price American Options in the future
def binomial_tree(type, s, K, r, sigma, t, N=2000,american="false"):
    #we improve the previous tree by checking for early exercise for american options
   
    #calculate delta T    
    deltaT = float(t) / N
 
    # up and down factor will be constant for the tree so we calculate outside the loop
    u = np.exp(sigma * np.sqrt(deltaT))
    d = 1.0 / u
 
    #to work with vector we need to init the arrays using numpy
    fs =  np.asarray([0.0 for i in range(N + 1)])
        
    #we need the stock tree for calculations of expiration values
    fs2 = np.asarray([(s * u**j * d**(N - j)) for j in range(N + 1)])
    
    #we vectorize the strikes as well so the expiration check will be faster
    fs3 =np.asarray( [float(K) for i in range(N + 1)])
    
 
    #rates are fixed so the probability of up and down are fixed.
    #this is used to make sure the drift is the risk free rate
    a = np.exp(r * deltaT)
    p = (a - d)/ (u - d)
    oneMinusP = 1.0 - p
 
   
    # Compute the leaves, f_{N, j}
    if type =="C":
        fs[:] = np.maximum(fs2-fs3, 0.0)
    else:
        fs[:] = np.maximum(-fs2+fs3, 0.0)
    
   
    #calculate backward the option prices
    for i in range(N-1, -1, -1):
       fs[:-1]=np.exp(-r * deltaT) * (p * fs[1:] + oneMinusP * fs[:-1])
       fs2[:]=fs2[:]*u
      
       if american=='true':
           #Simply check if the option is worth more alive or dead
           if type =="C":
                fs[:]=np.maximum(fs[:],fs2[:]-fs3[:])
           else:
                fs[:]=np.maximum(fs[:],-fs2[:]+fs3[:])
                
    # print fs
    return fs[0]