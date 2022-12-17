"""A moduel for creating different figures"""

import plotly.express as px 
import pandas as pd
import numpy as np

def plot_time_series(
        ts,
        x_axes_name="date",
        y_axes_name=None,
        series_name=None,
        plot_title=None,
        template="simple_white",
        trendline=True,
        window=10
    ):
    """
    Creates a time series plot. 

    Args:
        ts (pd.Series): The time series to plot.
        x_axes_name (str): The name of the x-axis.
        y_axes_name (str): The name of the y-axis.
        series_name (str): Name of the time series.
        plot_title (str): Title of figure.
        template (str): Plotly template.
        trendline (bool): If true will overlay a roll average trend line.
        window (int): window to use for rolling average.

    Output:
        fig (px.Scatter): Time series plot.
    """
    y = ts.values
    x = ts.index

    if series_name is None:
        series_name = "Raw Time Series"

    if trendline:
        fig = px.scatter(
            x=pd.to_datetime(x),
            y=y,
            title=plot_title,
            template=template,
            color=[series_name]*len(x),
            trendline="rolling",
            trendline_options=dict(window=window),
            trendline_color_override="red"
        )
    else:
        fig = px.scatter(
            x=pd.to_datetime(x),
            y=y,
            title=plot_title,
            template=template
        )
    fig.update_traces(mode = 'lines')
    fig.update_xaxes(
        title=x_axes_name,
        rangeslider_visible=True
    )
    fig.update_yaxes(
        title=y_axes_name
    )
    
    return fig

def plot_iv_simle(
        opt_chain,
        expiration=None,
        dte=None,
        iv_col="IV",
        moneyness_col="Moneyness",
        x_axes_name="Moneyness",
        y_axes_name="Implied Volatility",
        trendline="lowess"
    ):
    """
    Plots the implied volatility (IV) against the moneyness for a given experation.
    Must pass in either an expiration date or a days till expiration value.

    Args:
        opt_chain (pd.DataFrame): A chain of option data.
        expiraton (str): The experation date of the simle 
            you would like to plot.
        dte (int): Days till experation of the simle 
            you would like to plot.
        iv_col (str): The name of the IV collumn in the 
            opt_chain df.
        moneyness_col (str): The name of the moneyness
            collumn in the opt_chain df.
        x_axes_name (str): Name of the x-axes.
        y_axes_name (str): Name of y-axes.
        trendline (str): Curve fitting method to include 
            trendline in plot. If None no trendline will be 
            added.

    Output: 
        fig (px.Figure): The IV smile plot.  
    """
    if not expiration is None:
        pdata = opt_chain[opt_chain.EXPIRE_DATE==expiration]
        ptitle = expiration
    elif not dte is None:
        pdata = opt_chain[opt_chain.DTE==dte]
        ptitle = dte
    else:
        print(f"ERROR: Must provide either a DTE value or Expiration date.")
    fig = px.scatter(
        pdata,
        y=iv_col,
        x=moneyness_col,
        template="simple_white",
        trendline=trendline
    )
    fig.update_xaxes(title=x_axes_name)
    fig.update_yaxes(title=y_axes_name)
    fig.update_layout(title=ptitle)
    return fig

def plot_iv_surface(
        opt_chain,
        iv_col="IV",
        moneyness_col="Moneyness",
        tte_col="stte",
        x_axes_name="Moneyness",
        y_axes_name="TTE",
        z_axes_name="Implied Volatility",
        title=None
    ):
    """
    Creates a 3d scatter plot of the implied volatility surface. 

    Args: 
        opt_chain (pd.DataFrame): A chain of option data.
        iv_col (str): The name of the IV collumn in the 
            opt_chain df.
        moneyness_col (str): The name of the moneyness
            collumn in the opt_chain df.
        tte_col (str): The name of the tte collumn in the 
            opt_chain df.
        x_axes_name (str): Name of the x-axes.
        y_axes_name (str): Name of y-axes.
        z_axes_name (str): Name of z-axes
        title (str): Title of the figure.
        
    Output:
        fig (px.Figure): The 3d scatter plot. 
    """
    fig = px.scatter_3d(
        opt_chain,
        x=moneyness_col,
        y=tte_col,
        z=iv_col,
        color=iv_col,
        template="simple_white",
        size_max=10,
        title=title
    )
    fig.update_layout(
        scene = dict(
            xaxis_title=x_axes_name,
            yaxis_title=y_axes_name,
            zaxis_title=z_axes_name
        )
    )
    fig.update_traces(marker_size = 2)
    return fig

def plot_iv_surface_tile(
        opt_chian,
        strike_col="STRIKE",
        dte_col="DTE",
        iv_col="IV",
        x_axes_title="Strike",
        y_axes_title="dte",
        title=None,
        dropna=False,
        normalize=True,
        underlying_col="UNDERLYING_LAST"
    ):
    """
    Creates a tile plot of the volatility surface. 

    Args: 
        opt_chain (pd.DataFrame): A chain of option data.
        strike_col (str): The name of the strike
            collumn in the opt_chain df.
        dte_col (str): The name of the date collumn in the 
            opt_chain df.
        iv_col (str): The name of the IV collumn in the 
            opt_chain df.
        title (str): Title of the figure.
        dropna (bool): Remvoe NAs from plot. 
        normalize (bool): If true will use the log of the
            strike divided by the underlying for the x-axis.
        underlying_col (str): Name of the underlying price
            collumn in the opt_chain df. Only applicable if 
            normalize is True.

    Output: 
        fig (px.Figure): The figure. 
    """
    if normalize:
        s = opt_chian[[underlying_col]].values[0]
    iv_grid = opt_chian[[strike_col,dte_col,iv_col]].\
        pivot(
            index=strike_col,
            columns=dte_col,
            values=iv_col
        )
    if normalize:
        iv_grid.index = np.log(iv_grid.index/s)
        iv_grid.columns = np.sqrt(iv_grid.columns/252)
    if dropna:
        iv_grid = iv_grid.dropna()
    fig = px.imshow(
        iv_grid.transpose(),
        labels=dict(x="Strike", y="Days Till Experation", color="IV"),
        template="simple_white",
        title=title,
        origin='lower',
        aspect='auto'
    )
    fig.update_xaxes(title=x_axes_title)
    fig.update_yaxes(title=y_axes_title)

    return fig