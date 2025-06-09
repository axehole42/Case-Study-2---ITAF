# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 20:07:00 2024

@author: Viktor Tsonevski


"""

'This version of the code includes the Optimal Complete Portfolio, as well as the investors indifference curves for A =3, I ran this in Google Collab, might not work in Spyder'


####################### WITH 0.35% Interest Rate ################################

import numpy as np
import time
import yfinance as yf
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import os
import kaleido

os.chdir(r"C:\Users\Viktor\Desktop\UNI MUNSTER 2024-2026\Introduction to Advanced Finance\Case Study 1\")


'List of stock tickers'
stocklist = [
    'MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS',
    'DOW', 'XOM', 'GS', 'HD', 'INTC', 'IBM', 'JNJ', 'JPM', 'MCD',
    'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'VZ',
    'V', 'WBA', 'WMT'
]

'Define the start and end dates'
startdate = '2018-12-31'  # Start date
enddate = '2023-12-31'    # End date


'Removed weights variable here since weights are given by the optimization functions below, this was just for testing'
#weights = np.array([0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333,
# 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333,
# 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333,
# 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333,
# 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333,
# 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333])

stocks=stocklist
def getdata(stocks, start, end):
    # Download the data from Yahoo Finance
    stockdata = yf.download(stocks, start=start, end=end, interval='1d')['Adj Close']

    "Resample to get the last day of each month"
    monthly_closes = stockdata.resample('M').last()

    returns=monthly_closes.pct_change()
    meanreturns=returns.mean()
    meanreturns_special=returns.mean()*100
    meanreturns_percent=meanreturns.apply(lambda x: f"{x:0.2f}%")
    covmatrix=returns.cov()
    return meanreturns, covmatrix


def portfolio_performance(weights, meanreturns, covmatrix):
    returns = np.sum(meanreturns*weights)
    std = np.sqrt(np.dot(weights.T, np.dot(covmatrix, weights)))
    return returns, std

# Fetch and print the end-of-month closing prices


print(getdata(stocks, start=startdate, end=enddate))

meanreturns, covmatrix=getdata(stocks, start=startdate, end=enddate)
#returns, std = portfolio_performance(weights,meanreturns, covmatrix)

#print(f"Expected Portfolio-Equal Weight:{returns}")
#print(f"Std Deviation-Equal Weight:{std}")


def negativeSR(weights, meanreturns, covmatrix, riskfreerate=0.0035):
    'Define the negative Sharpe Ratio function'
    p_returns, p_std = portfolio_performance(weights, meanreturns, covmatrix)
    return -((p_returns-riskfreerate)/p_std)

def maxSR(meanreturns, covmatrix,riskfreerate=0.0035, constraintset=((0,1))):
    "Minimize the Negative SR, by alterring the weights of the portfolio"
    numassets = len(meanreturns)
    args = (meanreturns, covmatrix, riskfreerate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintset
    bounds=tuple(bound for asset in range(numassets))
    result = sc.optimize.minimize(negativeSR, numassets*[1./numassets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# result=maxSR(meanreturns, covmatrix)
# maxSR,maxweights=result['fun'],result['x']
# print(result)
# print(maxSR, maxweights)

def portfoliovariance(weights,meanreturns, covmatrix):
    return portfolio_performance(weights,meanreturns, covmatrix)[1]

def MVP(meanreturns, covmatrix, constraintset=((0,1))):
    "Minimize the portfolio variance"
    numassets=len(meanreturns)
    args = (meanreturns, covmatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintset
    bounds=tuple(bound for asset in range(numassets))
    result = sc.optimize.minimize(portfoliovariance, numassets*[1./numassets],
                                  args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def utility_function(weights, meanreturns, covmatrix, risk_aversion=3):
      """Utility function U = E(rp) - 0.5 * A * std^2"""
      portfolio_return, portfolio_std = portfolio_performance(weights, meanreturns, covmatrix)
      return portfolio_return - 0.5 * risk_aversion * portfolio_std**2

def negative_utility(weights, meanreturns, covmatrix, risk_aversion=3):
    "Negative of the utility function for minimization"
    return -utility_function(weights, meanreturns,covmatrix, risk_aversion)

def UMAX(meanreturns, covmatrix, risk_aversion=3, constraintset=((0,1))):
    "Maximize the utility function"
    numassets = len(meanreturns)
    args = (meanreturns, covmatrix, risk_aversion)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintset
    bounds = tuple(bound for asset in range(numassets))
    result = sc.optimize.minimize(negative_utility, numassets*[1./numassets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def utility_function_new(weights, meanreturns, covmatrix, risk_aversion_new=10):
      """Utility function U = E(rp) - 0.5 * A * std^2"""
      portfolio_return_new, portfolio_std_new = portfolio_performance(weights, meanreturns, covmatrix)
      return portfolio_return_new - 0.5 * risk_aversion_new * portfolio_std_new**2

def negative_utility_new(weights, meanreturns, covmatrix, risk_aversion_new=10):
    "Negative of the utility function for minimization"
    return -utility_function_new(weights, meanreturns,covmatrix, risk_aversion_new)

def UMAX_new(meanreturns, covmatrix, risk_aversion_new=10, constraintset=((0,1))):
    "Maximize the utility function"
    numassets = len(meanreturns)
    args = (meanreturns, covmatrix, risk_aversion_new)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintset
    bounds = tuple(bound for asset in range(numassets))
    result = sc.optimize.minimize(negative_utility_new, numassets*[1./numassets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# minvar_result=MVP(meanreturns, covmatrix)
# MVP,MVPweights=minvar_result['fun'],minvar_result['x']

# print(MVP, MVPweights)

def portfolioreturn(weights, meanreturns, covmatrix):
    return portfolio_performance(weights,meanreturns, covmatrix)[0]

def efficient_optimization(meanreturns, covmatrix, returntarget, constraintset=(0,1)):
    "For each return target, we want to optimize the portfolio for min variance"
    numassets = len(meanreturns)
    args=(meanreturns, covmatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: portfolioreturn(x,meanreturns, covmatrix)-returntarget}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound=constraintset
    bounds=tuple(bound for asset in range(numassets))
    effopt=sc.optimize.minimize(portfoliovariance, numassets*[1./numassets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return effopt


def calculated_results(meanreturns, covmatrix, riskfreerate=0.0035, constraintset=(0,1), risk_aversion=3, risk_aversion_new=10):
    "Read in all information"
    "we want the MAX sharpe Ratio, Min Vol, UMAX PF (A=3), the other UMAX with A=10 and the Efficient frontier"
    maxSR_Portfolio=maxSR(meanreturns, covmatrix)
    maxSR_returns,maxSR_std = portfolio_performance(maxSR_Portfolio["x"], meanreturns, covmatrix)
    maxSR_allocation=pd.DataFrame(maxSR_Portfolio["x"], index=meanreturns.index, columns=['allocation'])
    maxSR_allocation.allocation=[round(i*100,2) for i in maxSR_allocation.allocation]

    #return maxSR_returns, maxSR_std, maxSR_allocation

    MVP_Portfolio=MVP(meanreturns, covmatrix)
    MVP_returns,MVP_std = portfolio_performance(MVP_Portfolio["x"], meanreturns, covmatrix)
    MVP_allocation=pd.DataFrame(MVP_Portfolio["x"], index=meanreturns.index, columns=['allocation'])
    MVP_allocation.allocation=[round(i*100,23) for i in MVP_allocation.allocation]

    'Adding in the UMAX portfolio as well'

    UMAX_Portfolio = UMAX(meanreturns, covmatrix, risk_aversion)
    UMAX_returns, UMAX_std = portfolio_performance(UMAX_Portfolio["x"], meanreturns, covmatrix)
    UMAX_allocation = pd.DataFrame(UMAX_Portfolio["x"], index=meanreturns.index, columns=['allocation'])
    UMAX_allocation.allocation = [round(i*100, 2) for i in UMAX_allocation.allocation]

    'Adding in the second UMAX PF with A=10 just to have both on the EF Chart'
    UMAX_Portfolio_new = UMAX_new(meanreturns, covmatrix, risk_aversion_new)
    UMAX_returns_new, UMAX_std_new = portfolio_performance(UMAX_Portfolio_new["x"], meanreturns, covmatrix)
    UMAX_allocation_new = pd.DataFrame(UMAX_Portfolio_new["x"], index=meanreturns.index, columns=['allocation'])
    UMAX_allocation_new.allocation = [round(i*100, 2) for i in UMAX_allocation_new.allocation]

    'EFFICIENT FRONTIER'
    efficientlist=[]
    targetreturns=np.linspace(MVP_returns, maxSR_returns, 20)
    for target in targetreturns:
      efficientlist.append(efficient_optimization(meanreturns, covmatrix,target)['fun'])

    maxSR_returns,maxSR_std = round(maxSR_returns*100,2), round(maxSR_std*100,2)
    MVP_returns,MVP_std = round(MVP_returns*100,2), round(MVP_std*100,2)
    UMAX_returns,UMAX_std = round(UMAX_returns*100,2), round(UMAX_std*100,2)
    UMAX_returns_new,UMAX_std_new = round(UMAX_returns_new*100,2), round(UMAX_std_new*100,2)

    return maxSR_returns, maxSR_std, maxSR_allocation, MVP_returns,MVP_std, MVP_allocation, UMAX_returns, UMAX_std, UMAX_allocation, UMAX_returns_new, UMAX_std_new,UMAX_allocation_new, efficientlist, targetreturns

print(calculated_results(meanreturns, covmatrix))

def plot_indifference_curves(fig, max_std, risk_aversion=3, num_curves=3, utility_levels=None):
    if utility_levels is None:
        # Generate utility levels for the curves based on arbitrary increments for visualization
        utility_levels = [0.013015583214529* i for i in range(1, num_curves + 1)]

    for i, u in enumerate(utility_levels):
        # Skip the second indifference curve (i = 1 corresponds to the second curve)
        if i == 1:
            continue
        # Generate data points for indifference curves
        std_vals = np.linspace(0.01, max_std*4, 100)  # Standard deviation range
        returns_vals = [u + 0.5 * risk_aversion * (s ** 2) for s in std_vals]  # Calculate returns for each std

        # Plotting indifference curve
        indifference_curve = go.Scatter(
            name=f'Indifference Curve (U={u})',
            x=[s*100  for s in std_vals],  # Converting std to percentage for consistency
            y=[r*100  for r in returns_vals],  # Converting returns to percentage
            mode='lines',
            line=dict(width=1, dash='dot', color='orange'),
            hovertext=f'Indifference Curve (U={u})',
            hoverinfo='name',
        )

        fig.add_trace(indifference_curve)

def plot_specific_indifference_curve(fig, max_std, utility_level=0.0260312, risk_aversion=3):
    # Generate data points for the indifference curve
    std_vals = np.linspace(0.01, max_std * 4, 100)  # Standard deviation range
    returns_vals = [utility_level + 0.5 * risk_aversion * (s ** 2) for s in std_vals] 

    # Plotting the specific indifference curve
    indifference_curve = go.Scatter(
        name=f'Indifference Curve (U={utility_level})',
        x=[s * 100 for s in std_vals],
        y=[r * 100 for r in returns_vals],  
        mode='lines',
        line=dict(width=1, dash='dot', color='darkred'),
        hovertext=f'Indifference Curve (U={utility_level})', 
        hoverinfo='text', 
    )

    fig.add_trace(indifference_curve)


def find_tangency_point(meanreturns, covmatrix, riskfreerate, maxSR_returns, maxSR_std, utility_level, risk_aversion):
    'slope of the CML'
    cml_slope = (maxSR_returns / 100 - riskfreerate) / (maxSR_std / 100)

    'standard deviation (sigma_p) where the indifference curve has the same slope as the CML'
    sigma_p_tangent = cml_slope / risk_aversion

    'expected return at the tangency point'
    expected_return_tangent = utility_level + 0.5 * risk_aversion * sigma_p_tangent**2

    return expected_return_tangent, sigma_p_tangent


def EF_graph_with_CML(meanreturns, covmatrix, riskfreerate=0.0035, constraintset=(0,1), risk_aversion=3, risk_aversion_new=10, utility_level=0.0260312):
    ' Extract results from the calculated function'
    maxSR_returns, maxSR_std, maxSR_allocation, MVP_returns, MVP_std, MVP_allocation, UMAX_returns, UMAX_std,UMAX_allocation, UMAX_returns_new, UMAX_std_new,UMAX_allocation_new, efficientlist, targetreturns = calculated_results(meanreturns, covmatrix, riskfreerate, constraintset, risk_aversion)
    
    ' Calculate tangency point'
    expected_return_tangent, sigma_p_tangent = find_tangency_point(meanreturns, covmatrix, riskfreerate, maxSR_returns, maxSR_std, utility_level, risk_aversion)

    'maxSR (Maximum Sharpe Ratio portfolio)'
    MaxSharpeRatio = go.Scatter(
        name='Optimal Risky Portfolio',
        x=[maxSR_std],
        y=[maxSR_returns],
        mode='markers',
        marker=dict(color='red', size=14, line=dict(width=3, color='black')),
        text='Max Sharpe Ratio'
    )

    ' MVP (Minimum Variance Portfolio)'
    MinimumVariancePortfolio = go.Scatter(
        name='Minimum Variance Portfolio',
        x=[MVP_std],
        y=[MVP_returns],
        mode='markers',
        marker=dict(color='green', size=14, line=dict(width=3, color='black')),
        text='MVP'
    )
    ' Efficient Frontier curve'
    EF_Curve = go.Scatter(
        name='Efficient Frontier',
        x=[round(ef_std*100,4) for ef_std in efficientlist],
        y=[round(target*100,4) for target in targetreturns],
        mode='lines',
        line=dict(color='black', width=3, dash='dashdot'),
        text='Efficient Frontier',
    )

    ' Tangency Point '
    TangencyPoint = go.Scatter(
        name='Optimal Complete Portfolio for A = 3 (Tangency Point)',
        x=[round(sigma_p_tangent * 100,2)],
        y=[round(expected_return_tangent * 100,2)],
        mode='markers',
        marker=dict(color='orange', size=14, line=dict(width=3, color='black')),
        text=f'Tangency Point: σ={sigma_p_tangent*100:.2f}%, E(r)={expected_return_tangent*100:.2f}%',
        
    )

    'Capital Market Line (CML)'
    cml_x = np.linspace(0, max(efficientlist) *3* 100, 100)  # X-axis (std deviations)
    cml_y = riskfreerate * 100 + (maxSR_returns - riskfreerate * 100) / maxSR_std * cml_x  # CML Equation
    CML = go.Scatter(
        name='Capital Market Line (CML)',
        x=cml_x,
        y=cml_y,
        mode='lines',
        text="Capital Market Line",
        line=dict(color='blue', width=2, dash='solid'),
    )
    dummy_trace = go.Scatter(
    x=[15],  
    y=[0],  
    mode='markers',
    marker=dict(opacity=0),  
    showlegend=False
    )
    data = [MaxSharpeRatio, MinimumVariancePortfolio, EF_Curve, CML, TangencyPoint]

    layout = go.Layout(
        title='Portfolio Optimization - The Efficient Frontier, Capital Market Line, MVP, ORP and the Optimal Complete Portfolio for A=3',
        xaxis=dict(title='Standard Deviation of Portfolio, σ<sub>p</sub> (%)'),
        yaxis=dict(title='Expected Portfolio Return, μ<sub>p</sub> (%)'),
        showlegend=True,
        legend=dict(
            x=0.75, y=0, traceorder='normal',
            bgcolor='#e2e2e2',
            bordercolor='#FFFFFF',
            borderwidth=2
        ),
        width=1500,
        height=800,
    )




    fig = go.Figure(data=data, layout=layout)
    fig.add_trace(dummy_trace)
    fig.update_traces(hoverlabel=dict(font_size=16, namelength=-1))
    plot_specific_indifference_curve(fig, max(efficientlist))
    plot_indifference_curves(fig, max(efficientlist), risk_aversion=risk_aversion, num_curves=3) 
    fig.write_html("Efficient Frontier 2.html")
    fig.show()



' Call the modified graph function'
EF_graph_with_CML(meanreturns, covmatrix)
