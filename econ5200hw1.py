# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:40:59 2024

@author: aadit
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import seaborn as sns
import warnings

def file_setup(filename, sel_vars, binary_vars, forced_binary_vars):
    chunks = []
    for chunk in pd.read_csv(filename, chunksize=1000):
        # Processing each chunk
        df = chunk.copy()  # Make a copy to avoid modifying the original chunk

        # Perform necessary modifications
        # Filtering binary values
        for var in binary_vars:
            df.loc[:, var] = df[var].apply(lambda x: clean_and_convert_to_int(x))
            df.loc[df[var] == 'N', var] = 1
            df.loc[df[var] != 1, var] = 0
        categorical_cols = ['BLD','HHRACE','HHSPAN','HHGRAD']
        for var in categorical_cols:
            df.loc[:, var] = df[var].apply(lambda x: clean_and_convert_to_int(x))
        
        df['HINCP'] = np.where(df['HINCP'] < 1, 1, df['HINCP'])
        df['HINCP_log'] = np.log(df['HINCP'])

        # Create additional binary variables
        df['detached'] = np.where(df['BLD'] == 2, 1, 0)
        df['trailer'] = np.where(df['BLD'] == 1, 1, 0)
        df['white'] = np.where(df['HHRACE'] == 1, 1, 0)
        df['black'] = np.where(df['HHRACE'] == 2, 1, 0)
        df['asian'] = np.where(df['HHRACE'] == 1, 1, 0)  # This line seems incorrect. It assigns 1 for 'asian' if 'HHRACE' is 1.
        df['hispanic'] = np.where(df['HHSPAN'] == 1, 1, 0)
        df['hs_grad'] = np.where((df['HHGRAD'] == 39) | (df['HHGRAD'] == 41), 1, 0)
        df['bachelors'] = np.where(df['HHGRAD'] == 41, 1, 0)
        df['grad_school'] = np.where(df['HHGRAD'].isin([45, 46, 47]), 1, 0)
        
        # Filter the DataFrame based on sel_vars
        df = df[sel_vars]
        
        # Append the modified chunk to the list of chunks
        chunks.append(df)
    result_df = pd.concat(chunks, ignore_index=True)
    
    return result_df



def clean_and_convert_to_int(s):
    cleaned_string = s.strip("'")
    return (int(cleaned_string.lstrip('0')) if cleaned_string != '0' else 0)


def summary_statistics(df):
    print('DESCRIPTIVE STATISTICS')
    print(df.describe())
    x_vars = [i for i in sel_vars if i != 'HINCP_log']
# Assign x variables
    x = df[x_vars]        
    correlation_matrix = x.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()
    
def ols_summary(df, sel_vars):
    
    model_name = 'OLS'
    x_vars = [i for i in sel_vars if i != 'HINCP_log']
    # Assign x variables
    X = df[x_vars]    
    Y = df['HINCP_log']
    df = df.rename(columns={'HINCP_log': 'y'})
    # Adding constant to the independent variables
    X = sm.add_constant(X)
    
    # Fit the OLS model
    model = sm.OLS(Y, X).fit()
    residual = model.resid
    fitted_values = model.fittedvalues
    # Print model summary
    print(model.summary())
    residuals_plot(fitted_values,residual,model_name)

    
    
def ridge(df, sel_vars):
    model_name = 'Ridge'
    df = df.rename(columns={'HINCP_log': 'y'})
    y = df['y']
    x_vars = [i for i in sel_vars if i != 'HINCP_log']
    # Assign x variables
    x = df[x_vars]
    print(x_vars)

    # Standardizing features made my model perform much worse
    #scaler.fit_transform(x)
    
    # Define a range of lambda values to search
    lambda_list = [10 ** exp for exp in range(-7, 4)]
    param_grid = {'alpha': [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    
    # Create Ridge regression model
    ridge_reg = linear_model.Ridge()
    
    # Perform grid search with 10 fold cross-validation, minimizing MSE
    grid_search = GridSearchCV(ridge_reg, param_grid, cv=10, scoring='neg_mean_squared_error')
    grid_search.fit(x, y)
    
    # Get the best estimator from grid search
    best_ridge_reg = grid_search.best_estimator_
    
    # Make predictions using the best estimator
    y_pred = best_ridge_reg.predict(x)
    
    residuals = y - y_pred
    residuals_plot(y_pred, residuals, model_name)
    ridge_summary(best_ridge_reg, grid_search, x_vars, x,y)
    return best_ridge_reg
    
def ridge_summary(ridge_reg, grid_search,x_vars, x, y):
    print('RIDGE MODEL SUMMARY')
    # Print model coefficients
    print("Model Coefficients:")
    for var, coef in zip(x_vars, ridge_reg.coef_):
        print(f"{var}: {coef}")
    
    # Print intercept
    print("Intercept:", ridge_reg.intercept_)
    
    # Calculate predicted values
    y_pred = ridge_reg.predict(x)
    
    # Calculate mean squared error
    mse = mean_squared_error(y, y_pred)
    print("Mean Squared Error (MSE):", mse)
    
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    
    
    # Calculate R-squared
    r_squared = ridge_reg.score(x, y)
    print("R-squared:", r_squared)
    
    # Print the best lambda value
    print("Best Lambda (alpha):", grid_search.best_params_['alpha'])


def residuals_plot(fitted_values,residual,model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(fitted_values, residual, color='blue')
    plt.title(f'Residuals vs Fitted, {model_name}')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True)
    plt.show()


    
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filename = 'ahs2021n.csv'
        forced_binary_vars = ['detached', 'trailer', 'white', 'black', 'asian', 'hispanic', 'hs_grad', 'bachelors', 'grad_school']
        #Lots of data is stored categorically, eg. BLD represents whether a house is detached
        #but also number of units in an apartment, so i break them up into binaries in the data processing
        #each needs to be encoded seperately, so they are handled apart from the binary vars
        binary_vars = ['SOLAR','ADEQUACY','ROOFSAG','ROOFHOLE','NHQSCHOOL','NHQPCRIME','NHQSCRIME','FS']
        sel_vars = ['HINCP_log','YRBUILT', 'BEDROOMS', 'HHMOVE','TOTROOMS','NUMPEOPLE','HHAGE',]
        sel_vars = sel_vars + binary_vars + forced_binary_vars
            
        # Call the functions with the loaded DataFrame
        df = file_setup(filename, sel_vars,binary_vars,forced_binary_vars)
        summary_statistics(df)
        ols_summary(df, sel_vars)
        ridge = ridge(df,sel_vars)
        
    
