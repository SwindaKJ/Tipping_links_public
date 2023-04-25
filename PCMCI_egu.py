#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:18:04 2023

@author: S.K.J. Falkena (s.k.j.falkena@uu.nl)

PCMCI applied to (a subset of) AMOC, ENSO and tradewind variables.

"""


#%% IMPORT

import os
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from matplotlib import pyplot as plt

# To download tigramite: https://github.com/jakobrunge/tigramite
# Check out the tutorials there
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb


#%% FUNCTIONS

def variable_selection(index_selection, exp_id="piControl", 
                       dir_path="/Users/3753808/Library/CloudStorage/" \
                                "OneDrive-UniversiteitUtrecht/Code/" \
                                "Tipping_links_public/CMIP6_indices/"):
    """
    Select the variables to consider for the causal effect network.

    Parameters
    ----------
    index_selection : A list of the variable names.
    exp_id : Optional. The experiment ID. The default is "piControl".
    dir_path : Optional. The path to where the data is stored. The default is 
        "/Users/3753808/Library/CloudStorage/OneDrive-UniversiteitUtrecht/Code/
        Tipping_links/CMIP6_indices/".

    Returns
    -------
    index_path : A list of the directory paths to the files in which each of 
        the variables is stored.
    index_list : A list of all the files in the variable directory (models).
    mod_list0 : A list of all available models for each of the variables.
    mod_list : A list of the models that are available for all variables.

    """
    # Initialise
    index_path = [[] for i in range(len(index_selection))]
    index_list = [[] for i in range(len(index_selection))]
    mod_list0 = [[] for i in range(len(index_selection))]
    # For each variable
    for i in range(len(index_selection)):
        # Set the path to the directory where the data is stored
        index_path[i] = os.path.join( dir_path, index_selection[i], exp_id )
        # Get a list of files in that directory
        index_list[i] = [x for x in os.listdir( index_path[i] ) if x[0] == 'C']
        index_list[i].sort()
        # Match the models (should be the same, but he)
        mod_list0[i] = np.array([filename.split('.')[1]+"_"+filename.split('.')[3] 
                              for filename in index_list[i]])
    
    # Select models which are available for all variables
    mod_list = np.copy(mod_list0[0])
    for i in range(1, len(index_selection)):
        mod_temp = np.array([mod for mod in mod_list0[i] if mod in mod_list])
        mod_list = np.copy(mod_temp)
    
    return index_path, index_list, mod_list0, mod_list


def data_prep(mod, index_selection, index_path, index_list, mod_all_list, 
              aggregation_time, time_data, time_ind=None):
    """
    Load the data for each model (all variables)

    Parameters
    ----------
    mod : The model.
    index_path : A list of the directory paths to the files in which each of 
        the variables is stored.
    index_list : A list of all the files in the variable directory (models).
    mod_all_list : A list of all available models for each of the variables.
    aggregation_time : The time over which to aggregate the data (years).
    time_data : The type of data, annual, seasonal or monthly mean.
    time_ind : Optional. Indicate the month(s) to consider for seasonal and 
        monthly data The default is None.

    Returns
    -------
    data_var : A list containing the timeseries of each of the variables.

    """
    # Initialise data array for yearly, seasonal or monthly data
    data_var = [[] for i in range(len(index_list))]
    # For each variable
    for i in range(len(index_list)):
        # Select file names
        file = np.array(index_list[i])[mod_all_list[i]==mod][0]
        # Create path to file
        filepath = os.path.join(index_path[i], file)
        # Load dataset
        dataset = Dataset(filepath, mode='r')
        # Transfer to xarray
        data_xr = xr.open_dataset(xr.backends.NetCDF4DataStore(dataset))
        # Get names of variables
        var_data = list(data_xr.keys())[0]
        
        # The same for all variable, need to make variable dependent
        if time_data == "year":
            # Moving average over 1 year
            year_series = pd.Series(np.array(data_xr[var_data]))
            year_window = year_series.rolling(12)
            year_movav = np.array(year_window.mean().tolist()[11::])
            # Set new data as moving averages
            data_year = year_movav[0::12]
        elif time_data == "season":
            data_month = [[] for i in range(len(time_ind[i]))]
            # Not crossing into the next year
            if not 0 in time_ind[i] and 12 in time_ind[i]:
                for j in range(len(time_ind[i])):
                    data_month[j] = np.array(data_xr[var_data])[time_ind[i][j]::12]
            else: # Crossing into the next year
                time_ind[i].sort()
                for j in range(len(time_ind[i])):
                    if j == 0 or time_ind[i][j] - time_ind[i][j-1] == 1:
                        data_month[j] = np.array(data_xr[var_data])[time_ind[i][j]+12::12]
                    else:
                        data_month[j] = np.array(data_xr[var_data])[time_ind[i][j]:-11:12]
            # Average over the months belonging to the season of interest            
            data_year = np.mean(np.array(data_month), axis=0)
        elif time_data == "month":
            data_year = np.array(data_xr[var_data])[time_ind[i]::12]
        
        # If index is ENSO, compute absolute value
        if index_selection[i] == "nino34":
            data_year = np.abs(data_year)
        
        # Moving average
        data_series = pd.Series(data_year)
        data_window = data_series.rolling(aggregation_time)
        data_movav = np.array(data_window.mean().tolist()[(aggregation_time-1)::])
        # Select only relevant timesteps
        data_var[i] = data_movav[0::aggregation_time]
    
    return data_var


def data_check(data_var, minlen=100):
    """
    Check the dataset is the same length for all variables, long enough and
    does not contain NaN's.'

    Parameters
    ----------
    data_var : A list containing the timeseries of each of the variables.
    minlen : The minimum length of the time series (in years).

    Returns
    -------
    break_flag : Indication whether the data meets the conditions, True if it
                    does not, False if it does..

    """
    break_flag = False
    # Check all time series are the same length
    data_shape = [data_var[i].shape[0] for i in range(len(data_var))]
    if not all(i == data_shape[0] for i in data_shape):
        break_flag = True
        print("Time series not of the same length.")
    # Time series of sufficient length
    if data_shape[0] < minlen:
        break_flag = True
        print("Time series not long enough, length is: "+repr(data_shape[0]))
    # Check for nan
    for i in range(len(data_var)):
        if np.isnan(data_var[i]).any():
            break_flag = True
            print("NaN in "+ repr(index_selection[i]))
    return break_flag


def data_pcmci(data_var, index_selection, max_lag=10, alpha=0.1):
    """
    Run PCMCI for the selected variables and choices of data.

    Parameters
    ----------
    data_var : A list containing the timeseries of each of the variables.
    index_selection : The names of the variables.
    max_lag : The maximum lag to consider in the network (in years). 
        The default is 10.
    alpha : The significance level for inclusion of links in the network. 
        The default is 0.1.

    Returns
    -------
    results : The PCMCI results.
    names : The names of the variables.

    """
    # Get data as array
    data = np.array(data_var).T
    print(data.shape)
    
    ##### PCMCI #####
    # Get dimensions
    T, N = data.shape
    # Initialize dataframe object, specify time axis and variable names
    names = index_selection
    dataframe = pp.DataFrame(data, 
                             datatime = {0:np.arange(len(data))}, 
                             var_names=names)
    # Set partial correlation as measure
    parcorr = ParCorr(significance='analytic')
    # Initialize PCMCI
    pcmci = PCMCI(
        dataframe=dataframe, 
        cond_ind_test=parcorr,
        verbosity=1)
    # Run PCMCI
    pcmci.verbosity = 1
    results = pcmci.run_pcmci(tau_max=max_lag, pc_alpha=None, alpha_level=alpha)
    
    return results, names


def plot_pcmciresults(results, names, ind=1, save_fig=False, save_name=None, 
                      save_pcmci_fig="/Users/3753808/Library/CloudStorage/" \
                      "OneDrive-UniversiteitUtrecht/Code/Tipping_links_public/" \
                      "PCMCI_results/"):
    """
    Plot the graph and time series graph of the PCMCI result and save the plots
    if desired.

    Parameters
    ----------
    results : The PCMCI results.
    names : The names of the variables.
    ind : Indicates which plots to give, use 0 for only graph.
        The default is 1, giving also the timeseries graph.
    save_fig : Optional. Indicate when to save the figures. 
        The default is False.
    save_name : The name under which to save the figures.
    save_pcmci_fig: The folder in which to save the figures.

    Returns
    -------
    None.

    """
    # Name the sub-folder after the included variables
    save_folder = names[0]
    for i in range(1,len(names)):
        save_folder = save_folder+"_"+names[i]
    # Plot graph
    tp.plot_graph(
        val_matrix=results['val_matrix'],
        graph=results['graph'],
        var_names=names,
        link_colorbar_label='cross-MCI',
        node_colorbar_label='auto-MCI',
        ); plt.title(mod)
    # Save
    if save_fig == True:
        plt.savefig(save_pcmci_fig + save_name+"_resultsgraph.pdf", 
                    format='pdf', dpi=1200)
    else:
        plt.show()
    
    # If want both plots (standard)
    if ind == 1:
        try:
            # Plot time series graph    
            tp.plot_time_series_graph(
                figsize=(6, 4),
                val_matrix=results['val_matrix'],
                graph=results['graph'],
                var_names=names,
                link_colorbar_label='MCI',
                ); plt.title(mod)
            # Save
            if save_fig == True:
                plt.savefig(save_pcmci_fig + save_name+"_timegraph.pdf", 
                            format='pdf', dpi=1200)
            else:
                plt.show()
        except:
            pass
    
    return


#%% SETTINGS

# List of all available variables
index_names = ["amoc26", 
               "nino34", 
               "tradewind_pac_west", 
               "tradewind_pac_central", 
               "tradewind_pac_east", 
               "tradewind_atl"]

# Which variables to include
index_set = [0,1,4]
index_selection = [index_names[i] for i in index_set]

# Annual mean, season, or month: select one from ["year", "season", "month"]
time_data = "season"
# Which month (numbered from January to December), or season (subset of months)
# Can be different for each variable
month = [5,0,11]
season = [[5,6,7],[11,0,1],[8,9,10]]
# Set corresponding time_ind index
if time_data == "year":
    time_ind = None
elif time_data == "season":
    time_ind = season
elif time_data == "month":
    time_ind = month

# Set the aggreation time (in years)
aggregation_time = 5

# Set experiment id (only piCOntrol atm)
exp_id = "piControl"
# Directory where the variables are stored
dir_path = '/Users/3753808/Library/CloudStorage/' \
            'OneDrive-UniversiteitUtrecht/Code/Tipping_links_public/' \
            'CMIP6_indices/'
# Directory to save the figures
save_pcmci_fig = '/Users/3753808/Library/CloudStorage/' \
                 'OneDrive-UniversiteitUtrecht/Code/Tipping_links_public/' \
                 'PCMCI_results/'


##### RUN #####

# Select the variables of interest
index_path, index_list, mod_all_list, mod_list = variable_selection(index_selection)  

# Set the subfolder for saving the data (depending on the selected variables)
save_subfolder = index_selection[0]
for i in range(1, len(index_selection)):
    save_subfolder = save_subfolder+"_"+index_selection[i]
save_subfolder = save_subfolder+"/"
# If the directory does not exist, create it
if not os.path.exists(save_pcmci_fig + save_subfolder):
    os.makedirs(save_pcmci_fig + save_subfolder)
# print(save_subfolder)

for mod in mod_list:
    # print(mod)
    # Set the name for saving the plots
    if time_data == "year":
        save_filename = mod+"_"+time_data+"_aggregation"+repr(aggregation_time)
    else:
        if time_data == "season":
            name_time_ind = "".join(str(j) for j in time_ind[0])
            for i in range(1,len(time_ind)):
                name_time_one = "".join(str(j) for j in time_ind[i])
                name_time_ind = name_time_ind+"-"+name_time_one
            
        elif time_data == "month":
            name_time_ind = str(time_ind[0])
            for i in range(1,len(time_ind)):
                name_time_ind = name_time_ind+"-"+str(time_ind[i])
        save_filename = mod+"_"+time_data+"_"+name_time_ind+"_agg"+ \
                        repr(aggregation_time)
    # print(save_filename)
    # Preprocessing
    data_var = data_prep(mod, index_selection, index_path, index_list, 
                         mod_all_list, aggregation_time, time_data, time_ind)
    # Check conditions
    break_flag = data_check(data_var, 98)
    if not break_flag:
        # Run PCMCI
        results, names = data_pcmci(data_var, index_selection)
        # Plot results
        plotres = plot_pcmciresults(results, names, 0, True, 
                                    save_subfolder+save_filename)

        
