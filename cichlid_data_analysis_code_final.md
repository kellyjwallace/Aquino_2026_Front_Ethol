# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:23:40 2025

@author: eaquinovasquez
"""
# import libraries
import math
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd 
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from datetime import datetime
from statsmodels.stats.multitest import multipletests

# current date
current_date = datetime.now().strftime("%m%d%Y")

# =============================================================================
# DATA INPUT (BEHAVIOR AND IHC)
# =============================================================================
# BEHAVIOR
## import data and restructure data
behavior_large = pd.read_excel('raw_data/BehaviorSize_ReunionPilot_v1.xlsx', 
                               usecols='C,F:J,O')
behavior_large.index = ['T1a','T2a','T3a','T4a','T5a','T6a','T7a','T8a']

behavior_small = pd.read_excel('raw_data/BehaviorSize_ReunionPilot_v1.xlsx', 
                               usecols='C,K:N,P')
behavior_small.index = ['T1b','T2b','T3b','T4b','T5b','T6b','T7b','T8b']

## fill missing variable lateral display for small fish with zeros
behavior_small.insert(4, 'LateralDisplay', [0,0,0,0,0,0,0,0])

## change header name so same for each dataframe; important for when combining
behavior_large = behavior_large.rename(columns={"Large_App_Small":"ApproachInitiated", "Large_Displ_Small":"Displacement", 
                               "Large_Territory_Duration":"TerritoryDuration", "Large_Territory_Duration.1":"Territory_Frequency",
                               "Large_Mass":"Mass"})
behavior_small = behavior_small.rename(columns={"Small_App_Large":"ApproachInitiated", "Small_Displ_Large":"Displacement", 
                               "Small_Territory_Duration":"TerritoryDuration", "Small_Territory_Frequency":"Territory_Frequency",
                               "Small_Mass":"Mass"})

## vertical stack dataframes 
behavior_df = pd.concat([behavior_large, behavior_small])

## create new size column based on mass
behavior_df['Size'] = np.where(behavior_df['Mass'] > 0.156, 'Larger', 'Smaller')

# IHC
IHC_df = pd.read_excel('raw_data/DCX_PS6_Colabeling_cell_count.xlsx', usecols='A:Z')

## average individual slice values to obtain single cell count per animal per cell type
dups = IHC_df.duplicated(subset='Fish', keep=False)
IHC_averaged = IHC_df.groupby(IHC_df['Fish']).mean().reset_index()

## add size column 
IHC_averaged['IHC_Size'] = [
    'Larger' if x.endswith('a') else 'Smaller' for x in IHC_averaged['Fish']]

## sort to match behavior dataframe
IHC_sorted = IHC_averaged.sort_values(by=['IHC_Size']).set_index(behavior_df.index)

## Create Master Dataframe
cichlid = pd.concat([behavior_df,IHC_sorted], axis=1, ignore_index=False)

## Average left and right hemisphere data 
cichlid['dcx_avg'] = (cichlid['dcx_L'] + cichlid['dcx_R'])/2
cichlid['ps_avg'] = (cichlid['ps_L'] + cichlid['ps_R'])/2
cichlid['col_avg'] = (cichlid['co_L'] + cichlid['co_R'])/2
cichlid['col_y/n'] = ['Y' if x > 0 else 'N' for x in cichlid['col_avg']]
cichlid['dcx_avg_1'] = (cichlid['dcx_L_1'] + cichlid['dcx_R_1'])/2
cichlid['dcx_avg_2'] = (cichlid['dcx_L_2'] + cichlid['dcx_R_2'])/2
cichlid['dcx_avg_3'] = (cichlid['dcx_L_3'] + cichlid['dcx_R_3'])/2
cichlid['ps_avg_1'] = (cichlid['ps_L_1'] + cichlid['ps_R_1'])/2
cichlid['ps_avg_2'] = (cichlid['ps_L_2'] + cichlid['ps_R_2'])/2
cichlid['ps_avg_3'] = (cichlid['ps_L_3'] + cichlid['ps_R_3'])/2
cichlid['Territory_Frequency_Binary'] = [int(1) if x > 0 else int(0) for x in cichlid['Territory_Frequency']]

## delete excess columns
extras = ['TankID', 'dcx_L', 'dcx_R',  'ps_L', 'ps_R', 'co_L', 'co_R', 'Fish', 
          'Slice_#', 'IHC_Size', 'dcx_L_1', 'dcx_L_2', 'dcx_L_3', 
          'dcx_R_1', 'dcx_R_2', 'dcx_R_3', 'ps_L_1', 'ps_L_2', 'ps_L_3',
          'ps_R_1', 'ps_R_2', 'ps_R_3', 'co_L_1', 'co_L_2', 'co_L_3', 
          'co_R_1', 'co_R_2', 'co_R_3']
cichlid_final = cichlid.drop(extras, axis=1) 

# =============================================================================
# BEHAVIOR V IHC ANALYSIS GROUP DIFFERENCES
# =============================================================================
# list with behavior and ihc names
behaviors = ['ApproachInitiated', 'Displacement', 'TerritoryDuration', 'LateralDisplay']

IHCs = ['dcx_avg', 'ps_avg', 'dcx_avg_1', 'dcx_avg_2', 'dcx_avg_3', 
        'ps_avg_1', 'ps_avg_2', 'ps_avg_3'] 

target = behaviors + IHCs

# test data distribution 
def run_shapiro(group_df):
    results = {}
    for col in target:
        # Calculate statistic and p-value, ignoring potential NaNs if necessary
        stat, p_val = stats.shapiro(group_df[col].dropna())
        results[f'{col}_stat'] = stat
        results[f'{col}_pvalue'] = p_val
        if p_val <= 0.05:
            results[f'{col}_sig'] = 'yes' 
        else: 
            results[f'{col}_sig'] = 'no' 
    return pd.Series(results)

#  Group by your categorical column and apply the function
shapiro_results_beh = cichlid_final.groupby('Size').apply(run_shapiro, include_groups=False)

print(shapiro_results_beh)

# split data into groups by size, replace na with zeros 
l_group = cichlid_final.groupby('Size').get_group('Larger').drop('Size', axis=1)
s_group = cichlid_final.groupby('Size').get_group('Smaller').drop('Size', axis=1)

# cohen's d formula
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    # Pooled standard deviation
    pool_sd = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pool_sd

# T-test
wilcox_size_results = {}
p_values_behav = []
p_values_ps = []
p_values_dcx = []
for col in l_group.columns:
    if col == 'col_y/n' or col == 'col_avg' or col =='Territory_Frequency_Binary' or col=='Territory_Frequency' or col == 'Mass':
        pass
    else:
        print(col) 
        # ttest with 2 independent variables
        wstats, pvalue = stats.wilcoxon(l_group[col], s_group[col], axis=0,
                                         nan_policy='omit')
        # cohen's d 
        d = cohen_d(l_group[col],s_group[col])
        
        # save stats to dictionary
        wilcox_size_results[col]= {'wilcox_stats':wstats, 'p_values': pvalue, 
                             'cohen_d': d}
        if 'Territory' in col:
            p_values_behav.append(pvalue)
        elif 'dcx' in col:
            if any(char.isdigit() for char in col) == True:
                p_values_dcx.append(pvalue)
            else:
                pass
        elif 'ps' in col:
            if any(char.isdigit() for char in col) == True:
                p_values_ps.append(pvalue)
            else: 
                pass
        
# multi-test correction: size vs IHC (FDR correction)
reject, pvals_corrected_dcx, _, _ = multipletests(p_values_dcx, alpha = 0.05, 
                                  method='fdr_bh')
reject, pvals_corrected_ps, _, _ = multipletests(p_values_ps, alpha = 0.05, 
                                  method='fdr_bh')

# multi-test correction: size vs behavior (FDR correction)
reject, pvals_corrected_behav, _, _ = multipletests(p_values_behav, alpha = 0.05, 
                                  method='fdr_bh')

# add the adjusted p values to t-results 
i = 0 
ii = 0 
iii = 0
for key in wilcox_size_results:
   if 'ps' in key:
      if any(char.isdigit() for char in key) == True:
           wilcox_size_results[key].update({'p_adjusted': pvals_corrected_ps[i]})
           i += 1
      else:
           pass
   elif 'dcx' in key:
      if any(char.isdigit() for char in key) == True:
          wilcox_size_results[key].update({'p_adjusted': pvals_corrected_dcx[ii]})
          ii += 1
      else:
          pass
   elif 'Territory' in key:
       wilcox_size_results[key].update({'p_adjusted': pvals_corrected_behav[iii]})
       iii += 1
      
# boxplot (size v ihc/behavior)
for key in wilcox_size_results:
    # store p value for graphs
    if 'p_adjusted' in wilcox_size_results[key]:
        p = wilcox_size_results[key]['p_adjusted']
    else:
        p = wilcox_size_results[key]['p_values']
    
    # labels for y axis
    if key == 'ps_avg':
        ylabel = 'Total pS6+ Cell Count\n(DC+Dm3+Vsm)'
    elif key == 'dcx_avg':
        ylabel = 'Total DCX+ Cell Count\n(DC+Dm3+Vsm)'
    elif key == 'dcx_avg_1':
        ylabel = 'DC DCX+ Cell Count'
    elif key == 'dcx_avg_2':
        ylabel = 'Dm3 DCX+ Cell Count'
    elif key == 'dcx_avg_3':
        ylabel = 'Vsm DCX+ Cell Count'
    elif key == 'ps_avg_1':
        ylabel = 'DC pS6+ Cell Count'
    elif key == 'ps_avg_2':
        ylabel = 'Dm3 pS6+ Cell Count'
    elif key == 'ps_avg_3':
        ylabel = 'Vsm pS6+ Cell Count'
    elif key == 'ApproachInitiated':
        ylabel = 'Number of Initiated Approaches'
    elif key == 'TerritoryDuration':
        ylabel = 'Time in Territory(s)'
    elif key == 'LateralDisplay':
        ylabel = 'Number of Lateral Displays'
    elif key == 'Displacement':
        ylabel = 'Number of Displacements'
        
    # plot
    fig, ax = plt.subplots()
    ax = sns.boxplot(cichlid_final, x='Size', y=f'{key}', hue='Size',
                gap=0.1, palette=('#EAC117','#E56717'), linewidth=1.5, 
                showcaps=False, legend=False, showfliers=False)
    ax = sns.stripplot(x='Size', y=f'{key}', data=cichlid_final, 
                         color="k", alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(which='major', labelsize=14)
    fig.set_size_inches(4, 5)
    ax.set_ylabel(f'{ylabel}', fontsize=18)
    ax.set_xlabel(None)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    ymax = max(x for x in cichlid_final[f'{key}'] if not math.isnan(x))
    # p value 
    if math.isnan(p):
        ax.text(xlims[0]+0.83, ymax+0.18, 'n.s.', fontsize=10) 
    else:
        ax.text(xlims[0]+0.83, ymax+0.18, f'p={round(p, 4):.4f}', fontsize=10)
    ax.hlines(ymax+0.1, xlims[0]+0.5, xlims[1]-0.5, colors='k', lw=1.5)
    # sample size
    large_n = l_group[key].count()
    small_n = s_group[key].count()
    ax.text(xlims[0]+0.40, ylims[0]+0.05, f'N={large_n}')
    ax.text(xlims[1]-0.60, ylims[0]+0.05, f'N={small_n}')
    plt.savefig(f'graphs/{key}_relativesize_{current_date}.tiff', dpi=300, 
               bbox_inches='tight')
    plt.show() 
    
# =============================================================================
# FREQUENCY IN TERRITORY V EVERYTHING ELSE GROUP DIFFERENCES
# =============================================================================
# test distribution
behaviors_2 = ['ApproachInitiated', 'Displacement', 'TerritoryDuration', 
             'LateralDisplay']

target = behaviors_2 + IHCs
shapiro_results_freq = cichlid_final.groupby('Territory_Frequency_Binary').apply(run_shapiro, include_groups=False)

print(shapiro_results_freq)

# split data into groups by territory freq 
no_entry_t = cichlid_final.groupby('Territory_Frequency_Binary').get_group(int(0)).drop('Territory_Frequency_Binary', axis=1)
entry_t = cichlid_final.groupby('Territory_Frequency_Binary').get_group(int(1)).drop('Territory_Frequency_Binary', axis=1)   

# Mann Whitney U test 
mann_freq_results = {}
dcx=[]
ps=[]
for col_y in target:
    # t test w/ two ind variables
    mstats, pvalue = stats.mannwhitneyu(no_entry_t[col_y], entry_t[col_y], axis=0,
                                     nan_policy='omit')
    # cohen's d 
    d = cohen_d(no_entry_t[col_y], entry_t[col_y])
    
    ## save stats to dictionary
    mann_freq_results[col_y]= {'mann_stats': mstats, 'p_values': pvalue, 
                                 'cohen_d': d}
    if 'dcx' in col_y:
        if any(char.isdigit() for char in col_y) == True:
            dcx.append(pvalue)
         
    if 'ps' in col_y:
        if any(char.isdigit() for char in col_y) == True:
            ps.append(pvalue)
    else:
        pass

# add lists to a single dictionary 
pval_freq_ihc = {'dcx': dcx, 'ps': ps}

# multi-test correction: IHC region v behavior (FDR correction)
## dcx correction 
reject, corrected_p_dcx, _, _ = multipletests(pval_freq_ihc['dcx'], 
                                              alpha = 0.05, method='fdr_bh')

## add adjusted p values to dictionary
for x in range(len(pval_freq_ihc['dcx'])):
    mann_freq_results[f'dcx_avg_{x+1}']['p_adjusted']=corrected_p_dcx[x]
   
## ps6 correction
reject, corrected_p_ps, _, _ = multipletests(pval_freq_ihc['ps'],
                                             alpha = 0.05, method='fdr_bh')
   
## add adjusted p values to dictionary
for x in range(len(pval_freq_ihc['ps'])):
    mann_freq_results[f'ps_avg_{x+1}']['p_adjusted']=corrected_p_ps[x]
    
group_results = {'territory_entry': mann_freq_results}

# boxplots
i = 0  
for key in group_results['territory_entry']:
    # store p value for graphs
    if 'p_adjusted' in group_results['territory_entry'][key]:
        p = group_results['territory_entry'][key]['p_adjusted']
    else:
        p = group_results['territory_entry'][key]['p_values']  
    # labels for y axis
    if key == 'ps_avg':
        ylabel = 'Total pS6+ Cell Count\n(DC+Dm3+Vsm)'
    elif key == 'dcx_avg':
        ylabel = 'Total DCX+ Cell Count\n(DC+Dm3+Vsm)'
    elif key == 'dcx_avg_1':
        ylabel = 'DC DCX+ Cell Count'
    elif key == 'dcx_avg_2':
        ylabel = 'Dm3 DCX+ Cell Count'
    elif key == 'dcx_avg_3':
        ylabel = 'Vsm DCX+ Cell Count'
    elif key == 'ps_avg_1':
        ylabel = 'DC pS6+ Cell Count'
    elif key == 'ps_avg_2':
        ylabel = 'Dm3 pS6+ Cell Count'
    elif key == 'ps_avg_3':
        ylabel = 'Vsm pS6+ Cell Count'
    elif key == 'ApproachInitiated':
        ylabel = 'Number of Initiated Approaches'
    elif key == 'TerritoryDuration':
        ylabel = 'Time in Territory(s)'
    elif key == 'LateralDisplay':
        ylabel = 'Number of Lateral Displays'
    elif key == 'Displacement':
        ylabel = 'Number of Displacements'
        
    # plot
    fig, ax = plt.subplots()
    ax = sns.boxplot(cichlid_final, x='Territory_Frequency_Binary', y=f'{key}',
                     hue='Territory_Frequency_Binary',gap=0.1, 
                     palette=('#3090C7','#2B547E'),  linewidth=1.5,
                     showcaps=False, legend=False, showfliers=False)
    ax = sns.stripplot(x='Territory_Frequency_Binary', y=f'{key}', data=cichlid_final, 
                         color="k", alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(which='major', labelsize=14)
    fig.set_size_inches(4, 5)
    ax.set_ylabel(f'{ylabel}', fontsize=18)
    ax.set_xlabel(None)
    ax.set_xticks([0,1], ['No Entry', 'Entry'])
    xlims = ax.get_xlim()
    ymax = max(cichlid_final[f'{key}'])
    ylims = ax.get_ylim()
    # p value 
    ax.text(xlims[0]+0.83, ymax+0.18, f'p={round(p, 4):.4f}', fontsize=10)
    ax.hlines(ymax+0.1, xlims[0]+0.5, xlims[1]-0.5, colors='k', lw=1.5)
    # sample size 
    no_n = no_entry_t[key].count()
    entry_n = entry_t[key].count()
    ax.text(xlims[0]+0.40, ylims[0], f'N={no_n}')
    ax.text(xlims[1]-0.60, ylims[0], f'N={entry_n}')
    plt.savefig(f'graphs/freq_{key}_{current_date}.tiff', dpi=300, 
                bbox_inches='tight')
    plt.show()
    i+=1

# =============================================================================
# BEHAVIOR V IHC ANALYSIS CORRELATIONS
# =============================================================================f
# new list with no freq territory data
behaviors_bin = ['ApproachInitiated', 'Displacement', 'TerritoryDuration', 
                 'LateralDisplay']

# GLM regression
lreg_behav_ihc = {}
pval_behav_ihc = {}
df_predictions = {}
summary_frames = {}
for col_x in behaviors_bin:
    pval_behav_ihc[col_x] = {}
    dcx=[]
    ps=[]
    for col_y in IHCs:
        # regression:    
        reg_result = smf.glm(formula=f"{col_y} ~ {col_x}", data=cichlid_final, 
                             missing='drop', family=sm.families.Poisson()).fit()
        # calc f value 
        f_reg = reg_result.f_test(f'{col_x}=0') 
        # generate grid for predicted values for plotting 
        x_grid = np.linspace(cichlid_final[col_x].min(), cichlid_final[col_x].max(), 100)
        df_predictions[f'{col_x}_v_{col_y}'] = pd.DataFrame({col_x: x_grid})
        #Calculate predicted values for plotting
        prediction_obj = reg_result.get_prediction(df_predictions[f'{col_x}_v_{col_y}'])
        summary_frames[f'{col_x}_v_{col_y}'] = prediction_obj.summary_frame()
        # save stats to dictionary
        lreg_behav_ihc[f'{col_x}_v_{col_y}'] = {'rsquared': reg_result.pseudo_rsquared(), 
                                         'coeff': {'intercept': reg_result.params.iloc[0],
                                                   f'{col_x}': reg_result.params.iloc[1]},
                                         'pvalues': {'intercept': reg_result.pvalues.iloc[0],
                                                     f'{col_x}': reg_result.pvalues.iloc[1]},
                                         'fstats': {'fvalue': f_reg.fvalue, 
                                                    'df_denom': f_reg.df_denom, 
                                                    'df_num': f_reg.df_num}}
        if 'dcx' in col_y:
            if any(char.isdigit() for char in col_y) == True:
                dcx.append(reg_result.pvalues.iloc[1])
             
        if 'ps' in col_y:
            if any(char.isdigit() for char in col_y) == True:
                ps.append(reg_result.pvalues.iloc[1])
        else:
            pass
        pval_behav_ihc[col_x] = {'dcx': dcx, 'ps': ps}

# multi-test correction: IHC region v behavior (FDR correction)
for p in pval_behav_ihc:
    # dcx correction 
    reject, corrected_p_dcx, _, _ = multipletests(pval_behav_ihc[f'{p}']['dcx'], 
                                                  alpha = 0.05, method='fdr_bh')
    
    ## add adjusted p values to dictionary
    for x in range(len(pval_behav_ihc[p]['dcx'])):
        lreg_behav_ihc[f'TerritoryDuration_v_dcx_avg_{x+1}']['p_adjusted']=corrected_p_dcx[x]
   
    # ps6 correction
    reject, corrected_p_ps, _, _ = multipletests(pval_behav_ihc[p]['ps'],
                                                 alpha = 0.05, method='fdr_bh')
   
    ## add adjusted p values to dictionary
    for x in range(len(pval_behav_ihc[p]['ps'])):
        lreg_behav_ihc[f'{p}_v_ps_avg_{x+1}']['p_adjusted']=corrected_p_ps[x]
    
# plot
j = 0 
for col_x in behaviors_bin:
    for col_y in IHCs:
        # labels based on column names
        if col_y == 'ps_avg':
            y = 'Total pS6+ Cell Count\n(DC+Dm3+Vsm)'
        elif col_y == 'dcx_avg':
            y = 'Total DCX+ Cell Count\n(DC+Dm3+Vsm)'
        elif col_y == 'dcx_avg_1':
            y = 'DC DCX+ Cell Count'
        elif col_y == 'dcx_avg_2':
            y = 'Dm3 DCX+ Cell Count'
        elif col_y == 'dcx_avg_3':
            y = 'Vsm DCX+ Cell Count'
        elif col_y == 'ps_avg_1':
            y = 'DC pS6+ Cell Count'
        elif col_y == 'ps_avg_2':
            y = 'Dm3 pS6+ Cell Count'
        elif col_y == 'ps_avg_3':
            y = 'Vsm pS6+ Cell Count'
        if col_x == 'ApproachInitiated':
            x = 'Initiated Approaches'
        elif col_x == 'TerritoryDuration':
            x = 'Time in Territory (s)'
        elif col_x == 'LateralDisplay':
            x = 'Lateral Displays'
        else:
            x = 'Displacements'

        # add p value to graph
        if 'p_adjusted' in lreg_behav_ihc[f'{col_x}_v_{col_y}']:
            p = lreg_behav_ihc[f'{col_x}_v_{col_y}']['p_adjusted']
        else:
            p = lreg_behav_ihc[f'{col_x}_v_{col_y}']['pvalues'][f'{col_x}']
        
        # plot
        #raw data points
        ax = sns.scatterplot(data=cichlid_final, x=f'{col_x}', y=f'{col_y}', color='k')
        # Plot the GLM regression line
        plt.plot(df_predictions[f'{col_x}_v_{col_y}'], summary_frames[f'{col_x}_v_{col_y}']['mean'], 
                 color='orange', lw=2)
        ax.set_ylabel(f'{y}', fontsize=18, wrap=True)
        ax.set_xlabel(f'{x}', fontsize=18)
        xlims = plt.xlim() 
        ylims = plt.ylim()
        ax.text(xlims[1], ylims[1], f'p={round(p, 4):.4f}', fontsize=12, color='k')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)  
        plt.savefig(f'graphs/regplot_{x}_v_{col_y}_{current_date}.tiff', dpi=300, 
                    bbox_inches='tight')        
        plt.show()
        j += 1 

# =============================================================================
# DIFF IN MASS V DIFF IN EVERYTHING ELSE ANALYSIS
# =============================================================================
# create new dataframe with difference
cichlid_no_col = cichlid_final.drop(['col_y/n', 'col_avg', 'Territory_Frequency'], axis=1)
cichlids_large = cichlid_no_col.groupby('Size').get_group('Larger').drop('Size',
                                                                        axis=1).reset_index(drop=True)
cichlids_small = cichlid_no_col.groupby('Size').get_group('Smaller').drop('Size',
                                                                        axis=1).reset_index(drop=True)
diff_size = abs(cichlids_large.sub(cichlids_small, fill_value=0))

# Regression 
lreg_diff = {}
pval_mass_diff = {}
territory = []
dcx=[]
ps=[]
df_predictions = {}
summary_frames = {}
for col in diff_size:
    if col != 'Mass':
        # regression 
        reg_result = smf.glm(formula=f"{col} ~ Mass", data=diff_size, 
                             missing='drop').fit()
        # calc f value
        f_reg = reg_result.f_test('Mass=0')
        
        # generate grid for predicted values for plotting 
        x_grid = np.linspace(diff_size['Mass'].min(), diff_size['Mass'].max(), 100)
        df_predictions[f'mass_v_{col}'] = pd.DataFrame({'Mass': x_grid})
        #Calculate predicted values for plotting
        prediction_obj = reg_result.get_prediction(df_predictions[f'mass_v_{col}'])
        summary_frames[f'mass_v_{col}'] = prediction_obj.summary_frame()

        # save stats to dictionary
        lreg_diff[f'mass_diff_v_diff_{col}'] = {'rsquared': reg_result.pseudo_rsquared(), 
                                         'coeff': {'intercept': reg_result.params.iloc[0],
                                                   'mass': reg_result.params.iloc[1]},
                                         'pvalues': {'intercept': reg_result.pvalues.iloc[0],
                                                     'mass': reg_result.pvalues.iloc[1]},
                                         'fstats': {'fvalue': f_reg.fvalue, 
                                                    'df_denom': f_reg.df_denom, 
                                                    'df_num': f_reg.df_num}}
        if 'dcx' in col:
            if any(char.isdigit() for char in col) == True:
                dcx.append(reg_result.pvalues.iloc[1]) 
        elif 'ps' in col:
            if any(char.isdigit() for char in col) == True:
                ps.append(reg_result.pvalues.iloc[1])
    pval_mass_diff['Mass'] = {'dcx': dcx, 'ps': ps}

# multi-test correction
for p in pval_mass_diff['Mass']:
    # FDR correction 
    reject, corrected_p, _, _ = multipletests(pval_mass_diff['Mass'][p], alpha = 0.05, 
                                  method='fdr_bh')
    
    ## add adjusted p values to dictionary
    for x in range(len(pval_mass_diff['Mass'][p])):
        if 'dcx' in p:
            lreg_diff[f'mass_diff_v_diff_dcx_avg_{x+1}']['p_adjusted']=corrected_p[x]
        elif 'ps' in p:
            lreg_diff[f'mass_diff_v_diff_ps_avg_{x+1}']['p_adjusted']=corrected_p[x]

# plot        
k = 0
for col in diff_size:
    if col == 'Mass': # mass will always be the independent variable 
        pass
    else:
        # ylabels for graph
        if col == 'ps_avg':
            y = 'Total pS6+ Cell Count\n(DC+Dm3+Vsm)'
        elif col == 'dcx_avg':
            y = 'Total DCX+ Cell Count\n(DC+Dm3+Vsm)'
        elif col == 'dcx_avg_1':
            y = 'DC DCX+ Cell Count'
        elif col == 'dcx_avg_2':
            y = 'Dm3 DCX+ Cell Count'
        elif col == 'dcx_avg_3':
            y = 'Vsm DCX+ Cell Count'
        elif col == 'ps_avg_1':
            y = 'DC pS6+ Cell Count'
        elif col == 'ps_avg_2':
            y = ' Dm3 pS6+ Cell Count'
        elif col == 'ps_avg_3':
            y = 'Vsm pS6+ Cell Count'
        elif col == 'ApproachInitiated':
            y = 'Initiated Approaches'
        elif col == 'TerritoryDuration':
            y = 'Time in Territory(s)'
        elif col == 'Territory_Frequency_Binary':
            y = 'Entry into Territory'
        elif col == 'LateralDisplay':
            y = 'Lateral Displays'
        elif col == 'Displacement':
            y = 'Displacements'
            
        # pvalue
        if 'p_adjusted' in lreg_diff[f'mass_diff_v_diff_{col}']:
            p = lreg_diff[f'mass_diff_v_diff_{col}']['p_adjusted']
        else:
            p = lreg_diff[f'mass_diff_v_diff_{col}']['pvalues']['mass']
            
        #plot
        #raw data points
        ax = sns.scatterplot(data=diff_size, x='Mass', y=f'{col}', color='k')
        # Plot the GLM regression line
        plt.plot(df_predictions[f'mass_v_{col}'], summary_frames[f'mass_v_{col}']['mean'], 
                 color='orange', lw=2)
        ax.set_ylabel(f'Difference in {y}', fontsize=18, wrap=True)
        ax.set_xlabel('Difference in Mass(g)', fontsize=18)
        xlims = plt.xlim() 
        ylims = plt.ylim()
        ax.text(xlims[1], ylims[1], f'p={round(p, 4):.4f}', fontsize=12, color='k')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)       
        plt.savefig(f'graphs/lreg_mass_{col}_{current_date}.tiff', dpi=300, 
                    bbox_inches='tight')        
        plt.show()
        k += 1

# =============================================================================
# CORTISOL DATA INPUT AND RESTRUCTURING
# =============================================================================
#import data
cort_data_all = pd.read_excel('raw_data/Cortisol_dilution_corrections_DCX_v1.xlsx')

# remove all unwanted columns and form new dataframe
cort_data_size = cort_data_all[['RelativeSize', 'Cortisol_pg_mL_hr_g']]

# reorganize so larger is first then smaller fish 
l_cort = cort_data_size.groupby('RelativeSize').get_group('Larger')
s_cort = cort_data_size.groupby('RelativeSize').get_group('Smaller')
index = cichlid_final.index
size_cort = pd.concat([l_cort, s_cort]).set_index(index)

# add cort data to master dataframe
cichlid_cort = pd.concat([cichlid_final, size_cort], axis=1) 
column_names = behaviors + IHCs

# =============================================================================
# CORTISOL LEVELS V SIZE/ZONE ENTRY
# =============================================================================
#  Group by your categorical column and apply the function
target = ['Cortisol_pg_mL_hr_g']
shapiro_results_cort_size = cort_data_size.groupby('RelativeSize').apply(run_shapiro, include_groups=False)

print(shapiro_results_cort_size)

## T-test
# cort v size 
cort_tstats, cort_pvalue = stats.ttest_ind(l_cort['Cortisol_pg_mL_hr_g'], 
                                           s_cort['Cortisol_pg_mL_hr_g'])
# cohen's d 
d = cohen_d(l_cort['Cortisol_pg_mL_hr_g'], s_cort['Cortisol_pg_mL_hr_g'])

# input into ttest dataframe 
cort_ttest_results = {'Relative_Size': {'t_stats': cort_tstats, 'p_values': cort_pvalue,
                           'cohen_d': d}}

# restructure for cort v territory entry 
entry_cort = cichlid_cort.groupby('Territory_Frequency_Binary').get_group(int(1))
no_entry_cort = cichlid_cort.groupby('Territory_Frequency_Binary').get_group(int(0))

# Group by your categorical column and apply the function
target = ['Cortisol_pg_mL_hr_g']
shapiro_results_cort_entry = cichlid_cort.groupby('Territory_Frequency_Binary').apply(run_shapiro, include_groups=False)

print(shapiro_results_cort_entry)

# cort v zone entry
cort_bin_tstats, cort_bin_pvalue = stats.ttest_ind(entry_cort['Cortisol_pg_mL_hr_g'], 
                                           no_entry_cort['Cortisol_pg_mL_hr_g'])

# cohen's d 
d = cohen_d(entry_cort['Cortisol_pg_mL_hr_g'], no_entry_cort['Cortisol_pg_mL_hr_g'])

# input into ttest dataframe 
cort_ttest_results['Territory_Frequency_Binary'] = {'t_stats': cort_bin_tstats, 
                                            'p_values': cort_bin_pvalue,
                                            'cohen_d': d}

# boxplots
cort_tt = ['RelativeSize', 'Territory_Frequency_Binary']
size_pal = ['#EAC117','#E56717']
tert_pal = ['#3090C7','#2B547E']

for x in cort_tt:
    if x == 'RelativeSize':
        palette = size_pal
        pvalue = cort_pvalue
        l_n = l_cort[x].count()
        s_n = s_cort[x].count() 
    else: 
        palette = tert_pal
        pvalue = cort_bin_pvalue
        l_n = entry_cort[x].count()
        s_n = no_entry_cort[x].count() 
    
    fig14, ax14 = plt.subplots()
    ax14 = sns.boxplot(cichlid_cort, x=f'{x}', y='Cortisol_pg_mL_hr_g', hue=f'{x}',
                gap=0.1, palette=palette, flierprops={'mfc':'k'}, 
                linewidth=1.5, showcaps=False, showfliers=False)
    ax14 = sns.stripplot(x=f'{x}', y='Cortisol_pg_mL_hr_g', data=cichlid_cort, 
                         color="k", alpha=0.5)
    ax14.spines[['top', 'right']].set_visible(False)
    ax14.tick_params(which='major', labelsize=14)
    fig14.set_size_inches(4, 5)
    ax14.set_ylabel('Cortisol (pg/ml/h/g)', fontsize=18)
    ax14.set_xlabel(None)
    xlims = ax14.get_xlim()
    ylims = ax14.get_ylim()
    # p value
    ax14.text(xlims[0]+0.83, ylims[1]-0.15, f'p={round(pvalue, 4):.4f}', fontsize=10)
    ax14.hlines(ylims[1]-10, xlims[0]+0.5, xlims[1]-0.5, colors='k', lw=1.5)
    ax14.text(xlims[0]+0.40, ylims[0], f'N={l_n}')
    ax14.text(xlims[1]-0.60, ylims[0], f'N={s_n}')
    plt.savefig(f'graphs/cort_{x}_{current_date}.tiff', dpi=300, 
                bbox_inches='tight')
    plt.show()

# =============================================================================
# CORTISOL AND BEHAVIOR AND IHC DATA ANALYSIS (minus territory freq)
# =============================================================================
# regression
lreg_cort = {}
pval_cort = {}
territory = []
dcx=[]
ps=[]
df_predictions = {}
summary_frames = {}
cort_glm = {}
for y in column_names:
    if y in behaviors:
        if "Territory" not in y:
            # regression 
            reg_result = smf.glm(formula=f"Cortisol_pg_mL_hr_g ~ {y}", data=cichlid_cort, missing='drop', family=sm.families.Poisson()).fit()
            # rsquared value
            r2 = reg_result.pseudo_rsquared()
            test = 'Poisson GLM'
        elif 'Binary' in y: 
            # regression 
            reg_result = smf.ols(formula=f"Cortisol_pg_mL_hr_g ~ {y}", data=cichlid_cort, missing='drop').fit()
            test = 'Ordinary Least Squared Regression'
            # rsquared value
            r2 = reg_result.rsquared
        else: 
            # regression 
            reg_result = smf.glm(formula=f"Cortisol_pg_mL_hr_g ~ {y}", data=cichlid_cort, missing='drop').fit()
            test = 'GLM '
            # rsquared value
            r2 = reg_result.pseudo_rsquared()
        # calc f value 
        f_reg = reg_result.f_test(f'{y}=0') 
        # generate grid for predicted values for plotting 
        x_grid = np.linspace(cichlid_cort[y].min(), cichlid_cort[y].max(), 100)
        df_predictions[f'cort_v_{y}'] = pd.DataFrame({f'{y}': x_grid})
        #Calculate predicted values for plotting
        prediction_obj = reg_result.get_prediction(df_predictions[f'cort_v_{y}'])
        summary_frames[f'cort_v_{y}'] = prediction_obj.summary_frame()
        
    else:
        # regression
        reg_result = smf.glm(formula=f"{y} ~ Cortisol_pg_mL_hr_g", data=cichlid_cort,
                         missing='drop', family=sm.families.Poisson()).fit()
        test = 'Poisson GLM'
        # rsquared value
        r2 = reg_result.pseudo_rsquared()
        # calc f value
        f_reg = reg_result.f_test('Cortisol_pg_mL_hr_g=0')
        # generate grid for predicted values for plotting 
        x_grid = np.linspace(cichlid_cort['Cortisol_pg_mL_hr_g'].min(), cichlid_cort['Cortisol_pg_mL_hr_g'].max(), 100)
        df_predictions[f'cort_v_{y}'] = pd.DataFrame({'Cortisol_pg_mL_hr_g': x_grid})
        #Calculate predicted values for plotting
        prediction_obj = reg_result.get_prediction(df_predictions[f'cort_v_{y}'])
        summary_frames[f'cort_v_{y}'] = prediction_obj.summary_frame()
    
    # save stats to a dictionary 
    lreg_cort[f'cort_v_{y}'] = {'rsquared': r2, 
                                     'coeff': {'intercept': reg_result.params.iloc[0],
                                               'cort': reg_result.params.iloc[1]},
                                     'pvalues': {'intercept': reg_result.pvalues.iloc[0],
                                                 'cort': reg_result.pvalues.iloc[1]},
                                     'fstats': {'fvalue': f_reg.fvalue, 
                                                'df_denom': f_reg.df_denom, 
                                                'df_num': f_reg.df_num}}
     
    if 'dcx' in y:
        if any(char.isdigit() for char in y) == True:
            dcx.append(reg_result.pvalues.iloc[1]) 
    elif 'ps' in y:
        if any(char.isdigit() for char in y) == True:
            ps.append(reg_result.pvalues.iloc[1])
    pval_cort['Cort'] = {'dcx': dcx, 'ps': ps}

## multi-test correction
for p in pval_cort['Cort']:
    # bonferonni correction 
    reject, corrected_p, _, _ = multipletests(pval_cort['Cort'][f'{p}'], alpha = 0.05, 
                                  method='fdr_bh')
    ## add adjusted p values to dictionary
    for x in range(len(pval_cort['Cort'][f'{p}'])):
        if 'dcx' in p:
            lreg_cort[f'cort_v_dcx_avg_{x+1}']['p_adjusted']=corrected_p[x]
        elif 'ps' in p:
            lreg_cort[f'cort_v_ps_avg_{x+1}']['p_adjusted']=corrected_p[x]

# plot
m = 0
for col in column_names:
    # ylabels for graph
    if col == 'ps_avg':
        y = 'Total pS6+ Cell Count\n(DC+Dm3+Vsm)'
    elif col == 'dcx_avg':
        y = 'Total DCX+ Cell Count\n(DC+Dm3+Vsm)'
    elif col == 'dcx_avg_1':
        y = 'DC DCX+ Cell Count'
    elif col == 'dcx_avg_2':
        y = 'Dm3 DCX+ Cell Count'
    elif col == 'dcx_avg_3':
        y = 'Vsm DCX+ Cell Count'
    elif col == 'ps_avg_1':
        y = 'DC pS6+ Cell Count'
    elif col == 'ps_avg_2':
        y = ' Dm3 pS6+ Cell Count'
    elif col == 'ps_avg_3':
        y = 'Vsm pS6+ Cell Count'
    elif col == 'ApproachInitiated':
        y = 'Initiated Approaches'
    elif col == 'TerritoryDuration':
        y = 'Time in Territory(s)'
    elif col == 'LateralDisplay':
        y = 'Lateral Displays'
    elif col == 'Displacement':
        y = 'Displacements'
    elif col == 'Territory_Frequency_Binary':
        y = 'Entry into Territory'
        
    # pvalue
    if 'p_adjusted' in lreg_cort[f'cort_v_{col}']:
        p = lreg_cort[f'cort_v_{col}']['p_adjusted']
    else:
        p = lreg_cort[f'cort_v_{col}']['pvalues']['cort']
   
    #plot
    plt.figure(figsize = (4, 5))
    if col in behaviors:
        #raw data points
        ax = sns.scatterplot(data=cichlid_cort, x=f'{col}', y='Cortisol_pg_mL_hr_g', 
                        hue='RelativeSize', palette=('#EAC117','#E56717'))
        # Plot the GLM regression line
        plt.plot(df_predictions[f'cort_v_{col}'], summary_frames[f'cort_v_{col}']['mean'], 
                 color='black', lw=2)
        ax.set_xlabel(f'{y}', fontsize=18)
        ax.set_ylabel('Cortisol Levels (pg/mL/hr/g)', fontsize=18)
    else:
        #raw data points
        ax = sns.scatterplot(data=cichlid_cort, x='Cortisol_pg_mL_hr_g', y=f'{col}', 
                        hue='RelativeSize', palette=('#EAC117','#E56717'))
        # Plot the GLM regression line
        plt.plot(df_predictions[f'cort_v_{col}'], summary_frames[f'cort_v_{col}']['mean'], 
                 color='black', lw=2)
        ax.set_ylabel(f'{y}', fontsize=18)
        ax.set_xlabel('Cortisol Levels (pg/mL/hr/g)', fontsize=18)
    
    xlims = plt.xlim() 
    ylims = plt.ylim()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title='Relative Size', fontsize=12, frameon=False, borderpad=0.1)
    ax.text(xlims[1], ylims[1], f'p={round(p, 4):.4f}', fontsize=12, color='k')
    plt.savefig(f'graphs/lreg_cort_{col}_{current_date}.tiff', dpi=300, 
                bbox_inches='tight')        
    plt.show()
    m += 1

# =============================================================================
# Cortisol (High v Low Groups)
# =============================================================================
# insert low and high cort column 
cichlid_cort['Cort_H_L'] = np.where(cichlid_cort['Cortisol_pg_mL_hr_g'] > 250, 'High', 'Low')

# split dataframe based on high/low groups 
high_c = cichlid_cort.groupby('Cort_H_L').get_group('High')
low_c = cichlid_cort.groupby('Cort_H_L').get_group('Low')

#  Group by your categorical column and apply the function
target = behaviors + IHCs

shapiro_results_cort_rel = cichlid_cort.groupby('Cort_H_L').apply(run_shapiro, include_groups=False)

print(shapiro_results_cort_rel)

# T-test
p_values_behav_cort = []
p_values_ps_cort = []
p_values_dcx_cort = []
wilcox_relcort_results = {}
rel_cort = {}
for col in high_c.columns:
    if col not in target:
        pass
    else: 
        # ttest with 2 independent variables
        wstats, pvalue = stats.wilcoxon(high_c[col], low_c[col], axis=0,
                                         nan_policy='omit')
        # cohen's d 
        d = cohen_d(high_c[col],low_c[col])
        
        # save stats to dictionary
        wilcox_relcort_results[col]={'wilcox_stats': wstats, 'p_values': pvalue, 
                             'cohen_d': d}
        
        
        if 'Territory' in col:
            p_values_behav_cort.append(pvalue)
        elif 'dcx' in col:
            if any(char.isdigit() for char in col) == True:
                p_values_dcx_cort.append(pvalue)
            else:
                pass
        elif 'ps' in col:
            if any(char.isdigit() for char in col) == True:
                p_values_ps_cort.append(pvalue)
            else: 
                pass
     
# multi-test correction: size vs IHC (FDR correction)
reject, pvals_corrected_dcx_cort, _, _ = multipletests(p_values_dcx_cort, alpha = 0.05, 
                                  method='fdr_bh')
reject, pvals_corrected_ps_cort, _, _ = multipletests(p_values_ps_cort, alpha = 0.05, 
                                  method='fdr_bh')

# multi-test correction: size vs behavior (FDR correction)
reject, pvals_corrected_behav_cort, _, _ = multipletests(p_values_behav, alpha = 0.05, 
                                  method='fdr_bh')

# add the adjusted p values to t-results 
i = 0 
ii = 0 
iii = 0
for key in wilcox_relcort_results:
    if 'ps' in key:
       if any(char.isdigit() for char in key) == True:
            wilcox_relcort_results[key].update({'p_adjusted': pvals_corrected_ps_cort[i]})
            i += 1
       else:
            pass
    elif 'dcx' in key:
       if any(char.isdigit() for char in key) == True:
           wilcox_relcort_results[key].update({'p_adjusted': pvals_corrected_dcx_cort[ii]})
           ii += 1
       else:
           pass
    elif 'Territory' in key:
        wilcox_relcort_results[key].update({'p_adjusted': pvals_corrected_behav_cort[iii]})
        iii += 1
      
# boxplot (cort groups v ihc/behavior)
for key in wilcox_relcort_results:
    # store p value for graphs
    if 'p_adjusted' in wilcox_relcort_results[key]:
        p = wilcox_relcort_results[key]['p_adjusted']
    else:
        p = wilcox_relcort_results[key]['p_values']

    # labels for y axis
    if key == 'ps_avg':
        ylabel = 'Total pS6+ Cell Count\n(DC+Dm3+Vsm)'
    elif key == 'dcx_avg':
        ylabel = 'Total DCX+ Cell Count\n(DC+Dm3+Vsm)'
    elif key == 'dcx_avg_1':
        ylabel = 'DC DCX+ Cell Count'
    elif key == 'dcx_avg_2':
        ylabel = 'Dm3 DCX+ Cell Count'
    elif key == 'dcx_avg_3':
        ylabel = 'Vsm DCX+ Cell Count'
    elif key == 'ps_avg_1':
        ylabel = 'DC pS6+ Cell Count'
    elif key == 'ps_avg_2':
        ylabel = 'Dm3 pS6+ Cell Count'
    elif key == 'ps_avg_3':
        ylabel = 'Vsm pS6+ Cell Count'
    elif key == 'ApproachInitiated':
        ylabel = 'Number of Initiated Approaches'
    elif key == 'TerritoryDuration':
        ylabel = 'Time in Territory(s)'
    elif key == 'Territory_Frequency_Binary':
        ylabel = 'Entry into Territory'
    elif key == 'LateralDisplay':
        ylabel = 'Number of Lateral Displays'
    elif key == 'Displacement':
        ylabel = 'Number of Displacements'
        
    # plot
    fig, ax = plt.subplots()
    ax = sns.boxplot(cichlid_cort, x='Cort_H_L', y=f'{key}', hue='Cort_H_L',
                gap=0.1, palette=('#EAC117','#E56717'), linewidth=1.5, 
                showcaps=False, legend=False, showfliers=False)
    ax = sns.stripplot(x='Cort_H_L', y=f'{key}', data=cichlid_cort, 
                         color="k", alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(which='major', labelsize=14)
    fig.set_size_inches(4, 5)
    ax.set_ylabel(f'{ylabel}', fontsize=18)
    ax.set_xlabel('Cortisol Levels', fontsize=18)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    ymax = max(x for x in cichlid_cort[f'{key}'] if not math.isnan(x))
    # pvalue text
    if math.isnan(p):
        ax.text(xlims[0]+0.83, ymax+0.18, 'n.s.', fontsize=10) 
    else:
        ax.text(xlims[0]+0.83, ymax+0.18, f'p={round(p, 4):.4f}', fontsize=10)
    ax.hlines(ymax+0.1, xlims[0]+0.5, xlims[1]-0.5, colors='k', lw=1.5)
    # sample size text 
    high_n = high_c[f'{key}'].count()
    low_n = low_c[f'{key}'].count()
    ax.text(xlims[0]+0.40, ylims[0], f'N={high_n}')
    ax.text(xlims[1]-0.60, ylims[0], f'N={low_n}')
    plt.savefig(f'graphs/{key}_relativecort_{current_date}.tiff', dpi=300, 
               bbox_inches='tight')
    plt.show() 

# =============================================================================
# Fisher Exact Test
# =============================================================================
# chi-square
f_targets = ['col_y/n', 'Territory_Frequency_Binary']
# size difference
fish_results = {}
for x in f_targets:
    counts = cichlid_cort.value_counts(['Size', f'{x}']).unstack(level=-1)
    fish_results[f'size_v_{x}'] = stats.fisher_exact(counts)

# cort difference 
counts = cichlid_cort.value_counts(['Cort_H_L', f_targets[1]]).unstack(level=-1)
fish_results[f'relcort_v_{f_targets[1]}'] = stats.fisher_exact(counts)

# =============================================================================
# Regression: DCX v pS6
# =============================================================================
# GLM regression
lreg_ihc = {}
pval_ihc = {}
df_predictions = {}
summary_frames = {}
     
reg_result = smf.glm(formula='dcx_avg ~ ps_avg', data=cichlid_final,
                     missing='drop', family=sm.families.Poisson()).fit()
# calc f value 
f_reg = reg_result.f_test('ps_avg=0') 
# generate grid for predicted values for plotting 
x_grid = np.linspace(cichlid_final['ps_avg'].min(), cichlid_final['ps_avg'].max(), 100)
df_predictions['ps_v_dcx'] = pd.DataFrame({'ps_avg': x_grid})
#Calculate predicted values for plotting
prediction_obj = reg_result.get_prediction(df_predictions['ps_v_dcx'])
summary_frames['ps_v_dcx'] = prediction_obj.summary_frame()
# save stats to dictionary
lreg_ihc['ps_v_dcx'] = {'rsquared': reg_result.pseudo_rsquared(), 
                                 'coeff': {'intercept': reg_result.params.iloc[0],
                                           'ps': reg_result.params.iloc[1]},
                                 'pvalues': {'intercept': reg_result.pvalues.iloc[0],
                                             'ps': reg_result.pvalues.iloc[1]},
                                 'fstats': {'fvalue': f_reg.fvalue, 
                                            'df_denom': f_reg.df_denom, 
                                            'df_num': f_reg.df_num}}
    
# p value variable
p = lreg_ihc['ps_v_dcx']['pvalues']['ps']

# plot
#raw data points
ax = sns.scatterplot(data=cichlid_final, x='ps_avg', y='dcx_avg', color='k')
# Plot the GLM regression line
plt.plot(df_predictions['ps_v_dcx'], summary_frames['ps_v_dcx']['mean'], 
         color='orange', lw=2)
ax.set_xlabel('Total pS6+ Cell Count', fontsize=18, wrap=True)
ax.set_ylabel('Total DCX+ Cell Count', fontsize=18)
xlims = plt.xlim() 
ylims = plt.ylim()
ax.text(xlims[1], ylims[1], f'p={round(p, 4):.4f}', fontsize=12, color='k')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)  
plt.savefig(f'graphs/regplot_ps_v_dcx_{current_date}.tiff', dpi=300, 
            bbox_inches='tight')        
plt.show()


# =============================================================================
# DISCRIPTIVE STATS
# ============================================================================
# all data grouped together
descriptives = cichlid_cort.describe()

# grouped by:
# relative size
descr_larger = cichlid_cort.groupby('Size').get_group('Larger').describe()
descr_smaller = cichlid_cort.groupby('Size').get_group('Smaller').describe()
# territory entry
descr_no_entry = cichlid_cort.groupby('Territory_Frequency_Binary').get_group(int(0)).describe()
descr_entry = cichlid_cort.groupby('Territory_Frequency_Binary').get_group(int(1)).describe()
# relative cort level
descr_cort_h= cichlid_cort.groupby('Cort_H_L').get_group('High').describe()
descr_cort_l = cichlid_cort.groupby('Cort_H_L').get_group('Low').describe()

# =============================================================================
# Summary Stats Table 
# =============================================================================
# include principle component analysis done
PC1= {'Independent Variable':'Relative Size', 'Dependent Variable':'PC1', 
      'P Value': '0.041', 'Analysis': 'T-Test'}
PC2= {'Independent Variable':'Relative Size', 'Dependent Variable':'PC2', 
      'P Value': '0.038', 'Analysis': 'T-Test'}

# make a list of dictionaries and names
group_dict = [wilcox_size_results, mann_freq_results, cort_ttest_results, wilcox_relcort_results]

reg_dict = [lreg_diff, lreg_behav_ihc, lreg_ihc, lreg_cort]

group_names = ['wilcox_size_results', 'mann_freq_results', 'cort_ttest_results', 'wilcox_relcort_results']


# loop thru list to populate list of variables (n=98) 
IV = []       # x column 
DV = []       # y column
pv = []       # pvalue column
test = []     # stats test column

# group differences
for i in range(len(group_dict)): 
    df = group_dict[i]
    for key in df.keys():
        parts = key.split('_')
    # populate pv list 
        if 'p_adjusted' in df[key]:
            p = df[key]['p_adjusted']
            pv.append(p)
        elif 'p_values' in df[key]:
            p = df[key]['p_values']
            pv.append(p)
    # append test to test_list 
        if 'wilcox' in group_names[i]:
            test.append('Wilcoxon Signed Rank')
        if 'mann' in group_names[i]:
            test.append('Mann-Whitney U')
        if 'ttest' in group_names[i]:
            test.append('Independent T-Test')
    # append to IV list
        if 'size' in group_names[i]:
            IV.append('Relative Size')
        if 'relcort' in group_names[i]:
            IV.append('Relative Cortisol Levels')
        if 'freq' in group_names[i]:
            IV.append('Entry into Territory')
        if 'cort_ttest' in group_names[i]:
            IV.append('Cortisol Levels')
    # append to DV list 
        if key == 'ps_avg':
            DV.append('Total pS6+ Cell Count')
        elif key == 'dcx_avg':
            DV.append('Total DCX+ Cell Count')
        elif key == 'dcx_avg_1':
            DV.append('DC DCX+ Cell Count')
        elif key == 'dcx_avg_2':
            DV.append('Dm3 DCX+ Cell Count')
        elif key == 'dcx_avg_3':
            DV.append('Vsm DCX+ Cell Count')
        elif key == 'ps_avg_1':
            DV.append('DC pS6+ Cell Count')
        elif key == 'ps_avg_2':
            DV.append('Dm3 pS6+ Cell Count')
        elif key == 'ps_avg_3':
            DV.append('Vsm pS6+ Cell Count')
        elif key == 'ApproachInitiated':
            DV.append('Initiated Approaches')
        elif key == 'TerritoryDuration':
            DV.append('Time in Territory(s)')
        elif key == 'Territory_Frequency_Binary':
            DV.append('Entry into Territory')
        elif key == 'LateralDisplay':
            DV.append('Lateral Displays')
        elif key == 'Displacement':
            DV.append('Displacements') 
        elif key == 'Relative_Size':
            DV.append('Relative Size')
        
#regression     
for i in range(len(reg_dict)): 
    df = reg_dict[i]
    for key in df.keys():
        parts = key.split('_') 
     # populate pv list 
        if 'p_adjusted' in df[key]:
             p = df[key]['p_adjusted']
             pv.append(p)
        elif 'p_values' in df[key]:
             p = df[key]['p_values']
             pv.append(p)
        else: 
             kp = list(df[key]['pvalues'])[1]
             p = df[key]['pvalues'][kp]
             pv.append(p)
    # populate IV and test list
        if 'mass' == parts[0]:
            IV.append('Mass Difference')
            test.append('General Linearized Model')
        if 'ApproachInitiated' == parts[0]:
            IV.append('Initiated Approaches')
            test.append('Poisson GLM')
        if 'Displacement' == parts[0]:
            IV.append('Displacements')
            test.append('Poisson GLM')
        if 'Territory' in parts[0]:
            IV.append('Time in Territory(s)')
            test.append('Poisson GLM')
        if 'LateralDisplay'== parts[0]:
            IV.append('Lateral Displays')
            test.append('Poisson GLM')
        if 'ps' == parts[0]:
            IV.append('Total pS6+ Cell Count')
            test.append('Poisson GLM')
        if 'cort' == parts[0]:
            IV.append('Cortisol Levels')
            if 'TerritoryDuration' == parts[-1]:
                test.append('Poisson GLM')
            elif 'Binary' in key:
                test.append('Ordinary Least Squares')
            else:   
                test.append('General Linearized Model')
    # populate DV   
        if 'dcx' in key:
            if parts[-1] == '1':
                DV.append('DC DCX+ Cell Count')
            elif parts[-1] == '2':
                DV.append('Dm3 DCX+ Cell Count')
            elif parts[-1] == '3':
                DV.append('Vsm DCX+ Cell Count')
            else: 
                DV.append('Total DCX+ Cell Count')
        elif 'ps' in key:
            if parts[0] == 'ps':
                pass
            elif parts[-1] == '1':
                DV.append('DC pS6+ Cell Count')
            elif parts[-1] == '2':
                DV.append('Dm3 pS6+ Cell Count')
            elif parts[-1] == '3':
                DV.append('Vsm pS6+ Cell Count')
            else:
                DV.append('Total pS6+ Cell Count')
        elif 'ApproachInitiated' == parts[-1]:
            DV.append('Initiated Approaches')
        elif 'TerritoryDuration' == parts[-1]:
            DV.append('Time in Territory(s)')
        elif 'Binary' == parts[-1]:
            DV.append('Entry into Territory')
        elif 'LateralDisplay' == parts[-1]:
            DV.append('Lateral Displays')
        elif 'Displacement'== parts[-1]:
            DV.append('Displacements')  

for key in fish_results.keys(): 
    parts = key.split('_') 
    # populate test
    test.append('Fisher Exact Test')
    # populate pv
    p = fish_results[key][1]
    pv.append(p)
    # populate IV 
    if 'size' == parts[0]:
        IV.append('Relative Size')
    elif 'relcort' == parts[0]:
        IV.append('Relative Cortisol Levels')
    # populate DV
    if 'col' == parts[2]:
        DV.append('pS6 DCX Colocalization')
    elif 'Territory' == parts[2]:
        DV.append('Entry into Territory Binary')
    
# combine lists into df
summary_stats = pd.DataFrame({'Independent Variable':IV, 'Dependent Variable':DV, 'p-value':pv, 'Stats Test':test})

# =============================================================================
# EXPORT STATS RESUTLS TO EXCEL
# =============================================================================
# turn dictionary to df 
wilcox_size_df = pd.DataFrame(wilcox_size_results).T
wilcox_cort_df = pd.DataFrame(wilcox_relcort_results).T
mann_freq_df = pd.DataFrame(mann_freq_results).T
cort_df = pd.DataFrame(cort_ttest_results).T
beh_ihc_df = pd.DataFrame(lreg_behav_ihc).T
ihc_df = pd.DataFrame(lreg_ihc).T
diff_df = pd.DataFrame(lreg_diff).T
cort_df = pd.DataFrame(lreg_cort).T
fisher_df = pd.DataFrame(fish_results, index=['fisher_stats', 'pvalue'])

# export df to excel in sheets 
with pd.ExcelWriter(f'cichlid_data_analysis_{current_date}.xlsx') as writer: 
    cichlid_cort.to_excel(writer, sheet_name='Raw Data')
    summary_stats.to_excel(writer, sheet_name='Summary Stats Table')
    wilcox_size_df.to_excel(writer, sheet_name='Rel Mass Group Diff')
    wilcox_cort_df.to_excel(writer, sheet_name='Rel Cort Group Diff')
    mann_freq_df.to_excel(writer, sheet_name='Entry Group Diffs')
    beh_ihc_df.to_excel(writer, sheet_name='Behav v IHC regression')
    ihc_df.to_excel(writer, sheet_name='pS6 v DCX regression')
    diff_df.to_excel(writer, sheet_name='Diff Mass regression')
    cort_df.to_excel(writer, sheet_name='Cortisol Regression')
    descriptives.to_excel(writer, sheet_name='Descr Stats All')
    descr_larger.to_excel(writer, sheet_name='Descr Stats Larger')
    descr_smaller.to_excel(writer, sheet_name='Descr Stats Smaller')
    descr_cort_h.to_excel(writer, sheet_name='Descr Stats Cort High')
    descr_cort_l.to_excel(writer, sheet_name='Descr Stats Cort Low')
    descr_no_entry.to_excel(writer, sheet_name='Descr Stats No Entry')
    descr_entry.to_excel(writer, sheet_name='Descr Stats  Entry')
    fisher_df.to_excel(writer, sheet_name='Fisher Exact Results')
