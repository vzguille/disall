import numpy as np
from matplotlib import font_manager
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def set_font(plt, font_dir = '/scratch/group/arroyave_lab/guillermo.vazquez/FONTS/erewhon-math'):
    plt.rcParams.update({'font.size': 15})
    
    plt.rc('mathtext', fontset="cm")
    
    font_dirs = [font_dir]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
    
    # set font
    plt.rcParams['font.family'] = 'Erewhon Math'


def get_lims(xs, ys, panf=0.05):
    
    h_=np.append(xs, ys)
    mi,ma=np.min(h_),np.max(h_)
    pan=panf*(ma-mi)
    return mi-pan,ma+pan

def parity_plot(gtcl_list,s=5,colors=False,color='blue',unit='',alpha=1, xlabel='', ylabel=''):
    '''parity plot but only for one test'''
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))  # Set the figsize parameter to control the figure size
    
    ii=gtcl_list[0][0]
    jj=gtcl_list[0][1]

    
    # x0 = np.arange(0, 10, 1)
    # y1 = np.zeros(10)
    # y2 = np.ones(10)*0.05

    # y0 = np.arange(0, 10, 1)
    # x1 = np.zeros(10)
    # x2 = np.ones(10)*0.05

    # plt.fill_between(x0, y1, y2, color='blue', alpha=0.8, label='Area between y1 and y2')
    # plt.fill_betweenx([0.0, 1.0], 0.0, 0.05, color='yellow', alpha=0.8, label='Area between x=3 and x=7')

    if colors is not False:
        axs.scatter(ii, jj, s=s, c=colors, alpha=alpha)
    else:    
        axs.scatter(ii, jj, s=s, color=color, alpha=alpha)
    
    
    
        
    
    
    mi,ma=get_lims(ii, jj)
    axs.set_xlim(mi,ma)
    axs.set_ylim(mi,ma)
    axs.grid(True)
    
    axs.annotate(r'$r^2=$'+'{:.4f}'.format(r2_score(ii,jj)),xy=(0.1,0.9),xycoords='axes fraction')
    axs.annotate(r'$RMSE=$'+'{:.4f}{}'.format(np.sqrt(mean_squared_error(ii,jj)),unit),xy=(0.1,0.8),xycoords='axes fraction')
    axs.annotate(r'$MAE=$'+'{:.4f}{}'.format(mean_absolute_error(ii,jj), unit),xy=(0.1,0.7),xycoords='axes fraction')
    if xlabel != '':
        axs.set_xlabel(xlabel)
    if ylabel != '':
        axs.set_ylabel(ylabel)
    
    
    
        
    
    
    mi,ma=get_lims(ii, jj)
    axs.set_xlim(mi,ma)
    axs.set_ylim(mi,ma)
    axs.grid(True)
    
    axs.annotate(r'$r^2=$'+'{:.4f}'.format(r2_score(ii,jj)),xy=(0.1,0.9),xycoords='axes fraction')
    axs.annotate(r'$RMSE=$'+'{:.4f}{}'.format(np.sqrt(mean_squared_error(ii,jj)),unit),xy=(0.1,0.8),xycoords='axes fraction')
    axs.annotate(r'$MAE=$'+'{:.4f}{}'.format(mean_absolute_error(ii,jj), unit),xy=(0.1,0.7),xycoords='axes fraction')
    if xlabel != '':
        axs.set_xlabel(xlabel)
    if ylabel != '':
        axs.set_ylabel(ylabel)

def plot_ce(df_ce):
    df_ce['size'] = df_ce['multicomponent_vector'].apply(lambda x: len(x))
    df_plot = df_ce[df_ce['size']>1]

    fig, axs = plt.subplots(1, 1, figsize=(7, 5))
    axs.grid(True)
    axs.scatter(df_plot['radius'].values, 
                df_plot['eci'].values, 
                color= df_plot['size'].apply(lambda x: ['r','b','g'][x-2]).tolist(),
        s = 8)


    custom_legend = [plt.Line2D([0], [0], color=['r','b','g'][i], lw=2, 
                label='{}'.format(['doublet', 'triplet', 'quadruplet '][i])) 
                    for i in range(3)]
        
    axs.legend(handles=custom_legend, loc='lower right', fontsize = 10)

    axs.set_ylabel(r'ECI ($eV/atom$)', fontsize =13)
    axs.set_xlabel(r'Cluster radius $\AA$', fontsize =13)

    axs.set_xlim(0,)
    axs.set_ylim(df_plot['eci'].values.min() - 0.004,df_plot['eci'].values.max() + 0.001)