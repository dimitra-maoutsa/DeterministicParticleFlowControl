# -*- coding: utf-8 -*-



#@author: maout



import pickle
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

##load non path cost data

import pandas as pd

N = 1000
rhos = [-0.25, -0.5, -0.75, 0.25, 0.5, 0.75]
alldfs = dict()
for rho in rhos:

    file_name = 'evolutionary_gaussian\\Gaussian_evoltionary_Data_Frame_rho_%.2f_N_%d.dat'%(rho,N)
    alldfs[rho]= pd.read_pickle(file_name)
    
    
    
    
    
    
    
    
#%%


plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.facecolor'] = (1,1,1,0)
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['font.size'] = 10*1.2
# plt.rcParams['font.family'] = 'sans-serif'     # not available in Colab
# plt.rcParams['font.sans-serif'] = 'Helvetica'  # not available in Colab
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['xtick.labelsize'] = 10*1.2
plt.rcParams['ytick.labelsize'] = 10*1.2
plt.rcParams['axes.labelsize'] = 12*1.2
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.spines.top'] = True#False
plt.rcParams['axes.spines.right'] = True#False
#plt.rcParams['axes.Axes.tick_params'] = True
#plt.rcParams['axes.solid_capstyle'] = 'round'
plt.rc('axes',edgecolor='#4f4949')
plt.rcParams['figure.frameon'] = False
plt.rcParams['figure.subplot.hspace'] = 0.51
plt.rcParams['figure.subplot.wspace'] = 0.51
plt.rcParams['figure.subplot.left'] = 0.1
plt.rcParams['figure.subplot.right'] = 0.9
plt.rcParams['figure.subplot.top'] = 0.9
plt.rcParams['figure.subplot.bottom'] = 0.1
plt.rcParams['lines.solid_capstyle'] = 'round'
plt.rcParams['lines.solid_joinstyle'] = 'round'
#plt.rcParams['xtick.major.size'] = 20
#plt.rcParams['xtick.major.width'] = 4

plt.rcParams['xtick.major.size'] = 3.
plt.rcParams['ytick.major.size'] = 3.
plt.rcParams['xtick.minor.size'] = 1.5
plt.rcParams['ytick.minor.size'] = 1.5



#%%


cols = plt.cm.BuPu_r

col1 = cols(0.1)
col2 = cols(0.9)

which = ['control', 'err', 'path error' ]

for one in range(0,3):
    plt.figure(figsize=(7, 3.5),dpi=300)
    for i in range(6):
        plt.subplot(2, 3,i+1)
        rho = rhos[i]
        
        sns.pointplot(x="model", y=which[one], hue="constr",
                  data=alldfs[rho], dodge=.132, join=True, palette="BuPu_r",
                  markers="o", scale=.75, ci= 'sd' )
        #plt.title(rho)
        if one==0:
            plt.ylim(10, 10800)
        elif  one==1:
            plt.ylim(0, 0.2)
        elif one==2:
            plt.ylim(0, 0.151)
                
            
        ax = plt.gca()
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[0] = 'cor'
        labels[1] = 'mis'
        ax.set_xticklabels(labels)
        ax.ticklabel_format(style='sci',axis='y',scilimits=(1,4))
        #ax.yaxis.major.formatter._useMathText = True
        if i<3:
            ax.set_xlabel('')
        else:
            ax.set_xlabel(r'model')
        if i in [1,2,4,5]:
            ax.set_ylabel('')
        else:
            if one==0:
                
                ax.set_ylabel('control\n$ \\| u(x,t) \\|_2^2$', multialignment='center')
            elif one==1:
                
                ax.set_ylabel('terminal error\n$ (x^*- X_T)^2$', multialignment='center')
            elif one==2:
               
               ax.set_ylabel('path error\n$\int( X_t - y^*)^2 dt$', multialignment='center')
        if i<3:
            if one == 0:
                ax.text(x=0.5,y=28, s=r'$\rho_{xy}=%.2f$'%rho,
                        color='grey', size=12, 
                        horizontalalignment='center', verticalalignment='center',
                        weight='bold')
            elif one == 1:
                ax.text(x=0.5,y=0.175, s=r'$\rho_{xy}=%.2f$'%rho,
                        color='grey', size=12, 
                        horizontalalignment='center', verticalalignment='center',
                        weight='bold')
            elif one == 2:
                ax.text(x=0.5,y=0.12, s=r'$\rho_{xy}=%.2f$'%rho,
                        color='grey', size=12, 
                        horizontalalignment='center', verticalalignment='center',
                        weight='bold')
            
        else:
            if one == 0:
                ax.text(x=0.5,y=28, s=r'$\rho_{xy}=%.2f$'%rho,
                        color='grey', size=12, 
                        horizontalalignment='center', verticalalignment='center',
                        weight='bold')
            elif one == 1:
                ax.text(x=0.5,y=0.175, s=r'$\rho_{xy}=%.2f$'%rho,
                        color='grey', size=12, 
                        horizontalalignment='center', verticalalignment='center',
                        weight='bold')
            elif one == 2:
                ax.text(x=0.5,y=0.12, s=r'$\rho_{xy}=%.2f$'%rho,
                        color='grey', size=12, 
                        horizontalalignment='center', verticalalignment='center',
                        weight='bold')
                
        handles, labels = ax.get_legend_handles_labels()
        labels[0] = 'end'
        labels[1] = 'path'
        if i != 3:
            
            legend = ax.legend()
            legend.remove()
        else:
            leg3 = ax.legend(handles, labels, title='constraints',
                             handletextpad=0.5, columnspacing=3.2,
                             handlelength=0.8, bbox_to_anchor=[3.85, 1.355],
                             loc=2, ncol=1, frameon=True,fontsize = 'small',
                             shadow=None,framealpha =0,edgecolor ='#0a0a0a')
            for text in leg3.get_texts():
                text.set_color('#4f4949')
            plt.setp(leg3.get_title(),fontsize='small', color='#4f4949')
            
        #plt.yscale('log')
        if one == 0:
            plt.yscale('log')
        
        ax.tick_params(axis="both",direction="in", bottom=True, left=True,top=True, right=True,color='#4f4949')
        
        #ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
        #ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax.tick_params(axis="both", which='minor',direction="in", bottom=True, left=True, top=True, right=True,color='#4f4949')
        
        
        
        ax.spines['bottom'].set_color('#4f4949')
        ax.spines['top'].set_color('#4f4949')
        ax.xaxis.label.set_color('#4f4949')
        ax.tick_params(axis='x', colors='#4f4949')
        
        ax.yaxis.label.set_color('#4f4949')
        ax.tick_params(axis='y', colors='#4f4949')       
        plt.tick_params(axis='y', which='major',color='#4f4949')#, labelsize=16) 
        plt.tick_params(axis='x', which='major',color='#4f4949')
            
                    
    plt.subplots_adjust(left=0.12, bottom=None, right=0.85, top=None,
                    wspace=0.45, hspace=0.3)

    plt.savefig('Gaussian_evolitionary'+which[one]+'.png', bbox_inches='tight',dpi=300 , pad_inches = 0.1)
    plt.savefig('Gaussian_evolitionary'+which[one]+'.pdf', bbox_inches='tight',dpi=300,  pad_inches = 0.1)
