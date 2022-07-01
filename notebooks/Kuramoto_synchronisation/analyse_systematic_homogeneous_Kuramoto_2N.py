# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 19:03:18 2021

@author: maout
"""


import numpy as np


#from matplotlib import pyplot as plt

from scipy.spatial.distance import cdist
import time
import ot
import numba
import math
import random
import sys
import pickle

dim = 2
h = 0.001
t1 = 0
t2 = 1.5
T = t2-t1
timegrid = np.arange(0,T+h/2,h)
N = 2000#0#0#0#2000
#g = 1
k = timegrid.size
M = 80

y2 = np.ones(dim)  
sigmas = np.array([0.1,0.5,1])
rep_bridge = 5   #10 different bridge instances for every setting
reps = 20 ##instanses for stochastic path evaluation of each bridge
# dy0 = np.array([np.pi/4, np.pi/2, np.pi, 3*np.pi/2])    
# y0s = np.zeros((dim,rep_bridge, dy0.size))
random.seed(22)
# for i in range(rep_bridge):
#     y0s[0, i, :] = np.random.uniform( low=0, high=2*np.pi,size=1 ) 
#     y0s[1, i, :] = (y0s[0, i, :] + dy0) %(2* np.pi)



Ks   = np.linspace(0,4,11) 
dy0 = np.array([np.pi/4, np.pi/2])    
y0s = np.zeros((dim,rep_bridge, dy0.size))
Rttcont = np.zeros(( rep_bridge, Ks.size, dy0.size, sigmas.size, timegrid.size,reps   ))*np.nan
Rttnon = np.zeros(( rep_bridge, Ks.size, dy0.size, sigmas.size, timegrid.size,reps   ))*np.nan
used_us = np.zeros((dim, rep_bridge, Ks.size, dy0.size, sigmas.size,timegrid.size,  reps   ))*np.nan
bis = [2,3,4]
for bi in bis:#range(rep_bridge):
    ###loop over bridge iterations
    
    #for ki,K in enumerate(Ks): ##loop over interaction strengths
    for ki,K in enumerate(Ks):
        
        
        for yi in range(dy0.size): ###loop over initial distances
            
            
            
            for gii,g in enumerate(sigmas):
                
                #naming = 'kuramot\\2N_systematic_Kuramoto_homogeneous_repb_%d_ki_%d_yi_%d__gi_%d_N_%d_M_%d'%(bi, ki,yi,gii,N, M)
                naming = 'kurgamot\\New_2N_systematic_Kuramoto_homogeneous_repb_%d_ki_%d_yi_%d__gi_%d_N_%d_M_%d'%(bi, ki,yi,gii,N, M)
                
                try:
                    file = open(naming+'.dat','rb')
                    to_save = pickle.load(file)   
                    # naming2 = 'kuramot\\2N_systematic_Kuramoto_homogeneous_repb_%d_ki_%d_yi_%d__gi_%d_N_%d_M_%d_UNCONTROLLED'%(bi, ki,yi,gii,N, M)
                    # file2 = open(naming2+'.dat','rb')
                    # to_save2 = pickle.load(file2)
                    
                    #Fcont = to_save['Fcont'] 
                    Rttcont[bi,ki,yi, gii,:,:] = to_save['Rttcont']  
                    #Fnon = to_save2['Fnon'] 
                    Rttnon[bi,ki,yi, gii,:,:] = to_save['Rttnon']  
                    Kin = to_save['K']                     
                    y0s[:, bi, yi] = to_save['y0']                
                    used_us[:,bi,ki,yi, gii,:] = to_save['used_us']                     
                except FileNotFoundError:
                    i=0
                    print(naming)




#%%
                    
from scipy.spatial.distance import cdist
import time
import ot
import numba
import math
import random
import sys
import pickle

dim = 2
h = 0.001
t1 = 0
t2 = 1.5
T = t2-t1
timegrid = np.arange(0,T+h/2,h)
N = 2000#0#0#0#2000
#g = 1
k = timegrid.size
M = 80
bi = 2
ki = 3
gii = 2
yi = 1
g=1
naming = 'kurgamot\\New_2N_systematic_Kuramoto_homogeneous_repb_%d_ki_%d_yi_%d__gi_%d_N_%d_M_%d'%(bi, ki,yi,gii,N, M)

file = open(naming+'.dat','rb')
to_save = pickle.load(file) 
Rttconti = to_save['Rttcont'] 
Fcont = to_save['Fcont']
Fnon = to_save['Fnon']  
Rttnoni = to_save['Rttnon']       
used_u = to_save['used_us']  


#%%
import seaborn as sns
from matplotlib import pyplot as plt
repi = 0
pali = sns.diverging_palette(145, 300, s=60, as_cmap=True)
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import matplotlib.gridspec as gridspec
plt.figure(figsize= (8,2.)),
plt.subplot(1,3,1),
#cols = [pali(0), pali(0.99) ]
#cols2 = [pali(0.3),pali(0.7)]
cols = [pali(0.99), pali(0.8) ]
cols2 = [pali(0.),pali(0.2)]
for di in range(dim):
    plt.plot(timegrid[:-1],Fcont[di,:-1,repi],'-',lw=4,c=cols[di],label=r'$\theta_{%d}$'%(di+1))
    plt.plot(timegrid[:-1],Fnon[di,:-1,repi],'.',lw=3,c=cols2[di],alpha=0.7,zorder=0)
for di in range(dim):
    plt.plot(timegrid[:-1],Fnon[di,:-1, repi],'-',lw=3,c=cols2[di],zorder=0,label=r'$\theta_{%d}$'%(di+1)) #this line is only for the legend entry
    
    
plt.xlabel('time')
plt.ylabel(r'phase $\theta$')
ax = plt.gca()
ax.tick_params(axis='both',which='both',direction='out', length=3, width=1,colors='#4f4949',zorder=3)
ax.tick_params(bottom=True, top=False, left=True, right=False)

ax.set_yticks([0,np.pi/2,np.pi, 3*np.pi/2, 2*np.pi])
ax.set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$', r'${3 \pi}/{2}$', r'$2 \pi$'])


# legend = plt.legend(frameon = 1,prop={'size': 15})
# frame = legend.get_frame()
# frame.set_facecolor('white')
# frame.set_edgecolor('white')
plt.xticks([0,0.5,1,1.5])
legend = ax.legend()        
legend.get_frame().set_linewidth(1.8)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('white')
handles, labels = ax.get_legend_handles_labels()

if True:#g==0.5:
    ax.legend(handles, labels, title='controlled   uncontrolled',
              handletextpad=0.5, columnspacing=3.2,handlelength=0.8, bbox_to_anchor=[0.0, 0.785],
              loc=3, ncol=2, frameon=True,fontsize = 'small',shadow=None,framealpha =0,edgecolor ='#0a0a0a')
    
else:
    ax.legend(handles, labels, title='controlled   uncontrolled',
              handletextpad=0.5, columnspacing=3.2,handlelength=0.8,
              loc=3, ncol=2, frameon=True,fontsize = 8,shadow=None,framealpha =0,edgecolor ='#0a0a0a')
    
plt.setp(plt.gca().get_legend().get_title(), fontsize='10') 


#############################################################################

plt.subplot(1,3,2),
plt.plot(timegrid[:-1],Rttnoni[:-1,repi],lw=3,c= cols2[0],zorder=3,label='uncontrolled'),
plt.plot(timegrid[:-1],Rttconti[:-1,repi],lw=3, c=cols[0],zorder=4,label='controlled'),

plt.plot([0,timegrid[-1]], [1/np.sqrt(dim),1/np.sqrt(dim) ],linestyle=(0,(4,3)),lw=3, c='#ee9222',alpha=0.5,label='independent'),
plt.plot([0,timegrid[-1]], [1,1 ],linestyle=(0,(4,3)),c='grey',lw=3,dash_capstyle = "round"),
plt.xlabel('time')
plt.xticks([0,0.5,1,1.5])
plt.ylabel(r'order  param. $R$')   
ax7 = plt.gca()
ax7.tick_params(axis='both',which='both',direction='out', length=3, width=1,colors='#4f4949',zorder=3)
ax7.tick_params(bottom=True, top=False, left=True, right=False)

legend = ax7.legend()        
legend.get_frame().set_linewidth(1.8)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('white')
handles, labels = ax7.get_legend_handles_labels()
handles = [ handles[-2],handles[0] , handles[-1]]
labels= [ labels[-2],labels[0] , labels[-1]]
if True:
    ax7.legend(handles, labels, 
              handletextpad=0.5, columnspacing=0.3,handlelength=0.5, bbox_to_anchor=[-0.4, 0.99],
              loc=3, ncol=3, frameon=True,fontsize = 'small',shadow=None,framealpha =0,edgecolor ='#0a0a0a')    


###########
# axins = zoomed_inset_axes(ax,2, loc=1, bbox_to_anchor=(2400,800))
# mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5")
# axins.set_xlim([0.5,1.2])
# axins.set_ylim([0.95,1.01])

# # # Plot zoom window
# axins.plot(timegrid[:tti],Rtt[:tti],lw=3,c= cols2[0],zorder=3),
# axins.plot(timegrid[:tti],Rttcont[:tti],lw=3, c=cols[0],zorder=4),
# axins.plot([0,timegrid[tti]], [1,1 ],linestyle=(0,(4,3)),c='grey',lw=3,dash_capstyle = "round"),
# axins.tick_params(bottom=False, top=False, left=True, right=False)
#####################
plt.subplot(1,3,3),
#for ti in range(2):
for di in range(dim):
    plt.plot(timegrid[:],h*g**2*used_u[di,:,repi],lw=2.5, c=cols[di],label=r'$u_{%d}$'%(di+1))
plt.xticks([0,0.5,1,1.5])
plt.xlabel('time')
plt.ylabel(r'control $u$')   
ax = plt.gca()
ax.tick_params(axis='both',which='both',direction='out', length=3, width=1,colors='#4f4949',zorder=3)
ax.tick_params(bottom=True, top=False, left=True, right=False)
legend = ax.legend()        
legend.get_frame().set_linewidth(1.8)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('white')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title=None,
          handletextpad=0.5, columnspacing=0,handlelength=1, bbox_to_anchor=[0.50, 0.35],
          loc=1, ncol=1, frameon=True,fontsize = 'small',shadow=None,framealpha =0,edgecolor ='#0a0a0a')


naming2 = 'Kuramoto_2N_trajectories'    
##%%
#plt.savefig( 'naming.png', bbox_inches='tight')
#plt.savefig( 'naming.pdf', bbox_inches='tight')
plt.savefig( naming2+'.png', bbox_inches='tight')
plt.savefig(naming2 +'.pdf', bbox_inches='tight')          

#%%
from matplotlib import pyplot as plt                     

#figure and plotting settings
import seaborn as sns
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
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
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
plt.rcParams['text.usetex'] = True
#%%
bi=2                   
###bi=22 , k=4 and k=5 is good for plotting                    
# R [ bi, ki, yi, gi, :,:]
plt.figure(),
for ki in range(Ks.size):
    plt.subplot(2,6,ki+1)
    plt.title(ki)
    yi=0
    #for yi in range(dy0.size):
        #plt.subplot(1,dy0.size,yi+1)
    plt.plot(timegrid[:-2], np.nanmean(Rttcont[2:4,ki, :, :2,:-2 ], axis=(0,2,-1)).T  ) 
    plt.plot(timegrid[:-2], np.nanmean(Rttnon[2:4,ki, :, :2,:-2 ], axis=(0,2,-1)).T ,'--',alpha=0.4 ) 
    plt.ylim(0.5,1.1)

#%%


# R [ bi, ki, yi, gi, :,:]
bi=3
plt.figure(),
for yi in range(dy0.size):
    plt.subplot(1,dy0.size,yi+1)
    ax = plt.gca()
    # color = next(ax._get_lines.prop_cycler)['color']
    # plt.plot(Ks, np.nanmean(np.nanmean(Rttcont[bi,:, yi, 0 ], axis=-2), axis=(-1)) , c=color)
    # plt.plot(Ks, np.nanmean(np.nanmean(Rttnon[bi,:, yi, 0 ], axis=-2), axis=(-1)) ,'--', c=color)
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(Ks[0::2], np.nanmean(np.nanmean(Rttcont[2:5,0::2, yi, 1,:-2 ], axis=-2), axis=(0,-1)), c=color,marker='.' )  
    plt.plot(Ks[0::2], np.nanmean(np.nanmean(Rttnon[2:5,0::2, yi, 1,:-2 ], axis=-2), axis=(0,-1)) ,'--', c=color,marker='.' )
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(Ks[0::2], np.nanmean(np.nanmean(Rttcont[2:5,0::2, yi, 2 ,:-2], axis=-2), axis=(0,-1)) , c=color,marker='.' )                                       
    plt.plot(Ks[0::2], np.nanmean(np.nanmean(Rttnon[2:5,0::2, yi, 2,:-2 ], axis=-2), axis=(0,-1)) ,'--', c=color,marker='.' )
    plt.ylim(0.5,1)
#%%
plt.figure()
plt.plot(Rttcont[2,3,1,2,:-2]),
plt.plot(Rttnon[2,3,1,2,:-2],'grey')
#%%

plt.figure()
plt.plot(Rttcont[2,5,1,1,:-2]),
plt.plot(Rttnon[2,5,1,1,:-2],'grey',alpha=0.5)

plt.figure()
plt.plot(Rttcont[2,5,1,2,:-2]),
plt.plot(Rttnon[2,5,1,2,:-2],'grey',alpha=0.5)


#%%
bis= 2
bie = 5

plt.figure(),
for yi in range(dy0.size):
    plt.subplot(1,dy0.size,yi+1)
    ax = plt.gca()
    # color = next(ax._get_lines.prop_cycler)['color']
    # plt.plot(Ks, np.nanmean(np.nanmean(Rttcont[bi,:, yi, 0 ], axis=-2), axis=(-1)) , c=color)
    # plt.plot(Ks, np.nanmean(np.nanmean(Rttnon[bi,:, yi, 0 ], axis=-2), axis=(-1)) ,'--', c=color)
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(Ks[1::2], np.nanmean(np.nanmean(Rttcont[bis:bie,1::2, yi, 1,:-2 ], axis=-2), axis=(0,-1)), c=color,marker='.' )  
    plt.plot(Ks[1::2], np.nanmean(np.nanmean(Rttnon[bis:bie,1::2, yi, 1,:-2 ], axis=-2), axis=(0,-1)) ,'--', c=color,marker='.' )
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(Ks[1::2], np.nanmean(np.nanmean(Rttcont[bis:bie,1::2, yi, 2 ,:-2], axis=-2), axis=(0,-1)) , c=color,marker='.' )                                       
    plt.plot(Ks[1::2], np.nanmean(np.nanmean(Rttnon[bis:bie,1::2, yi, 2,:-2 ], axis=-2), axis=(0,-1)) ,'--', c=color,marker='.' )
    plt.ylim(0.5,1)

    
#%%



bi=3
plt.figure(),
for yi in range(0,4):
    plt.subplot(1,4,yi+1)
    ax = plt.gca()
    # color = next(ax._get_lines.prop_cycler)['color']
    # plt.plot(Ks[1::2], np.nanmean(np.nanmean(Rttcont[yi+0:5,1::2, :, 0,:-2 ], axis=-2), axis=(0,2,-1)), c=color,marker='.' )  
    # plt.plot(Ks[1::2], np.nanmean(np.nanmean(Rttnon[yi+0:5,1::2, :, 0,:-2 ], axis=-2), axis=(0,2,-1)) ,'--', c=color,marker='.' )
    # plt.plot(Ks, np.nanmean(np.nanmean(Rttnon[bi,:, yi, 0 ], axis=-2), axis=(-1)) ,'--', c=color)
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(Ks[1:-1:2], np.nanmean(np.nanmean(Rttcont[yi+0:5,1::2, :, 1,:-2 ], axis=-2), axis=(0,2,-1)), c=color,marker='.' )  
    plt.plot(Ks[1:-1:2], np.nanmean(np.nanmean(Rttnon[yi+0:5,1::2, :, 1,:-2 ], axis=-2), axis=(0,2,-1)) ,'--', c=color,marker='.' )
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(Ks[1:-1:2], np.nanmean(np.nanmean(Rttcont[yi+0:5,1::2, :, 2 ,:-2], axis=-2), axis=(0,2,-1)) , c=color,marker='.' )                                       
    plt.plot(Ks[1:-1:2], np.nanmean(np.nanmean(Rttnon[yi+0:5, 1::2, :, 2,:-2 ], axis=-2), axis=(0,2,-1)) ,'--', c=color,marker='.' )
    plt.ylim(0.5,1)    
    
#%%


 
#figure and plotting settings
import seaborn as sns
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
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
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
plt.rcParams['xtick.minor.size'] = 10
#plt.rcParams['xtick.major.width'] = 4
plt.rcParams['text.usetex'] = True


#%%  






pali = sns.diverging_palette(145, 300, s=80, as_cmap=True)
cols = [pali(0.99), pali(0.95), pali(0.85), pali(0.80), pali(0.75), pali(0.70) ]
cols2 = [pali(0.),pali(0.05),pali(0.1),pali(0.15),pali(0.2),pali(0.25)]

frecmap = plt.get_cmap( 'plasma')
# minw = np.abs(np.min(ws))+0.3
# maxw = np.max(ws)
# intervalw = maxw +minw +0.4
# print([ ( (wsi + np.pi/2)/(np.pi)   ) for wsi in ws       ])
# wscols = [ frecmap( (wsi + minw)/(intervalw)   ) for wsi in ws       ]

fig9 = plt.figure(constrained_layout=False, figsize=(6,3))
gs1 = fig9.add_gridspec(nrows=4, ncols=8, wspace=1.5, hspace=1.2)
################################################################
#ax01 = fig9.add_subplot(gs1[0:2,0:2 ]) 

fig9.text(0.7, -0.05, 'time', ha='center',fontsize=16)
fig9.text(0.45, 0.5, r'order  param. $R$', va='center', rotation='vertical',fontsize=14)

fig9.text(0.7, 0.95, r'coupling  $J= 2.0$', ha='center',fontsize=10)
fig9.text(0.95, 0.95, r'noise', ha='center',fontsize=10)
fig9.text(0.95, 0.73, r'$\sigma= 0.5$', ha='center',fontsize=10)
fig9.text(0.95, 0.23, r'$\sigma= 1.0$', ha='center',fontsize=10)
# plt.ylabel(r'order  param. $R$')   
# plt.xlabel('time')
# plt.ylabel(r'order  param. $R$')   

ax02 = fig9.add_subplot(gs1[0:4,0:4 ]) 




#color = next(ax02._get_lines.prop_cycler)['color']
plt.plot(Ks[1:-1:2], np.nanmean(np.nanmean(Rttcont[2:5,1::2, :, 1,:-2 ], axis=-2), axis=(0,2,-1)), c=cols[5],marker='^',label=r'$\sigma=0.5$',zorder=5 ,lw=2,markersize=6,markeredgecolor='#4f4949')  

#color = next(ax02._get_lines.prop_cycler)['color']
plt.plot(Ks[1:-1:2], np.nanmean(np.nanmean(Rttcont[2:5,1::2, :, 2 ,:-2], axis=-2), axis=(0,2,-1)) , c=cols[1],marker='.',label=r'$\sigma=1.0$' ,zorder=5,lw=2,markersize=10,markeredgecolor='#4f4949')    
plt.plot(Ks[1:-1:2], np.nanmean(np.nanmean(Rttnon[2:5,1::2, :, 1,:-2 ], axis=-2), axis=(0,2,-1)) ,'--', c=cols2[5],marker='^',label=r'$\sigma=0.5$',lw=2 ,markersize=6,markeredgecolor='#4f4949')                                   
plt.plot(Ks[1:-1:2], np.nanmean(np.nanmean(Rttnon[2:5, 1::2, :, 2,:-2 ], axis=-2), axis=(0,2,-1)) ,'--', c=cols2[1],marker='.' ,label=r'$\sigma=1.0$',lw=2,markersize=10,markeredgecolor='#4f4949')
plt.ylim(0.5,1)    
#plt.xlim(0,4.01)

legend = ax02.legend()        
legend.get_frame().set_linewidth(1.8)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('white')
handles, labels = ax02.get_legend_handles_labels()
#handles = [ handles[-2],handles[0] , handles[-1]]
#labels= [ labels[-2],labels[0] , labels[-1]]
if True:
    ax02.legend(handles, labels, title=r'controlled $\,$    uncontrolled',
              handletextpad=0.5, columnspacing=0.5,handlelength=1.3, bbox_to_anchor=[-0.0, 0.0],
              loc=3, ncol=2, frameon=True,fontsize = 10,shadow=None,framealpha =0,edgecolor ='#0a0a0a')    

plt.setp(plt.gca().get_legend().get_title(), fontsize='10') 
plt.ylabel(r'order  param. $R$') 
plt.xlabel(r'coupling $J$')

ax1 = fig9.add_subplot(gs1[0:2,4:6 ] ) 

plt.plot(timegrid[:-2],Rttcont[2,5,1,1,:-2],c=cols[5],alpha=1),
plt.plot(timegrid[:-2],np.mean(Rttcont[2,5,1,1,:-2],axis=1), '--',c='#4f4949')

ax1.set_ylim(0.0,1.01)
#plt.yticks([0.5,1.0])
plt.xticks([0.5,1.5])
ax1b = fig9.add_subplot(gs1[0:2,6:8 ] ,  sharey = ax1) 

plt.plot(timegrid[:-2],Rttnon[2,5,1,1,:-2],c=cols2[5],alpha=0.5)
ax1b.set_ylim(0.0,1.01)
plt.plot(timegrid[:-2],np.mean(Rttnon[2,5,1,1,:-2],axis=1), '--',c='#4f4949')
#plt.ylabel(r'order  param. $R$')   
plt.xticks([0.5,1.5])
#plt.yticks([0.5,1.0])
ax2 = fig9.add_subplot(gs1[ 2:4,4:6], sharex = ax1)
ax2.set_ylim(0.0,1.01)
plt.plot(timegrid[:-2],Rttcont[2,5,1,2,:-2],c=cols[1],alpha=1),
plt.plot(timegrid[:-2],np.mean(Rttcont[2,5,1,2,:-2],axis=1), '--',c='#4f4949')
#plt.xlabel('time')
#plt.ylabel(r'order  param. $R$')   
plt.xticks([0.5,1.5])
#plt.yticks([0.5,1.0])
ax2b = fig9.add_subplot(gs1[ 2:4,6:8], sharex = ax1b, sharey = ax2) 
plt.plot(timegrid[:-2],Rttnon[2,5,1,2,:-2],c=cols2[1],alpha=0.5)
plt.plot(timegrid[:-2],np.mean(Rttnon[2,5,1,2,:-2],axis=1), '--',c='#4f4949')
ax2b.set_ylim(0.0,1.01)
plt.xticks([0.5,1.5])



#plt.savefig("Kuramoto_2.png", bbox_inches='tight')
#plt.savefig("Kuramoto_2.pdf", bbox_inches='tight')
#%% onset of synchronisation

# R [ bi, ki, yi, gi, :,:]

onset_cont = np.zeros((3,Ks.size,2,3,20)) *np.nan
onset_uncont = np.zeros((3,Ks.size,2,3,20)) *np.nan

cont_synched_durations_m = np.zeros((3,Ks.size,2,3,20)) *np.nan
cont_synched_durations_std  = np.zeros((3,Ks.size,2,3,20)) *np.nan
cont_synched_durations_max = np.zeros((3,Ks.size,2,3,20)) *np.nan
cont_synched_durations_sum  = np.zeros((3,Ks.size,2,3,20)) *np.nan

uncont_synched_durations_m = np.zeros((3,Ks.size,2,3,20)) *np.nan
uncont_synched_durations_std  = np.zeros((3,Ks.size,2,3,20)) *np.nan
uncont_synched_durations_max = np.zeros((3,Ks.size,2,3,20)) *np.nan
uncont_synched_durations_sum  = np.zeros((3,Ks.size,2,3,20)) *np.nan

counter_cont = np.zeros((3,Ks.size,2,3))
counter_uncont =  np.zeros((3,Ks.size,2,3))

threshold = 0.99
min_duri = 20
for bi in range(3):
    bii = bi+2 ###bi should start from 2 for the data matrices since I don't have simulation results for bi:0-1
    for ki in range(Ks.size):
        for yi in range(2):#
            for gi in range(1,3):#3                
                for repi in range(20):                    
                    positions = np.where( Rttcont[bii, ki, yi, gi,1:-2, repi] >=threshold)[0]  ###this one detects if there is at all any synchronisation
                    synch_or_not = Rttcont[bii, ki, yi, gi,1:-2, repi] >=threshold ##this is boolean indicating the synchronised positions
                    
                    if positions.size==0: 
                        onset_cont[bi,ki,yi,gi,repi] = np.nan
                    else:
                        diffpos = np.diff(synch_or_not.astype(int)   ) #difference between consecutive steps
                        
                        diffpos2 = np.zeros(diffpos.size+1) # extend by one step at the beginning to detect 
                        diffpos2[1:] = diffpos
                        starts = np.argwhere(diffpos2 == 1)
                        stops = np.argwhere(diffpos2 == -1)                        
                        if starts.size == 0:
                            onset_cont[bi,ki,yi,gi,repi] = np.nan
                        elif stops.size == 0:
                            onset_cont[bi,ki,yi,gi,repi] = timegrid[ starts[0] ]
                            counter_cont[bi,ki,yi,gi] += 1
                            dur = timegrid[ -1 ] - timegrid[ starts[0] ]
                            cont_synched_durations_m[bi,ki,yi,gi,repi]  = dur
                            cont_synched_durations_std[bi,ki,yi,gi,repi]  = 0
                            cont_synched_durations_max[bi,ki,yi,gi,repi]  = dur
                            cont_synched_durations_sum[bi,ki,yi,gi,repi]  = np.sum(  synch_or_not )/ (timegrid.size-1 - starts[0] )
                        else:
                            durations = stops[:,0] - starts[:stops.size, 0]                            
                            synchned_more_than_50 = np.where( durations>=min_duri )[0]
                            if synchned_more_than_50.size == 0:
                                onset_cont[bi,ki,yi,gi,repi] = np.nan
                            else:                                
                                onset_cont[bi,ki,yi,gi,repi] = timegrid[ starts[synchned_more_than_50[0]] ]
                                counter_cont[bi,ki,yi,gi] += 1                                
                                cont_synched_durations_m[bi,ki,yi,gi,repi]  = np.mean(  durations  )
                                cont_synched_durations_std[bi,ki,yi,gi,repi]  = np.std(  durations )
                                cont_synched_durations_max[bi,ki,yi,gi,repi]  = np.max(  durations )
                                cont_synched_durations_sum[bi,ki,yi,gi,repi]  = np.sum(  synch_or_not[ starts[synchned_more_than_50[0]][0] :   ] )/(timegrid.size-1 -starts[synchned_more_than_50[0]] ) ##timesteps in synchronied

                       
                        
                    positions = np.where( Rttnon[bii, ki, yi, gi, 1:-2,repi] >=threshold)[0]
                    synch_or_not = Rttnon[bii, ki, yi, gi,1:-2, repi] >=threshold                     
                    if positions.size ==0: 
                        onset_uncont[bi,ki,yi,gi,repi] = np.nan
                    else:
                        diffpos = np.diff(synch_or_not.astype(int)   ) #difference between consecutive steps
                        
                        diffpos2 = np.zeros(diffpos.size+1) # extend by one step at the beginning to detect 
                        diffpos2[1:] = diffpos
                        starts = np.argwhere(diffpos2 == 1)
                        stops = np.argwhere(diffpos2 == -1)
                        #start_stop =[starts, stops - starts]
                        if starts.size == 0:
                            onset_uncont[bi,ki,yi,gi,repi] = np.nan
                        elif stops.size == 0:
                            onset_uncont[bi,ki,yi,gi,repi] = timegrid[ starts[0] ]
                            counter_uncont[bi,ki,yi,gi] += 1
                            dur = timegrid[ -1 ] - timegrid[ starts[0] ]
                            uncont_synched_durations_m[bi,ki,yi,gi,repi]  = dur
                            uncont_synched_durations_std[bi,ki,yi,gi,repi]  = 0
                            uncont_synched_durations_max[bi,ki,yi,gi,repi]  = dur
                            uncont_synched_durations_sum[bi,ki,yi,gi,repi]  = np.sum(  synch_or_not )/ (timegrid.size-1 - starts[0] )
                        else:
                            durations = stops[:,0] - starts[:stops.size, 0]                            
                            synchned_more_than_50 = np.where( durations>=min_duri )[0]
                            
                            if synchned_more_than_50.size == 0:
                                onset_uncont[bi,ki,yi,gi,repi] = np.nan
                            else:                                
                                onset_uncont[bi,ki,yi,gi,repi] = timegrid[starts[synchned_more_than_50[0]] ]
                                counter_uncont[bi,ki,yi,gi] += 1
                                uncont_synched_durations_m[bi,ki,yi,gi,repi]  = np.mean(  durations )
                                uncont_synched_durations_std[bi,ki,yi,gi,repi]  = np.std(  durations )
                                uncont_synched_durations_max[bi,ki,yi,gi,repi]  = np.max(  durations  )
                                uncont_synched_durations_sum[bi,ki,yi,gi,repi]  = np.sum(  synch_or_not[ starts[synchned_more_than_50[0]][0] :   ] ) /(timegrid.size-1 -starts[synchned_more_than_50[0]] )

#%%
count_perc = np.sum(counter_uncont[0:3,1::2,:1, :],axis=0)/60         #counts the percentage of synchronised uncontrolled network for the annotations               
                        
plt.figure(figsize=(6.5,2)) 
yi = 0
gi = 1
plt.subplot(1,2,1)
ax1 = plt.gca()
#er1 = plt.errorbar(Ks[1:-1:2], np.nanmean(onset_cont[0:3,1::2,yi, 1], axis=(0,-1)), yerr=np.nanstd(onset_cont[0:3,1::2,yi, 1], axis=(0,-1)), marker="o", linestyle="-",uplims=True,lolims=True,)
#er2 = plt.errorbar(Ks[1:-1:2], np.nanmean(onset_uncont[0:3,1::2,yi, 1], axis=(0,-1)), yerr=np.nanstd(onset_uncont[0:3,1::2,yi, 1], axis=(0,-1)), marker="o", linestyle="-",uplims=True,lolims=True)
plt.plot(Ks[1:-1:2], np.nanmean(onset_cont[0:3,1::2,:1, 1], axis=(0,2,-1)) , c=cols[5],marker='^',label=r'$\sigma=0.5$',zorder=5 ,lw=2.8,markersize=6,markeredgecolor='#4f4949') 
plt.plot(Ks[1:-1:2], np.nanmean(onset_uncont[0:3,1::2,:1, 1], axis=(0,2,-1)),'--', c=cols2[5],marker='^',label=r'$\sigma=0.5$' ,zorder=5,lw=2.8,markersize=6,markeredgecolor='#4f4949')  

plt.text(0.29, 0.49,"%.2f"%count_perc[0, 0 ,1], fontsize =8 , color = "grey")

plt.text(0.97, 0.57,"%.2f"%count_perc[1, 0 ,1], fontsize =8 , color = "grey")

plt.text(2.05, 0.44,"%.2f"%count_perc[2, 0 ,1], fontsize =8 , color = "grey")


plt.text(0.29, 0.28,"%.2f"%count_perc[0, 0 ,2], fontsize =8 , color = "grey")

plt.text(1.1, 0.37,"%.2f"%count_perc[1, 0 ,2], fontsize =8 , color = "grey")

plt.text(1.89, 0.28,"%.2f"%count_perc[2, 0 ,2], fontsize =8 , color = "grey")


plt.plot(Ks[1:-1:2], np.nanmean(onset_cont[0:3,1::2,:1, 2], axis=(0,2,-1)) , c=cols[1],marker='.',label=r'$\sigma=1.0$',lw=2.8 ,markersize=12,markeredgecolor='#4f4949') 
plt.plot(Ks[1:-1:2], np.nanmean(onset_uncont[0:3,1::2,:1, 2], axis=(0,2,-1)),'--', c=cols2[0],marker='.' ,label=r'$\sigma=1.0$',lw=2.8,markersize=12,markeredgecolor='#4f4949')   

#plt.yscale('log')                    
plt.ylabel(r'onset of synchrony $t^{syn}$') 
plt.xlabel(r'coupling $J$')                

ax6 = plt.gca()

ax6.tick_params(axis='both',which='both',direction='in', length=3, width=1,colors='#4f4949',zorder=3)
ax6.tick_params(bottom=True, top=True, left=True, right=True)
ax6.spines['top'].set_visible(True)
ax6.spines['right'].set_visible(True)
ax6.minorticks_on()
ax6.tick_params(axis='both',which='major',direction='in', length=3.5, width=1,colors='#4f4949',zorder=3)
ax6.tick_params(axis='both',which='minor',direction='in', length=2.5, width=0.5,colors='#4f4949',zorder=3)
ax6.tick_params(bottom=True, top=True, left=True, right=True)
ax6.tick_params(axis='both', which='minor', bottom=True, top=True, left=True, right=True)


plt.subplot(1,2,2)

plt.plot(Ks[1:-1:2], np.nanmean(cont_synched_durations_sum[0:3,1::2,:1, 1], axis=(0,2,-1)) , c=cols[5],marker='^',label=r'$\sigma=0.5$',zorder=2 ,lw=2.8,markersize=6,markeredgecolor='#4f4949') 
plt.plot(Ks[1:-1:2], np.nanmean(uncont_synched_durations_sum[0:3,1::2,:1, 1], axis=(0,2,-1)),'--', c=cols2[5],marker='^',label=r'$\sigma=0.5$' ,zorder=2,lw=2.8,markersize=6,markeredgecolor='#4f4949')  


plt.plot(Ks[1:-1:2], np.nanmean(cont_synched_durations_sum[0:3,1::2,:1, 2], axis=(0,2,-1)) , c=cols[1],marker='.',label=r'$\sigma=1.0$',lw=2.8 ,markersize=12,markeredgecolor='#4f4949') 
plt.plot(Ks[1:-1:2], np.nanmean(uncont_synched_durations_sum[0:3,1::2,:1, 2], axis=(0,2,-1)),'--', c=cols2[0],marker='.' ,label=r'$\sigma=1.0$',lw=2.8,markersize=12,markeredgecolor='#4f4949')   

plt.ylabel('$\%$ time synchronised\nafter $t^{syn}$', multialignment='center',labelpad=-0.1) # "Mat\nTTp\n123"
plt.xlabel(r'coupling $J$') 
plt.yticks([0, 0.5,1])
ax7 = plt.gca()
ax7.tick_params(axis='both',which='both',direction='in', length=3, width=1,colors='#4f4949',zorder=3)
ax7.tick_params(bottom=True, top=True, left=True, right=True)
ax7.spines['top'].set_visible(True)
ax7.spines['right'].set_visible(True)
ax7.minorticks_on()
ax7.tick_params(axis='both',which='major',direction='in', length=3.5, width=1,colors='#4f4949',zorder=3)
ax7.tick_params(axis='both',which='minor',direction='in', length=2.5, width=0.5,colors='#4f4949',zorder=3)
ax7.tick_params(bottom=True, top=True, left=True, right=True)
ax7.tick_params(axis='both', which='minor', bottom=True, top=True, left=True, right=True)

legend = ax7.legend()        
legend.get_frame().set_linewidth(1.8)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('white')
handles, labels = ax7.get_legend_handles_labels()
handles = [ handles[0],handles[2] , handles[1], handles[3]]
labels= [ labels[0],labels[2] , labels[1], labels[3]]
if True:
    ax7.legend(handles, labels, title=r'$\,$controlled $\,\,$ uncontrolled',
              handletextpad=0.5, columnspacing=1,handlelength=0.45, bbox_to_anchor=[-0.65, 0.99],
              loc=3, ncol=2, frameon=True,fontsize = 'small',shadow=None,framealpha =0,edgecolor ='#0a0a0a')    
plt.setp(plt.gca().get_legend().get_title(), fontsize='10') 
plt.subplots_adjust(wspace=0.325)#, hspace=0)
plt.savefig("Kuramoto_2_onset.png", bbox_inches='tight')
plt.savefig("Kuramoto_2_onset.pdf", bbox_inches='tight')
"""
plt.subplot(1,2,2)#,sharey=ax1)

# plt.plot(Ks[1:-1:2], np.nanmean(onset_cont[:,1::2,yi, 1], axis=(0,-1)) , c=cols[5],marker='^',label=r'$\sigma=0.5$',zorder=5 ,lw=2.8,markersize=6,markeredgecolor='#4f4949') 
# plt.plot(Ks[1:-1:2], np.nanmean(onset_uncont[:,1::2,yi, 1], axis=(0,-1)),'--', c=cols2[5],marker='.',label=r'$\sigma=1.0$' ,zorder=5,lw=2.8,markersize=10,markeredgecolor='#4f4949')  
plt.plot(Ks[1:-1:2], np.nanmean(onset_cont[0:3,1::2,0, 2], axis=(0,-1)) , c=cols[1],marker='^',label=r'$\sigma=0.5$',lw=2.8 ,markersize=6,markeredgecolor='#4f4949') 
plt.plot(Ks[1:-1:2], np.nanmean(onset_uncont[0:3,1::2,0, 2], axis=(0,-1)),'--', c=cols2[2],marker='.' ,label=r'$\sigma=1.0$',lw=2.8,markersize=12,markeredgecolor='#4f4949')   


plt.plot(Ks[1:-1:2], np.nanmean(onset_cont[0:3,1::2,1, 2], axis=(0,-1)) , c=cols[1],marker='d',label=r'$\sigma=0.5$',lw=2.8 ,markersize=6,markeredgecolor='#4f4949') 
plt.plot(Ks[1:-1:2], np.nanmean(onset_uncont[0:3,1::2,1, 2], axis=(0,-1)),'--', c=cols2[2],marker='p' ,label=r'$\sigma=1.0$',lw=2.8,markersize=7,markeredgecolor='#4f4949')   

#plt.yscale('log')                    
plt.ylabel(r'onset of synchrony $t^{syn}$') 
plt.xlabel(r'coupling $J$')        

"""

#%%
ki=1
yi=0
gi=1
bi=1
for repi in range(1):
    positions = np.where( Rttnon[bii, ki, yi, gi, 1:-2,repi] >=threshold)[0]
    print(positions)
    if positions.size ==0: 
        onset_uncont[bi,ki,yi,gi,repi] = np.nan
    else:
        diffpos = np.diff(positions)
        positions_consecutive = np.where(diffpos < 5)[0]
        if positions_consecutive.size ==0:
            onset_uncont[bi,ki,yi,gi,repi] =  np.nan
        else:
            onset_uncont[bi,ki,yi,gi,repi] =  timegrid[positions[positions_consecutive[0]] ]
    print(onset_uncont[bi,ki,yi,gi,repi])
    # print()
    # print()
                        
                
