# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:21:27 2022

@author: maout
"""

#run from folder codes\Bridges
#figs in folder codes\Bridges\fig2

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import pickle

                

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
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.spines.top'] =True
plt.rcParams['axes.spines.right'] = True
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


savefolder = 'fig2/'

b= 1
def f(x,t=0):    
    x0 =    - 4 *b*x[0]**3+x[0] *(4*b*x[1]-2) + 2
    x1 = - 2*b*(x[1] -  x[0]**2)
    return np.array([x0,x1])


def U_path(x):
    return -(np.power( x[1]- y2[1] ,2 ))


#%%
    

##load non path cost data
filename =  'evolutionary_rocken//Evolutionary_2d_rocken_WITHOUT_PATHCOST_N_80__M_400_t2_0.700_non_clipped_controls.dat'  
file = open(filename,'rb')
to_sav = pickle.load(file)

bridg2dB = to_sav['B'] 
bridg2dZ = to_sav['Z']  
timegrid = to_sav['timegrid'] 
y1 = to_sav['y1'] 
y2  = to_sav['y2'] 
t1 = to_sav['t1'] 
t2 = to_sav['t2'] 
g = to_sav['g']
M = to_sav['M'] 
N = to_sav['N'] 
Fnon = to_sav['Fnon'] 
Fcont = to_sav['Fcont'] 
used_u = to_sav['used_u'] 
b = to_sav['b'] 
#to_sav['fx'] = ' - 4 *b*x[0]**3+x[0] *(4*b*x[1]-2) + 2'
#to_sav['fy'] = '- 2*b*(x[1] -  x[0]**2)'

#%% load PIce data
nis = [500,1000,1500, 2000]
Fcont_pice = dict()
used_upice = dict()
alph =  dict()
iteri = dict()
for ni in nis:
    
    filename =  'evolutionary_rocken//Evolutionary_2d_rocken__PICE_WITHOUT_PATHCOST_N_%d__t2_0.700_order_3.dat'%(ni)
    file = open(filename,'rb')
    to_sav = pickle.load(file)
    
    Fcont_pice[ni] = to_sav['Fcont_pice'] 
    used_upice[ni] = to_sav['used_upice'] 
    alph[ni] = to_sav['alph'] 
    iteri[ni] = to_sav['iteri'] 
    
    
    
import matplotlib.gridspec as gridspec
from matplotlib import cm

#%% Load systematic 




Ns = [500, 600, 700, 800, 900, 1000]    
Ms = [50,  100]
FcontB = dict()
used_uB = dict()
for N in Ns:
    FcontB[N] = dict()
    used_uB[N] = dict()
    for M in Ms:  
        FcontB[N][M] = dict()
        used_uB[N][M] = dict()
        for repi in range(5):
            ##load non path cost data
            filename =  'evolutionary_rocken//dpf_non//Evolutionary_2d_rocken_WITHOUT_PATHCOST_M_%d__N_%d_t2_0.700_non_clipped_controls_repi_%d.dat'%(M,N,repi)  
            file = open(filename,'rb')
            to_sav = pickle.load(file)
            timegrid = to_sav['timegrid'] 
            y1 = to_sav['y1'] 
            y2  = to_sav['y2'] 
            t1 = to_sav['t1'] 
            t2 = to_sav['t2'] 
            g = to_sav['g']        
            FnonB = to_sav['Fnon'] 
            FcontB[N][M][repi] = to_sav['Fcont'] 
            used_uB[N][M][repi] = to_sav['used_u'] 
            b = to_sav['b'] 
    
##%% load PIce data
nis = [500, 600, 700, 800, 900, 1000] 
Fcont_piceB = dict()
used_upiceB = dict()
alphB =  dict()
iteriB = dict()
for ni in nis:
    Fcont_piceB[ni] = dict()
    used_upiceB[ni] = dict()
    alphB[ni] =  dict()
    iteriB[ni] = dict()
    for repi in range(5):
    
        filename =  'evolutionary_rocken//pice_non//Evolutionary_2d_rocken__PICE_WITHOUT_PATHCOST_N_%d__t2_0.700_order_3_reps_%d.dat'%(ni,repi)
        file = open(filename,'rb')
        to_sav = pickle.load(file)
        
        Fcont_piceB[ni][repi] = to_sav['Fcont_pice'] 
        used_upiceB[ni][repi] = to_sav['used_upice'] 
        alphB[ni][repi] = to_sav['alph'] 
        iteriB[ni][repi] = to_sav['iteri'] 
    
##%%
import pandas as pd    
#del dpf_df
repi = 1000
dt = 0.001
T = t2
controls = np.zeros(( len(Ms), len(Ns), repi ))
#M=80
indx = 0
midt = 350 ##midpoint in time axis
dpf_df = pd.DataFrame( columns = ['control', 'M','N','rep_u', 'control_1sthalf', 'control_2ndhalf','end_error' ])
for ni,N in enumerate(Ns):
    for rep in range(5):
        for mi,M in enumerate(Ms):        
            data = {'control': np.power(np.sum(np.sqrt(np.nansum(np.power(used_uB[N][M][rep][:,:,:-1],2), axis=0)),axis=-1),2) /(T/dt) ,\
                    'M': [M]*repi, 'N': [N]*repi, 'rep_u':[rep]*repi, \
                    'control_1sthalf':np.power(np.sum(np.sqrt(np.nansum(np.power(used_uB[N][M][rep][:,:,:midt],2), axis=0)),axis=-1),2) /(T/dt), \
                    'control_2ndhalf':np.power(np.sum(np.sqrt(np.nansum(np.power(used_uB[N][M][rep][:,:,midt:-1],2), axis=0)),axis=-1),2) /(T/dt), \
                    'end_error': np.sqrt(np.sum( (FcontB[N][M][rep][:,:,-1]-np.atleast_2d(y2).T)**2 ,axis=0  ))  }       
            
            df1 =  pd.DataFrame(data, columns = ['control', 'M','N','rep_u', 'control_1sthalf', 'control_2ndhalf','end_error'])
            dpf_df = dpf_df.append(df1, ignore_index=True)
        data = {'control': np.power(np.sum(np.sqrt(np.nansum(np.power(used_upiceB[N][rep][:,:,:-1],2), axis=0)),axis=-1),2) /(T/dt) , \
                'M': ['pice']*repi, 'N': [N]*repi, 'rep_u':[rep]*repi, \
                'control_1sthalf':np.power(np.sum(np.sqrt(np.nansum(np.power(used_upiceB[N][rep][:,:,:midt],2), axis=0)),axis=-1),2) /(T/dt), \
                'control_2ndhalf':np.power(np.sum(np.sqrt(np.nansum(np.power(used_upiceB[N][rep][:,:,midt:-1],2), axis=0)),axis=-1),2) /(T/dt), \
                'end_error': np.sqrt(np.sum( (Fcont_piceB[N][rep][:,:,-1]-np.atleast_2d(y2).T)**2 ,axis=0  ))  } 
            
        df1 =  pd.DataFrame(data, columns = ['control', 'M','N','rep_u', 'control_1sthalf', 'control_2ndhalf','end_error'])
        #df1 = df1[~df1.isin([np.nan, np.inf, -np.inf]).any(1)]
        
        df1 = df1.loc[ df1['control'] < 10000000   ]
        
        dpf_df = dpf_df.append(df1, ignore_index=True)    
        del df1

#%% 1st version


w = 2.
Y, X = np.mgrid[0:2:200j, -w:w:200j]
XY = np.concatenate( (X.reshape(1,-1), Y.reshape(1,-1)), axis=0 )

F_XY = f(XY)
U = F_XY[0]
V = F_XY[1]
speed =  np.log((1-U)**2 + b*(V - U**2)**2)#np.sqrt(U**2 + V**2)
#fixed_points = np.array([[1.996, 0.004, 1], [0.004, 1.996,  1]])
#speed2 = np.clip(speed, 0,20)
fig = plt.figure(figsize=(9.5, 2.5))
#gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1])
gs = gridspec.GridSpec(nrows=1, ncols=3, height_ratios=[1])

#  Varying density along a streamline
ax0 = fig.add_subplot(gs[0, 0])
plt.contourf(X, Y, speed.reshape(X.shape)  , cmap=plt.cm.BuPu_r,levels=150)

#plt.colorbar()
#


ax0.streamplot(X, Y, U.reshape(X.shape), V.reshape(Y.shape), 
               density=[0.75, 0.75], color='#4f4949')
#ax0.plot(-1.2,2,'o')


#ax0.plot( (bridg2d.Z[0]), (bridg2d.Z[1]),alpha=0.31, c='grey',lw=1) '#C70039'
ax0.plot( np.mean(bridg2dB[0],axis=0), np.mean(bridg2dB[1],axis=0),'.',
         alpha=0.85, c='#900C3F',lw=2.5,label= r'mean $q_{t}(x)$')
ax0.plot( np.mean(bridg2dB[0],axis=0)+ np.std(bridg2dB[0],axis=0), 
         np.mean(bridg2dB[1],axis=0) + np.std(bridg2dB[1],axis=0),'--',
         alpha=0.85,c='#900C3F',lw=2.5,label= r'std $q_{t}(x)$')
ax0.plot( np.mean(bridg2dB[0],axis=0)- np.std(bridg2dB[0],axis=0),
         np.mean(bridg2dB[1],axis=0) - np.std(bridg2dB[1],axis=0),'--',
         alpha=0.85, c='#900C3F',lw=2.5)
# ax0.plot( (bridg2d.B[0,::10]).T, (bridg2d.B[1,::10]).T,alpha=0.51, c='grey',lw=0.5)
# ax0.plot( (bridg2d.B[0,::10,200]).T, (bridg2d.B[1,::10,200]).T,'.',alpha=0.91, c=cm.viridis(0.685),lw=0.5,markersize=2)
# ax0.plot( (bridg2d.B[0,::10,200:218]).T, (bridg2d.B[1,::10,200:218]).T,'-',alpha=0.71, c=cm.viridis(0.685),lw=1)


# ax0.plot( np.mean(bridg2d.Z[0],axis=0), np.mean(bridg2d.Z[1],axis=0),'.',alpha=0.85, c=cm.viridis(0.97),lw=2.5)
# ax0.plot( np.mean(bridg2d.Z[0],axis=0)+ np.std(bridg2d.Z[0],axis=0), np.mean(bridg2d.Z[1],axis=0) + np.std(bridg2d.Z[1],axis=0),'--',alpha=0.85,c=cm.viridis(0.97),lw=2.5)
# ax0.plot( np.mean(bridg2d.Z[0],axis=0)- np.std(bridg2d.Z[0],axis=0), np.mean(bridg2d.Z[1],axis=0) - np.std(bridg2d.Z[1],axis=0),'--',alpha=0.85, c=cm.viridis(0.97),lw=2.5)
ax0.plot( (bridg2dZ[0,::2]).T, (bridg2dZ[1,::2]).T,alpha=0.51, c='grey',
         lw=0.55,label= r'$\rho_t(x)$ ')
ax0.plot( (bridg2dZ[0,::4,418]).T, (bridg2dZ[1,::4,418]).T,'.',alpha=0.91,
         c=cm.viridis(0.685),lw=0.5,markersize=2, label='particle')
ax0.plot( (bridg2dZ[0,::4,400:418]).T, (bridg2dZ[1,::4,400:418]).T,'-',
         alpha=0.71, c=cm.viridis(0.685),lw=1)


# ax0.plot(fixed_points[0,0], fixed_points[1,0], 'go')
# ax0.plot(fixed_points[0,2], fixed_points[1,2], 'X', c='silver')
# ax0.plot(fixed_points[0,1], fixed_points[1,1], 'o', c='#4f4949')
ax0.plot(y1[0], y1[1], 'go', label = r'$\mathbf{x}_0$')
ax0.plot(y2[0], y2[1], 'X', c='silver', label=r'$\mathbf{x}^*$')


handles, labels = ax0.get_legend_handles_labels()
to_skip = (bridg2dZ[0,::2]).shape[0]-1
handles2 = [ handles[ii] for ii in [2,0,1  ] ]
labels2 = [labels[ii] for ii in [ 2,0,1  ] ]
handles2[0].set_linewidth(1.5)

handles2b = [ handles[ii] for ii in [-2,-1,-3  ] ]
labels2b = [labels[ii] for ii in [ -2,-1,-3  ] ]
leg1 = ax0.legend(handles2b, labels2b, title=None,
          handletextpad=0.5, columnspacing=3.2,handlelength=0.8, 
          bbox_to_anchor=[-0.65, 0.30955],labelspacing = 0.3,
          loc=2, ncol=1, frameon=True,fontsize = 'medium',shadow=None,
          framealpha =0,edgecolor ='#0a0a0a')
for text in leg1.get_texts():
    text.set_color('#4f4949')

leg2 = ax0.legend(handles2, labels2, title=None,
          handletextpad=0.5, columnspacing=3.2,handlelength=0.8, 
          bbox_to_anchor=[-0.65, 1.0],labelspacing = 0.3,
          loc=2, ncol=1, frameon=True,fontsize = 'medium',shadow=None,
          framealpha =0,edgecolor ='#0a0a0a')
for text in leg2.get_texts():
    text.set_color('#4f4949')

ax0.add_artist(leg1)
#ax0.set_title('Varying Density')

ax0.spines['right'].set_visible(True)
ax0.spines['top'].set_visible(True)
#ax0.spines['left'].set_visible(False)
#ax0.spines['bottom'].set_visible(False)
ax0.set_xticks([-2,0, 2])
ax0.set_yticks([0, 1,2])
ax0.set_xlim(-2,2)
ax0.set_ylim(0,2)
ax0.set_xlabel(r'x', size=16)
ax0.set_ylabel(r'y', size=16)
#plt.axis('square')
#ax0.xaxis.set_major_formatter(plt.LinearLocator(-2,2))
#plt.minorticks_on()
# ax0.yaxis.get_ticklocs(minor=True)  
# # Initialize minor ticks
# ax0.minorticks_on()
# ax0.tick_params(axis="x", direction="in", length=25, width=5, color="red")
# #ax0.yaxis.set_minor_locator(MultipleLocator(1))
##########################################################################################################

# Varying color along a streamline
ax1 = fig.add_subplot(gs[0, 1])
plt.contourf(X, Y, speed.reshape(X.shape)  , cmap=plt.cm.BuPu_r,levels=150)
#strm = ax1.streamplot(X, Y, U.reshape(X.shape), V.reshape(X.shape), color=U.reshape(X.shape), linewidth=2, cmap='autumn')
strm = ax1.streamplot(X, Y, U.reshape(X.shape), V.reshape(X.shape), 
                      color='#4f4949', density=[0.75, 0.75])
ax1.plot(y1[0], y1[1], 'go',zorder=5)
ax1.plot(y2[0], y2[1], 'X', c='silver',zorder=5)

###############ax1.plot(Fnon[0,0],Fnon[1,0], c='grey', alpha=0.85,lw=1.8, label='uncontrolled')
indx = 2
for ti in range(0,timegrid.size):
    if ti==timegrid.size-200:
        ax1.plot(Fcont[0,indx,ti:ti+2],Fcont[1,indx,ti:ti+2],
                 c=cm.viridis(ti/(timegrid.size)),zorder=5,lw=1.8,
                 label='controlled')
        ax1.plot(Fnon[0,0,ti:ti+2],Fnon[1,0,ti:ti+2], 
                 c=cm.copper(0.2+ti/(0.2+timegrid.size)), alpha=0.85,lw=1.8, 
                 label='uncontrolled')
    else:
        ax1.plot(Fcont[0,indx,ti:ti+2],Fcont[1,indx,ti:ti+2], 
                 c=cm.viridis(ti/(timegrid.size)),zorder=5,lw=1.8)
        ax1.plot(Fnon[0,0,ti:ti+2],Fnon[1,0,ti:ti+2], 
                 c=cm.copper(0.2+ti/(0.2+timegrid.size)), alpha=0.85,lw=1.8)

#fig.colorbar(strm.lines)

#ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
ax1.spines['right'].set_visible(True)
ax1.spines['top'].set_visible(True)
#ax1.set_title('Varying Color')
ax1.set_xticks([-2,0, 2])
ax1.set_yticks([0, 1,2])
ax1.set_xlim(-2,2)
ax1.set_ylim(0,2)
ax1.set_xlabel(r'x', size=16)
ax1.set_ylabel(r'y', size=16)

handles, labels = ax1.get_legend_handles_labels()

leg3 = ax1.legend(handles, labels, title=None,
          handletextpad=0.5, columnspacing=3.2,handlelength=0.8,
          bbox_to_anchor=[0.15, 1.3],
          loc=2, ncol=1, frameon=True,fontsize = 'medium',shadow=None,
          framealpha =0,edgecolor ='#0a0a0a')
for text in leg3.get_texts():
    text.set_color('#4f4949')
##########################################################################################################################
#  Varying density along a streamline
ax2 = fig.add_subplot(gs[0, 2])
plt.contourf(X, Y, speed.reshape(X.shape)  , cmap=plt.cm.BuPu_r,levels=150)
#plt.colorbar()
ax2.streamplot(X, Y, U.reshape(X.shape), V.reshape(Y.shape), density=[1, 1],
               color='#4f4949')



ax2.plot( np.mean(bridg2dB[0],axis=0), np.mean(bridg2dB[1],axis=0),'.',
         alpha=0.85, c='#900C3F',lw=2.5,label= r'mean $q_{t}(x)$') 

ax2.plot( np.mean(bridg2dB[0],axis=0)+ np.std(bridg2dB[0],axis=0),
         np.mean(bridg2dB[1],axis=0) + np.std(bridg2dB[1],axis=0),
         linestyle=(0, (1, 3)),alpha=0.85,c='#900C3F',lw=2.5,
         label= r'std $q_{t}(x)$')

ax2.plot( np.mean(bridg2dB[0],axis=0)- np.std(bridg2dB[0],axis=0), 
         np.mean(bridg2dB[1],axis=0) - np.std(bridg2dB[1],axis=0),
         linestyle=(0, (1, 3)),alpha=0.85, c='#900C3F',lw=2.5)


#ax0.plot( (bridg2d.Z[0]), (bridg2d.Z[1]),alpha=0.31, c='grey',lw=1)
ax2.plot( np.mean(Fcont[0],axis=0), np.mean(Fcont[1],axis=0),'-',
         alpha=0.85, c=cm.viridis(0.97),lw=2.5,label= r'mean $\hat{q}_{t}(x)$')
ax2.plot( np.mean(Fcont[0],axis=0)+ np.std(Fcont[0],axis=0), 
         np.mean(Fcont[1],axis=0) + np.std(Fcont[1],axis=0),'--',
         alpha=0.85,c=cm.viridis(0.97),lw=2.5,label= r'std $\hat{q}_{t}(x)$')
ax2.plot( np.mean(Fcont[0],axis=0)- np.std(Fcont[0],axis=0), 
         np.mean(Fcont[1],axis=0) - np.std(Fcont[1],axis=0),'--',
         alpha=0.85, c=cm.viridis(0.97),lw=2.5)
# ax2.plot( (bridg2d.B[0,::10]).T, (bridg2d.B[1,::10]).T,alpha=0.51, c='grey',lw=0.5)
# ax2.plot( (bridg2d.B[0,::10,200]).T, (bridg2d.B[1,::10,200]).T,'.',alpha=0.91, c=cm.viridis(0.685),lw=0.5,markersize=2)
# ax2.plot( (bridg2d.B[0,::10,200:218]).T, (bridg2d.B[1,::10,200:218]).T,'-',alpha=0.71, c=cm.viridis(0.685),lw=1)

# ax0.plot(fixed_points[0,0], fixed_points[1,0], 'go')
# ax0.plot(fixed_points[0,2], fixed_points[1,2], 'X', c='silver')
# ax0.plot(fixed_points[0,1], fixed_points[1,1], 'o', c='#4f4949')
ax2.plot(y1[0], y1[1], 'go')
ax2.plot(y2[0], y2[1], 'X', c='silver')
#ax0.set_title('Varying Density')
ax2.spines['right'].set_visible(True)
ax2.spines['top'].set_visible(True)
#ax1.set_title('Varying Color')
ax2.set_xticks([-2,0, 2])
ax2.set_yticks([0, 1,2])
ax2.set_xlim(-2,2)
ax2.set_ylim(0,2)
ax2.set_xlabel(r'x', size=16)
ax2.set_ylabel(r'y', size=16)
#ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')

handles, labels = ax2.get_legend_handles_labels()

leg4 = ax2.legend(handles, labels, title=None,
          handletextpad=0.5, columnspacing=1.2,handlelength=0.8, 
          bbox_to_anchor=[-0.1, 1.3],
          loc=2, ncol=2, frameon=True,fontsize = 'medium',shadow=None,
          framealpha =0,edgecolor ='#0a0a0a')
for text in leg4.get_texts():
    text.set_color('#4f4949')

ax0.axis('on')
ax1.axis('on')
ax2.axis('on')
#plt.tight_layout()
plt.subplots_adjust(hspace=0.3)#, 
                    #hspace=0.4)
plt.show()
plt.savefig(savefolder+'UPDATEDnon_path_2d_pheno.png', 
            bbox_inches='tight',dpi=300 , pad_inches = -0, transparent='False',
            facecolor='white')
plt.savefig(savefolder+'UPDATEDnon_path_2d_pheno.pdf', bbox_inches='tight',
            dpi=300,  pad_inches = -0, transparent='False', facecolor='white')



#%% forward particles plot sinle for appendix


fig = plt.figure(figsize=(4, 4))
#gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1])
gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1])

#  Varying density along a streamline
ax0 = fig.add_subplot(gs[0, 0])
plt.contourf(X, Y, speed.reshape(X.shape)  , cmap=plt.cm.BuPu_r,levels=150)

#plt.colorbar()
#


ax0.streamplot(X, Y, U.reshape(X.shape), V.reshape(Y.shape), 
               density=[0.75, 0.75], color='#4f4949')
#ax0.plot(-1.2,2,'o')


#ax0.plot( (bridg2d.Z[0]), (bridg2d.Z[1]),alpha=0.31, c='grey',lw=1) '#C70039'
ax0.plot( np.mean(bridg2dB[0],axis=0), np.mean(bridg2dB[1],axis=0),'.',
         alpha=0.85, c='#900C3F',lw=2.5,label= r'$\mu_{q_{t}(x)}$') 
#cm.viridis(0.97)
ax0.plot( np.mean(bridg2dB[0],axis=0)+ np.std(bridg2dB[0],axis=0), 
         np.mean(bridg2dB[1],axis=0) + np.std(bridg2dB[1],axis=0),'--',
         alpha=0.85,c='#900C3F',lw=2.5,label= r'$\sigma_{q_{t}(x)}$')
ax0.plot( np.mean(bridg2dB[0],axis=0)- np.std(bridg2dB[0],axis=0), 
         np.mean(bridg2dB[1],axis=0) - np.std(bridg2dB[1],axis=0),'--',
         alpha=0.85, c='#900C3F',lw=2.5)
# ax0.plot( (bridg2d.B[0,::10]).T, (bridg2d.B[1,::10]).T,alpha=0.51, c='grey',lw=0.5)
# ax0.plot( (bridg2d.B[0,::10,200]).T, (bridg2d.B[1,::10,200]).T,'.',alpha=0.91, c=cm.viridis(0.685),lw=0.5,markersize=2)
# ax0.plot( (bridg2d.B[0,::10,200:218]).T, (bridg2d.B[1,::10,200:218]).T,'-',alpha=0.71, c=cm.viridis(0.685),lw=1)


# ax0.plot( np.mean(bridg2d.Z[0],axis=0), np.mean(bridg2d.Z[1],axis=0),'.',alpha=0.85, c=cm.viridis(0.97),lw=2.5)
# ax0.plot( np.mean(bridg2d.Z[0],axis=0)+ np.std(bridg2d.Z[0],axis=0), np.mean(bridg2d.Z[1],axis=0) + np.std(bridg2d.Z[1],axis=0),'--',alpha=0.85,c=cm.viridis(0.97),lw=2.5)
# ax0.plot( np.mean(bridg2d.Z[0],axis=0)- np.std(bridg2d.Z[0],axis=0), np.mean(bridg2d.Z[1],axis=0) - np.std(bridg2d.Z[1],axis=0),'--',alpha=0.85, c=cm.viridis(0.97),lw=2.5)
ax0.plot( (bridg2dZ[0,::2]).T, (bridg2dZ[1,::2]).T,alpha=0.51,
         c='grey',lw=0.55,label= r'$\rho_t(x)$ ')
ax0.plot( (bridg2dZ[0,::4,418]).T, (bridg2dZ[1,::4,418]).T,'.',
         alpha=0.91, c=cm.viridis(0.685),lw=0.5,markersize=2, label='particle')
ax0.plot( (bridg2dZ[0,::4,400:418]).T, (bridg2dZ[1,::4,400:418]).T,'-',
         alpha=0.71, c=cm.viridis(0.685),lw=1)


# ax0.plot(fixed_points[0,0], fixed_points[1,0], 'go')
# ax0.plot(fixed_points[0,2], fixed_points[1,2], 'X', c='silver')
# ax0.plot(fixed_points[0,1], fixed_points[1,1], 'o', c='#4f4949')
ax0.plot(y1[0], y1[1], 'go', label = r'$\mathbf{x}_0$')
ax0.plot(y2[0], y2[1], 'X', c='silver', label=r'$\mathbf{x}^*$')


handles, labels = ax0.get_legend_handles_labels()
to_skip = (bridg2dZ[0,::2]).shape[0]-1
handles2 = [ handles[ii] for ii in [2,0,1  ] ]
labels2 = [labels[ii] for ii in [ 2,0,1  ] ]
handles2[0].set_linewidth(1.5)

handles2b = [ handles[ii] for ii in [-2,-1,-3  ] ]
labels2b = [labels[ii] for ii in [ -2,-1,-3  ] ]
leg1 = ax0.legend(handles2b, labels2b, title=None,
          handletextpad=0.5, columnspacing=3.2,handlelength=0.8,
          bbox_to_anchor=[-0.5, 0.30955],labelspacing = 0.3,
          loc=2, ncol=1, frameon=True,fontsize = 'large',shadow=None,
          framealpha =0,edgecolor ='#0a0a0a')
for text in leg1.get_texts():
    text.set_color('#4f4949')

leg2 = ax0.legend(handles2, labels2, title=None,
          handletextpad=0.5, columnspacing=3.2,handlelength=0.8,
          bbox_to_anchor=[-0.5, 1.0],labelspacing = 0.3,
          loc=2, ncol=1, frameon=True,fontsize = 'large',shadow=None,
          framealpha =0,edgecolor ='#0a0a0a')
for text in leg2.get_texts():
    text.set_color('#4f4949')

ax0.add_artist(leg1)
#ax0.set_title('Varying Density')

ax0.spines['right'].set_visible(True)
ax0.spines['top'].set_visible(True)
#ax0.spines['left'].set_visible(False)
#ax0.spines['bottom'].set_visible(False)
ax0.set_xticks([-2,0, 2])
ax0.set_yticks([0, 1,2])
ax0.set_xlim(-2,2)
ax0.set_ylim(0,2)
ax0.set_xlabel(r'x')
ax0.set_ylabel(r'y')

plt.savefig(savefolder+'UPDATEDApp.non_path_2d_pheno_forward_particles_backward.png',
            bbox_inches='tight',dpi=300 , pad_inches = 1, transparent='False', 
            facecolor='white')
plt.savefig(savefolder+'UPDATEDApp.non_path_2d_pheno_forward_particles_backward.pdf',
            bbox_inches='tight',dpi=300,  pad_inches = 1, transparent='False',  
            facecolor='white')

#%%

"""
This is the final plot code!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

"""

import matplotlib.gridspec as gridspec
from matplotlib import cm
w = 2.
Y, X = np.mgrid[0:2:200j, -w:w:200j]
XY = np.concatenate( (X.reshape(1,-1), Y.reshape(1,-1)), axis=0 )

F_XY = f(XY)
U = F_XY[0]
V = F_XY[1]
speed =  np.log((1-U)**2 + b*(V - U**2)**2)#np.sqrt(U**2 + V**2)
#fixed_points = np.array([[1.996, 0.004, 1], [0.004, 1.996,  1]])
#speed2 = np.clip(speed, 0,20)
fig = plt.figure(figsize=(12, 3.5),dpi=300)
#gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1])
gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[1,1])



# Varying color along a streamline
gs01 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0:2, 0:2],
                                        wspace=0.3)
ax1 = fig.add_subplot(gs01[0:2, 0])
plt.contourf(X, Y, speed.reshape(X.shape)  , cmap=plt.cm.BuPu_r,levels=150,
             alpha=0.8)
#strm = ax1.streamplot(X, Y, U.reshape(X.shape), V.reshape(X.shape),
strm = ax1.streamplot(X, Y, U.reshape(X.shape), V.reshape(X.shape),
                      color='#4f4949', density=[0.75, 0.75])
ax1.plot(y1[0], y1[1], 'go',zorder=100, markersize=10)
ax1.plot(y2[0], y2[1], 'X', c='silver',zorder=100, markersize=10)

###############ax1.plot(Fnon[0,0],Fnon[1,0], c='grey', alpha=0.85,lw=1.8, 
indx = 422
for ti in range(0,timegrid.size):
    if ti==timegrid.size-200:
        ax1.plot(Fcont[0,indx,ti:ti+2],Fcont[1,indx,ti:ti+2], 
                 c=cm.viridis(ti/(timegrid.size)),zorder=5,lw=1.8, 
                 label=r'$X^{\mathrm{DPF}}_t$')
        ax1.plot(Fnon[0,0,ti:ti+2],Fnon[1,0,ti:ti+2], 
                 c=cm.copper(0.2+ti/(0.2+timegrid.size)), alpha=0.85,lw=1.8, 
                 label='$X_t^{\mathrm{unc}}$')
    elif ti==timegrid.size-1:
        ax1.plot(Fcont[0,indx,ti:ti+2],Fcont[1,indx,ti:ti+2],
                 c=cm.viridis(ti/(timegrid.size)),zorder=1001,lw=1.8)
        ax1.plot(Fnon[0,0,ti:ti+2],Fnon[1,0,ti:ti+2],
                 c=cm.copper(0.2+ti/(0.2+timegrid.size)), alpha=0.85,lw=1.8)

    else:
        ax1.plot(Fcont[0,indx,ti:ti+2],Fcont[1,indx,ti:ti+2],
                 c=cm.viridis(ti/(timegrid.size)),zorder=5,lw=1.8)
        ax1.plot(Fnon[0,0,ti:ti+2],Fnon[1,0,ti:ti+2], 
                 c=cm.copper(0.2+ti/(0.2+timegrid.size)), alpha=0.85,lw=1.8)

#fig.colorbar(strm.lines)

#ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
ax1.spines['right'].set_visible(True)
ax1.spines['top'].set_visible(True)
#ax1.set_title('Varying Color')
ax1.set_xticks([-2,0, 2])
ax1.set_yticks([0, 1,2])
ax1.set_xlim(-2,2)
ax1.set_ylim(0,2)
ax1.set_xlabel(r'x',fontsize=26)
ax1.set_ylabel(r'y',fontsize=26)

handles, labels = ax1.get_legend_handles_labels()

leg3 = ax1.legend(handles, labels, title=None,
          handletextpad=0.5, columnspacing=1.2,handlelength=0.8,
          bbox_to_anchor=[0.215, 0.95],
          loc=1, ncol=2, frameon=True,fontsize = 'large',shadow=None,
          framealpha=0, edgecolor ='#0a0a0a',bbox_transform=fig.transFigure)
for text in leg3.get_texts():
    text.set_color('#4f4949')
###############################################################################


########################################################################
    
gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs01[0:2,1],
                                        hspace=0.1)
ax3 = fig.add_subplot(gs00[0])
#ax3 = fig.add_subplot(gs[0, 1])


ds_palette = plt.get_cmap('plasma')#plt.get_cmap('cool')#sns.diverging_palette(145, 300, s=60, as_cmap=True)##sns.diverging_palette(145, 300, s=60, as_cmap=True)
my_mag =  ds_palette(0.33)
my_green = ds_palette(0.5)
orag = cm.copper(0.9)
orag2 = cm.copper(0.95)
#'#4f4949'
ni_pice = 500#2000

mnstdlc = np.mean(Fcont[0],axis=0)-np.std(Fcont[0],axis=0)
mnstdupc = np.mean(Fcont[0],axis=0)+np.std(Fcont[0],axis=0)
#plt.fill_between(timegrid,y1=mnstdlc,y2 = mnstdupc,color=my_mag, alpha=0.4,linewidth=0.15, zorder= 1 )

plt.plot(timegrid,np.mean(Fcont[0],axis=0),linestyle=(0, (3, 3)), c=my_mag, lw=4, zorder=2,label=r'$\mu_{\hat{q}_t^{\mathrm{ DPF}}}$')
plt.plot(timegrid[::1],np.mean(Fcont_pice[ni_pice][0, :,::1],axis=0),'-', c='grey',lw=4.,zorder=0,label=r'$\mu_{\hat{q}_t^{\mathrm{ pice}}}$')
#plt.plot(timegrid,np.mean(Fcont_pice[ni_pice][0],axis=0),linestyle=(0, (1, 4)), c='#4f4949',lw=4,dash_capstyle='butt')
plt.plot(timegrid,np.mean(Fnon[0],axis=0),'-',c=orag,lw=2, zorder=0,alpha=0.95,label=r'$\mu_{\hat{q}_t^{\mathrm{ unc}}}$')
plt.plot(timegrid[::20],np.mean(Fcont[0,:,::20],axis=0)-np.std(Fcont[0,:,::20],axis=0),'.', c=my_mag,lw=4,label=r'$\sigma_{\hat{q}_t^{\mathrm{ DPF}}}$' )
plt.plot(timegrid[::20],np.mean(Fcont[0,:,::20],axis=0)+np.std(Fcont[0,:,::20],axis=0),'.', c=my_mag,lw=4)
plt.plot(timegrid,np.mean(Fcont_pice[ni_pice][0],axis=0)-np.std(Fcont_pice[ni_pice][0],axis=0),linestyle=(0, (6, 5)),c='grey', lw=4,dash_capstyle='butt',zorder=0,label=r'$\sigma_{\hat{q}_t^{\mathrm{pice}}}$')
plt.plot(timegrid,np.mean(Fcont_pice[ni_pice][0],axis=0)+np.std(Fcont_pice[ni_pice][0],axis=0),linestyle=(0, (6, 5)),c='grey',lw=4,dash_capstyle='butt',zorder=0)
# plt.plot(timegrid[:],np.mean(Fnon[0],axis=0)-np.std(Fnon[0],axis=0),'--',c='grey',zorder=0,alpha=0.95,lw=1)
# plt.plot(timegrid[:],np.mean(Fnon[0],axis=0)+np.std(Fnon[0],axis=0),'--',c='grey',zorder=0,alpha=0.95,lw=1)
mnstdl = np.mean(Fnon[0],axis=0)-np.std(Fnon[0],axis=0)
mnstdup = np.mean(Fnon[0],axis=0)+np.std(Fnon[0],axis=0)
plt.fill_between(timegrid,y1=mnstdl,y2 = mnstdup,color=orag, alpha=0.2,linewidth=0, zorder= 0)
plt.plot(timegrid[0],y1[0],'go',zorder=9)

plt.plot(timegrid[-1],y2[0],'X', c='silver',zorder=10)
#ax2.plot(y1[0], y1[1], 'go')
#ax2.plot(y2[0], y2[1], 'X', c='silver')

handles, labels = ax3.get_legend_handles_labels()

handlesb = [handles[0], handles[3], handles[1], handles[4], handles[2]]
labelsb = [labels[0], labels[3], labels[1], labels[4], labels[2]]

leg5 = ax3.legend(handlesb, labelsb, title=None,
          handletextpad=0.5, columnspacing=0.6,handlelength=0.8,
          bbox_to_anchor=[0.485, 1.05],
          loc=1, ncol=3, frameon=True,fontsize = 'large',shadow=None,
          framealpha =0,edgecolor ='#0a0a0a',bbox_transform=fig.transFigure)
for text in leg5.get_texts():
    text.set_color('#4f4949')



import matplotlib.ticker as tck

ax4 = fig.add_subplot(gs00[1], sharex=ax3)
mnstdlc = np.mean(Fcont[1],axis=0)-np.std(Fcont[1],axis=0)
mnstdupc = np.mean(Fcont[1],axis=0)+np.std(Fcont[1],axis=0)
#plt.fill_between(timegrid,y1=mnstdlc,y2 = mnstdupc,color=my_mag, alpha=0.24,linewidth=0.15, zorder= 1 )

plt.plot(timegrid,np.mean(Fcont[1],axis=0),linestyle=(0, (3, 3)), 
         c=my_mag, lw=4, zorder=2)
plt.plot(timegrid[::1],np.mean(Fcont_pice[ni_pice][1, :,::1],axis=0),'-',
         c='grey',lw=4.,zorder=0)
#plt.plot(timegrid,np.mean(Fcont_pice[ni_pice][0],axis=0),linestyle=(0, (1, 4)), c='#4f4949',lw=4,dash_capstyle='butt')
plt.plot(timegrid,np.mean(Fnon[1],axis=0),'-',c=orag,lw=2, zorder=0,alpha=0.95)
plt.plot(timegrid[::20],
         np.mean(Fcont[1,:,::20],axis=0)-np.std(Fcont[1,:,::20],axis=0),'.',
         c=my_mag,lw=4,label=r'$\sigma_{\hat{q}_t^{\mathrm{ DPF}}}$')
plt.plot(timegrid[::20],
         np.mean(Fcont[1,:,::20],axis=0)+np.std(Fcont[1,:,::20],axis=0),'.',
         c=my_mag,lw=4)
plt.plot(timegrid,
         np.mean(Fcont_pice[ni_pice][1],axis=0)-np.std(Fcont_pice[ni_pice][1]
                                                       ,axis=0),
         linestyle=(0, (6, 5)),c='grey', lw=4,dash_capstyle='butt',
         zorder=0,label=r'$\sigma_{\hat{q}_t^{\mathrm{pice}}}$')
plt.plot(timegrid,
         np.mean(Fcont_pice[ni_pice][1],axis=0)+np.std(Fcont_pice[ni_pice][1],
                                                       axis=0),
         linestyle=(0, (6, 5)),c='grey',lw=4,dash_capstyle='butt',zorder=0)
# plt.plot(timegrid[:],np.mean(Fnon[0],axis=0)-np.std(Fnon[0],axis=0),'--',c='grey',zorder=0,alpha=0.95,lw=1)
# plt.plot(timegrid[:],np.mean(Fnon[0],axis=0)+np.std(Fnon[0],axis=0),'--',c='grey',zorder=0,alpha=0.95,lw=1)
mnstdl = np.mean(Fnon[1],axis=0)-np.std(Fnon[1],axis=0)
mnstdup = np.mean(Fnon[1],axis=0)+np.std(Fnon[1],axis=0)
plt.fill_between(timegrid,y1=mnstdl,y2 = mnstdup,color=orag, alpha=0.2,
                 linewidth=0, zorder= 0)
  
plt.plot(timegrid[0],y1[1],'go')
plt.plot(timegrid[-1],y2[1],'X', c='silver')
ax3.set_ylabel(r'x', labelpad=-5,fontsize=26)
ax4.set_ylabel(r'y', labelpad=5,fontsize=26)
ax4.set_xlabel(r'time',fontsize=26)
ax3.tick_params(axis="both",direction="in", top=True, right=True)

ax4.set_xticks([0,0.35,0.7])

ax3.yaxis.set_minor_locator(tck.AutoMinorLocator())
ax3.xaxis.set_minor_locator(tck.AutoMinorLocator())
ax3.tick_params(axis="both", which='minor',direction="in", top=True, 
                right=True)
ax4.tick_params(axis="both",direction="in", top=True, right=True)
ax4.yaxis.set_minor_locator(tck.AutoMinorLocator())
ax4.xaxis.set_minor_locator(tck.AutoMinorLocator())
ax4.tick_params(axis="both", which='minor',direction="in", top=True, 
                right=True)
plt.setp(ax3.get_xticklabels(), visible=False)

"""
handles, labels = ax4.get_legend_handles_labels()

leg6 = ax4.legend(handles, labels, title=None,
          handletextpad=0.5, columnspacing=3.2,handlelength=0.8, bbox_to_anchor=[-0.52, 1.2],
          loc=2, ncol=1, frameon=True,fontsize = 'large',shadow=None,framealpha =0,edgecolor ='#0a0a0a')
for text in leg6.get_texts():
    text.set_color('#4f4949')
"""

##########################################################################
ax5 = fig.add_subplot(gs[0, 2])

purple_pal = cm.magma_r 
grey_pal = plt.get_cmap('Greys_r')
ds_palette = plt.get_cmap('plasma')
my_mag =  ds_palette(0.33)
my_mag2 =  ds_palette(0.45)

my_orange = ds_palette(0.75)
my_orange2 = ds_palette(0.85)

my_palette = sns.color_palette( [my_mag , '#666666' ])
my_mag3 = (my_mag2[0], my_mag2[1],my_mag2[2],0.15)
my_orange3 = (my_orange2[0], my_orange2[1],my_orange2[2],0.15)
my_palette2 = sns.color_palette( [my_mag3 , '#939393' ]) 

dt = 0.001
T = t2
controls = np.log(np.array([np.power(np.sum(np.sqrt(np.nansum(np.power(used_u[:,:,:-1],2),
                                                              axis=0)),
                                            axis=-1),2) /(T/dt), 
                            np.power(np.sum(np.sqrt(np.nansum(np.power(used_upice[ni_pice][:,:,:-1],2),
                                                              axis=0)),
                                            axis=-1),2) /(T/dt)]) )
import pandas as pd
df = pd.DataFrame({'DPF': controls[0, :], 'pice': controls[1, :]})
#df.drop([962], inplace=True)

sns.violinplot( data=df, palette=my_palette2, alpha=0.65,saturation=0.81)
#color="0.8")
sns.stripplot( data=df,jitter=0.15,alpha=0.5, palette=my_palette,
              edgecolor='#363636',linewidth=0.25,size=3)

# distance across the "X" or "Y" stipplot column to span, in this case 30%
mean_width = 0.5

for tick, text in zip(ax5.get_xticks(), ax5.get_xticklabels()):
    sample_name = text.get_text()  # "X" or "Y"
    # calculate the median value for all replicates of either X or Y
    mean_val = df[sample_name].mean()
    # plot horizontal lines across the column, centered on the tick
    ax5.plot([tick-mean_width/2, tick+mean_width/2], [mean_val, mean_val],
            lw=4, color='silver',solid_capstyle='round',zorder=3)
sns.despine(trim=True,  top=True, right=True, bottom=True,ax=ax5)
#sns.despine(offset=10, trim=True)
plt.ylabel(r'control$ $'+'\n'+r' \fontsize{14pt}{3em}\selectfont{}{$\log \, \| u(x,t) \|_2^2$}',
           multialignment='center',fontsize=24, linespacing=0.75)
ax5.spines['bottom'].set_color('#363636')
ax5.spines['top'].set_color('#363636')
ax5.xaxis.label.set_color('#363636')
ax5.tick_params(axis='x', colors='#363636')
ax5.yaxis.label.set_color('#363636')
ax5.tick_params(axis='y', colors='#363636')    
plt.tick_params(axis='y', which='major')#, labelsize=16) 
plt.tick_params(axis='x', which='major')#, labelsize=16) 
ax5.xaxis.set_tick_params(width=0)


ax5.annotate(u"$\sim 250$ iterations ",
                xy=(0.625, 0.85), xycoords='axes fraction',
                xytext=(0.5, 1.05), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="-", color="1",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="bar, fraction=0.3, angle=0.",
                                ), fontsize=13, color='grey'
                )

ax5.annotate(u"$1$ iteration ",
                xy=(0.15, 0.85), xycoords='axes fraction',
                xytext=(0.025, 1.05), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="-", color="1",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="bar, fraction=0.3, angle=0.",
                                ), fontsize=13, color='grey'
                )

#ax5.set_ylim(None,7700)
#ax5.set_yscale('log')
#######

ax6 = fig.add_subplot(gs[1, 2])
end_cost1 = 1*np.sqrt(np.sum( (Fcont[:,:,-1]-np.atleast_2d(y2).T)**2 ,axis=0))

end_cost3 = np.sqrt(np.sum((Fcont_pice[ni_pice][:,:,-2]-np.atleast_2d(y2).T)**2,
                              axis=0))


df_end = pd.DataFrame({r'DPF': end_cost1, r'pice': end_cost3})

sns.violinplot(  data=df_end, palette=my_palette2, alpha=0.65,saturation=0.81)
#color="0.8")

sns.stripplot( data=df_end,jitter=0.15,alpha=0.5, palette=my_palette,
              edgecolor='#363636',linewidth=0.25,size=3)
mean_width = 0.5

for tick, text in zip(ax6.get_xticks(), ax6.get_xticklabels()):
    sample_name = text.get_text()  # "X" or "Y"
    # calculate the median value for all replicates of either X or Y
    mean_val = df_end[sample_name].mean()
    # plot horizontal lines across the column, centered on the tick
    ax6.plot([tick-mean_width/2, tick+mean_width/2], [mean_val, mean_val],
            lw=4, color='silver',solid_capstyle='round',zorder=3)
sns.despine(trim=True,  top=True, right=True,bottom=True,ax=ax6)    
ylabels = [r'terminal$ $', '\n',  r'$\vspace{-25pt}$error$ $', "\n",
           r"\fontsize{14pt}{3em}\selectfont{}{$(x^*- X_T)^2$}" ]
plt.ylabel(ylabels[0] +ylabels[1]+ylabels[2] +ylabels[3]+ylabels[4],
           multialignment='center', fontsize=24, linespacing=0.5)
ax6.spines['bottom'].set_color('#363636')
ax6.spines['top'].set_color('#363636')
ax6.xaxis.label.set_color('#363636')
ax6.tick_params(axis='x', colors='#363636')

ax6.yaxis.label.set_color('#363636')
ax6.tick_params(axis='y', colors='#363636')       
plt.tick_params(axis='y', which='major',color='#4f4949')#, labelsize=16) 
plt.tick_params(axis='x', which='major',color='#4f4949')#, labelsize=16) 
ax6.xaxis.set_tick_params(width=0)


########################################################
ax1.axis('on')
#ax2.axis('on')
plt.tight_layout()
plt.subplots_adjust(wspace=0.5,hspace=0.5)


#################################################
##here I add a nested gridspec to manipulate the horizontal whitespace
##since I dont want it to be equal for every subplot
gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0:2,3], 
                                        hspace=0.1)
ax10 = fig.add_subplot(gs01[0])#fig.add_subplot(gs[0, 3])

    
sns.pointplot(x="N",  y="control", hue='M',
              data=dpf_df, dodge=.532, join=True, palette=["m", "g", 'grey'],
              zorder=10,
              markers="o", scale=1, ci=None ,ax=ax10)
plt.yscale('log')
handles2, labels2 = ax10.get_legend_handles_labels()
for li,lab in enumerate(labels2[:2]):    
    labels2[li] = r'DPF - M:' +labels2[li] 
leg52 = ax10.legend(handles2[:3], labels2[:3], title=None,
          handletextpad=0.5, columnspacing=3.2,handlelength=0.8,# bbox_to_anchor=[-0.5, 0.655],
          loc=1, ncol=1, frameon=True,fontsize = 'small',shadow=None,
          framealpha =0,edgecolor ='#0a0a0a')

for text in leg52.get_texts():
    text.set_color('#4f4949')
    
ax10.set_ylabel(r'control$ $'+'\n'+r' \fontsize{14pt}{3em}\selectfont{}{$\log \, \| u(x,t) \|_2^2$}',
                multialignment='center',fontsize=24, linespacing=0.75)

ax10.set_xlabel(r'')
ax10.tick_params(axis="both",direction="in", top=True, 
                 right=True,color='#4f4949')

#ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
ax10.xaxis.set_minor_locator(tck.AutoMinorLocator())
ax10.tick_params(axis="both", which='minor',direction="in", top=True, 
                 right=True,color='#4f4949')

plt.setp(ax10.get_xticklabels(), visible=False)

ax11 = fig.add_subplot(gs01[1], sharex=ax10)#fig.add_subplot(gs[1, 3], sharex=ax10)


g = sns.pointplot(x="N",  y="end_error", hue='M',
              data=dpf_df, dodge=.532, join=True, palette=["m", "g", 'grey'],
              zorder=10,
              markers="o", scale=1, ci=None ,ax=ax11)

#plt.yscale('log')
# handles3, labels3 = ax11.get_legend_handles_labels()
# leg51 = ax11.legend(handles3[:3], labels3[:3], title=None,
#           handletextpad=0.5, columnspacing=3.2,handlelength=0.8, #bbox_to_anchor=[-0.5, 0.655],
#           loc=1, ncol=1, frameon=True,fontsize = 'small',shadow=None,framealpha =0,edgecolor ='#0a0a0a')
# for text in leg51.get_texts():
#     text.set_color('#4f4949')
plt.legend([],[], frameon=False)    
ylabels = [r'terminal$ $', '\n',  r'$\vspace{-25pt}$error$ $', "\n", r"\fontsize{14pt}{3em}\selectfont{}{$(x^*- X_T)^2$}" ]
ax11.set_ylabel(ylabels[0] +ylabels[1]+ylabels[2] +ylabels[3]+ylabels[4],
                multialignment='center', fontsize=24, linespacing=0.5)
ax11.set_xlabel('particles N', fontsize=26)
ax11.tick_params(axis="both",direction="in", top=True, right=True)
#ax11.locator_params(nbins=3,axis='x')
ax11.set_xticks([1,3,5],['$600$', '$800$', '$1000$'])
#ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
ax11.xaxis.set_minor_locator(tck.AutoMinorLocator())
ax11.tick_params(axis="both", which='minor',direction="in", top=True, 
                 right=True, color='#4f4949')
plt.ylim(0.02,0.12)
fig.subplots_adjust(wspace=0.65,hspace=0.55)
#plt.tight_layout()
plt.show()
plt.savefig(savefolder+'Evolutionary_non_path_2d_pheno_final.png', 
            bbox_extra_artists=(leg5,text), bbox_inches='tight',dpi=300, 
            pad_inches = 0.2, transparent='False',  facecolor='white')
plt.savefig(savefolder+'Evolutionary_non_path_2d_pheno_final.pdf', 
            bbox_extra_artists=(leg5,text), bbox_inches='tight',dpi=300,  
            pad_inches = 0.2, transparent='False',  facecolor='white')



#%% plot supplementary figures, forward, backward and controlled densities for supplement

w = 2.
Y, X = np.mgrid[0:2:200j, -w:w:200j]
XY = np.concatenate( (X.reshape(1,-1), Y.reshape(1,-1)), axis=0 )

F_XY = f(XY)
U = F_XY[0]
V = F_XY[1]
speed =  np.log((1-U)**2 + b*(V - U**2)**2)#np.sqrt(U**2 + V**2)
fig = plt.figure(figsize=(8, 3.4))
#gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1])
gs = gridspec.GridSpec(nrows=1, ncols=2, height_ratios=[1])

#  Varying density along a streamline
ax0 = fig.add_subplot(gs[0, 0])
plt.contourf(X, Y, speed.reshape(X.shape), cmap=plt.cm.BuPu_r,levels=150,
             alpha=0.65)

#plt.colorbar()
#


ax0.streamplot(X, Y, U.reshape(X.shape), V.reshape(Y.shape), 
               density=[0.75, 0.75], color='#4f4949')
#ax0.plot(-1.2,2,'o')


#ax0.plot( (bridg2d.Z[0]), (bridg2d.Z[1]),alpha=0.31, c='grey',lw=1) '#C70039'
ax0.plot( np.mean(bridg2dB[0],axis=0), np.mean(bridg2dB[1],axis=0),'.',
         alpha=0.85, c='#900C3F',lw=2.5,label= r'$\mu_{q_{t}(x)}$') 
#cm.viridis(0.97)
ax0.plot( np.mean(bridg2dB[0],axis=0)+ np.std(bridg2dB[0],axis=0), 
         np.mean(bridg2dB[1],axis=0) + np.std(bridg2dB[1],axis=0),'--',
         alpha=0.85,c='#900C3F',lw=2.5,label= r'$\sigma_{q_{t}(x)}$')
ax0.plot( np.mean(bridg2dB[0],axis=0)- np.std(bridg2dB[0],axis=0), 
         np.mean(bridg2dB[1],axis=0) - np.std(bridg2dB[1],axis=0),'--',
         alpha=0.85, c='#900C3F',lw=2.5)
# ax0.plot( (bridg2d.B[0,::10]).T, (bridg2d.B[1,::10]).T,alpha=0.51, c='grey',lw=0.5)
# ax0.plot( (bridg2d.B[0,::10,200]).T, (bridg2d.B[1,::10,200]).T,'.',alpha=0.91, c=cm.viridis(0.685),lw=0.5,markersize=2)
# ax0.plot( (bridg2d.B[0,::10,200:218]).T, (bridg2d.B[1,::10,200:218]).T,'-',alpha=0.71, c=cm.viridis(0.685),lw=1)


# ax0.plot( np.mean(bridg2d.Z[0],axis=0), np.mean(bridg2d.Z[1],axis=0),'.',alpha=0.85, c=cm.viridis(0.97),lw=2.5)
# ax0.plot( np.mean(bridg2d.Z[0],axis=0)+ np.std(bridg2d.Z[0],axis=0), np.mean(bridg2d.Z[1],axis=0) + np.std(bridg2d.Z[1],axis=0),'--',alpha=0.85,c=cm.viridis(0.97),lw=2.5)
# ax0.plot( np.mean(bridg2d.Z[0],axis=0)- np.std(bridg2d.Z[0],axis=0), np.mean(bridg2d.Z[1],axis=0) - np.std(bridg2d.Z[1],axis=0),'--',alpha=0.85, c=cm.viridis(0.97),lw=2.5)
ax0.plot( (bridg2dZ[0,::2]).T, (bridg2dZ[1,::2]).T,alpha=0.41, c='#C8C8C8',
         lw=0.55,label= r'$\rho_t(x)$ ')
ax0.plot( (bridg2dZ[0,::4,418]).T, (bridg2dZ[1,::4,418]).T,'.',alpha=0.91, 
         c=cm.viridis(0.685),lw=0.5,markersize=3, label='particle')
ax0.plot( (bridg2dZ[0,::4,400:425]).T, (bridg2dZ[1,::4,400:425]).T,'-',
         alpha=0.71, c=cm.viridis(0.685),lw=1)


# ax0.plot(fixed_points[0,0], fixed_points[1,0], 'go')
# ax0.plot(fixed_points[0,2], fixed_points[1,2], 'X', c='silver')
# ax0.plot(fixed_points[0,1], fixed_points[1,1], 'o', c='#4f4949')
ax0.plot(y1[0], y1[1], 'go', label = r'$\mathbf{x}_0$', markersize=10)
ax0.plot(y2[0], y2[1], 'X', c='silver', label=r'$\mathbf{x}^*$', markersize=10)


handles, labels = ax0.get_legend_handles_labels()
to_skip = (bridg2dZ[0,::2]).shape[0]-1
handles2 = [ handles[ii] for ii in [2,0,1  ] ]
labels2 = [labels[ii] for ii in [ 2,0,1  ] ]
handles2[0].set_linewidth(1.5)

handles2b = [ handles[ii] for ii in [-2,-1,-3  ] ]
labels2b = [labels[ii] for ii in [ -2,-1,-3  ] ]
leg1 = ax0.legend(handles2b, labels2b, title=None,
          handletextpad=0.5, columnspacing=3.2,handlelength=0.8, 
          bbox_to_anchor=[-0.52, 0.4255],labelspacing = 0.3,
          loc=2, ncol=1, frameon=True,fontsize = 'large',shadow=None,
          framealpha =0,edgecolor ='#0a0a0a')
for text in leg1.get_texts():
    text.set_color('#4f4949')

leg2 = ax0.legend(handles2, labels2, title=None,
          handletextpad=0.5, columnspacing=3.2,handlelength=0.8, 
          bbox_to_anchor=[-0.52, 1.05],labelspacing = 0.3,
          loc=2, ncol=1, frameon=True,fontsize = 'large',shadow=None,
          framealpha =0,edgecolor ='#0a0a0a')
for text in leg2.get_texts():
    text.set_color('#4f4949')

ax0.add_artist(leg1)
#ax0.set_title('Varying Density')

ax0.spines['right'].set_visible(True)
ax0.spines['top'].set_visible(True)
#ax0.spines['left'].set_visible(False)
#ax0.spines['bottom'].set_visible(False)
ax0.set_xticks([-2,0, 2])
ax0.set_yticks([0, 1,2])
ax0.set_xlim(-2,2)
ax0.set_ylim(0,2)
ax0.set_xlabel(r'x')
ax0.set_ylabel(r'y')



#  Varying density along a streamline
ax2 = fig.add_subplot(gs[0, 1])
plt.contourf(X, Y, speed.reshape(X.shape), cmap=plt.cm.BuPu_r,levels=150,
             alpha=0.65)
#plt.colorbar()
ax2.streamplot(X, Y, U.reshape(X.shape), V.reshape(Y.shape), density=[1, 1], 
               color='#4f4949')



ax2.plot( np.mean(bridg2dB[0],axis=0), np.mean(bridg2dB[1],axis=0),'.',
         alpha=0.85, c='#900C3F',lw=3.1,label= r'$\mu_{q_{t}}$') 
#cm.viridis(0.97)
ax2.plot( np.mean(bridg2dB[0],axis=0)+ np.std(bridg2dB[0],axis=0), 
         np.mean(bridg2dB[1],axis=0) + np.std(bridg2dB[1],axis=0),'--',
         alpha=0.85,c='#900C3F',lw=3.2,label= r'$\sigma_{q_{t}}$')

ax2.plot( np.mean(bridg2dB[0],axis=0)- np.std(bridg2dB[0],axis=0), 
         np.mean(bridg2dB[1],axis=0) - np.std(bridg2dB[1],axis=0),'--',
         alpha=0.85, c='#900C3F',lw=3.2)


#ax0.plot( (bridg2d.Z[0]), (bridg2d.Z[1]),alpha=0.31, c='grey',lw=1)
ax2.plot( np.mean(Fcont[0],axis=0), np.mean(Fcont[1],axis=0),'-',
         alpha=0.85, c=cm.viridis(0.97),lw=2.5,
         label= r' $\mu_{\hat{q}^{DPF}_{t}}$')
ax2.plot( np.mean(Fcont[0],axis=0)+ np.std(Fcont[0],axis=0), 
         np.mean(Fcont[1],axis=0) + np.std(Fcont[1],axis=0),
         linestyle=(0, (1.5, 3)),alpha=0.85,c=cm.viridis(0.97),lw=2.5,
         label= r'$\sigma_{\hat{q}^{DPF}_{t}}$')
ax2.plot( np.mean(Fcont[0],axis=0)- np.std(Fcont[0],axis=0), 
         np.mean(Fcont[1],axis=0) - np.std(Fcont[1],axis=0),
         linestyle=(0, (1.5, 3)),alpha=0.85, c=cm.viridis(0.97),lw=2.5)
# ax2.plot( (bridg2d.B[0,::10]).T, (bridg2d.B[1,::10]).T,alpha=0.51, c='grey',lw=0.5)
# ax2.plot( (bridg2d.B[0,::10,200]).T, (bridg2d.B[1,::10,200]).T,'.',alpha=0.91, c=cm.viridis(0.685),lw=0.5,markersize=2)
# ax2.plot( (bridg2d.B[0,::10,200:218]).T, (bridg2d.B[1,::10,200:218]).T,'-',alpha=0.71, c=cm.viridis(0.685),lw=1)

# ax0.plot(fixed_points[0,0], fixed_points[1,0], 'go')
# ax0.plot(fixed_points[0,2], fixed_points[1,2], 'X', c='silver')
# ax0.plot(fixed_points[0,1], fixed_points[1,1], 'o', c='#4f4949')
ax2.plot(y1[0], y1[1], 'go',zorder=10, markersize=9)
ax2.plot(y2[0], y2[1], 'X', c='silver',zorder=10, markersize=9)
#ax0.set_title('Varying Density')
ax2.spines['right'].set_visible(True)
ax2.spines['top'].set_visible(True)
#ax1.set_title('Varying Color')
ax2.set_xticks([-2,0, 2],size=18)
ax2.set_yticks([0, 1,2],size=18)
ax2.set_xlim(-2,2)
ax2.set_ylim(0,2)
ax2.set_xlabel(r'x',labelsize=20)
ax2.set_ylabel(r'y',labelsize=20)
#ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')

handles, labels = ax2.get_legend_handles_labels()

leg4 = ax2.legend(handles, labels, title=None,
          handletextpad=0.5, columnspacing=3.2,handlelength=0.8, 
          bbox_to_anchor=[-0.53, 1.05],
          loc=2, ncol=1, frameon=True,fontsize = 'large',shadow=None,
          framealpha=0,edgecolor ='#0a0a0a')
for text in leg4.get_texts():
    text.set_color('#4f4949')

plt.savefig(savefolder+'App.non_path_2d_pheno_forward_backward_constrained_particles_backward_fade.png',
            bbox_inches='tight',dpi=300 , pad_inches = 1, transparent='False', 
            facecolor='white')
plt.savefig(savefolder+'App.non_path_2d_pheno_forward_backward_constrained_particles_backward_fade.pdf',
            bbox_inches='tight',dpi=300,  pad_inches = 1, transparent='False', 
            facecolor='white')


