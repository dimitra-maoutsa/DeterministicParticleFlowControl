# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 21:19:37 2021

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

dim = 6
h = 0.001
t1 = 0
t2 = 0.5
T = t2-t1
timegrid = np.arange(0,T+h/2,h)
N = 3000#0#0#0#2000
#g = 1
k = timegrid.size
M = 300

y2 = np.ones(dim)  
sigmas = np.array([1.0,1.5])
rep_bridge = 1   #10 different bridge instances for every setting
reps = 20 ##instanses for stochastic path evaluation of each bridge
# dy0 = np.array([np.pi/4, np.pi/2, np.pi, 3*np.pi/2])    
# y0s = np.zeros((dim,rep_bridge, dy0.size))
random.seed(22)
# for i in range(rep_bridge):
#     y0s[0, i, :] = np.random.uniform( low=0, high=2*np.pi,size=1 ) 
#     y0s[1, i, :] = (y0s[0, i, :] + dy0) %(2* np.pi)



Ks   = np.linspace(0,6,7) 
repetition = 0
Rttcont = np.zeros((Ks.size,  sigmas.size, timegrid.size,reps   ))*np.nan
Rttnon = np.zeros((  Ks.size,  sigmas.size, timegrid.size,reps   ))*np.nan
used_us = np.zeros((dim,  Ks.size,  sigmas.size,timegrid.size,  reps   ))*np.nan    
for gii in range(0,2):
    noise = gii+1
    for ki in range(Ks.size):   
        K = ki#Ks[ki]
        naming = 'Kuramoto_6N\\dim6\\Latest%d_N_systematic_Kuramoto_inhomogeneous_k_%d_gi_%d_N_%d_M_%d_repetition_%d'%(dim, K,noise,N, M,repetition)
          
        try:
            file = open(naming+'.dat','rb')
            to_save = pickle.load(file)   
            # naming2 = 'kuramot\\2N_systematic_Kuramoto_homogeneous_repb_%d_ki_%d_yi_%d__gi_%d_N_%d_M_%d_UNCONTROLLED'%(bi, ki,yi,gii,N, M)
            # file2 = open(naming2+'.dat','rb')
            # to_save2 = pickle.load(file2)
            
            #Fcont = to_save['Fcont'] 
            Rttcont[ki,gii,:,:] = to_save['Rttcont']  
            #Fnon = to_save2['Fnon'] 
            Rttnon[ki, gii,:,:] = to_save['Rttnon']  
            #Kin = to_save['K']                     
                            
            used_us[:,ki, gii,:] = to_save['used_us']   
            
        except FileNotFoundError:
            i=0
            print(naming)
        
        
    
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

#%%onset of synchronisation




onset_cont = np.zeros((Ks.size,2,20)) *np.nan
onset_uncont = np.zeros((Ks.size,2,20)) *np.nan

cont_synched_durations_m = np.zeros((Ks.size,2,20)) *np.nan
cont_synched_durations_std  = np.zeros((Ks.size,2,20)) *np.nan
cont_synched_durations_max = np.zeros((Ks.size,2,20)) *np.nan
cont_synched_durations_sum  = np.zeros((Ks.size,2,20)) *np.nan

uncont_synched_durations_m = np.zeros((Ks.size,2,20)) *np.nan
uncont_synched_durations_std  = np.zeros((Ks.size,2,20)) *np.nan
uncont_synched_durations_max = np.zeros((Ks.size,2,20)) *np.nan
uncont_synched_durations_sum  = np.zeros((Ks.size,2,20)) *np.nan

counter_cont = np.zeros((Ks.size,2))
counter_uncont =  np.zeros((Ks.size,2))

threshold = 0.95
min_duri = 20

for ki in range(Ks.size):
    
    for gi in range(2):#3                
        for repi in range(20):                    
            positions = np.where( Rttcont[ ki,gi,1:-2, repi] >=threshold)[0]  ###this one detects if there is at all any synchronisation
            synch_or_not = Rttcont[ ki, gi,1:-2, repi] >=threshold ##this is boolean indicating the synchronised positions
            
            if positions.size==0: 
                onset_cont[ki,gi,repi] = np.nan
            else:
                diffpos = np.diff(synch_or_not.astype(int)   ) #difference between consecutive steps
                
                diffpos2 = np.zeros(diffpos.size+1) # extend by one step at the beginning to detect 
                diffpos2[1:] = diffpos
                starts = np.argwhere(diffpos2 == 1)
                stops = np.argwhere(diffpos2 == -1)                        
                if starts.size == 0:
                    onset_cont[ki,gi,repi] = np.nan
                elif stops.size == 0:
                    onset_cont[ki,gi,repi] = timegrid[ starts[0] ]
                    counter_cont[ki,gi] += 1
                    dur = timegrid[ -1 ] - timegrid[ starts[0] ]
                    cont_synched_durations_m[ki,gi,repi]  = dur
                    cont_synched_durations_std[ki,gi,repi]  = 0
                    cont_synched_durations_max[ki,gi,repi]  = dur
                    cont_synched_durations_sum[ki,gi,repi]  = np.sum(  synch_or_not )/ (timegrid.size-1 - starts[0] )
                else:
                    if stops[0,0] < starts[0,0]:
                        stops = stops[1:,:]
                    durations = stops[:,0] - starts[:stops.size, 0]                            
                    synchned_more_than_50 = np.where( durations>=min_duri )[0]
                    if synchned_more_than_50.size == 0:
                        onset_cont[ki,gi,repi] = np.nan
                    else:                                
                        onset_cont[ki,gi,repi] = timegrid[ starts[synchned_more_than_50[0]] ]
                        counter_cont[ki,gi] += 1                                
                        cont_synched_durations_m[ki,gi,repi]  = np.mean(  durations  )
                        cont_synched_durations_std[ki,gi,repi]  = np.std(  durations )
                        cont_synched_durations_max[ki,gi,repi]  = np.max(  durations )
                        cont_synched_durations_sum[ki,gi,repi]  = np.sum(  synch_or_not[ starts[synchned_more_than_50[0]][0] :   ] )/(timegrid.size-1 -starts[synchned_more_than_50[0]] ) ##timesteps in synchronied

               
                
            positions = np.where( Rttnon[ ki,  gi, 1:-2,repi] >=threshold)[0]
            synch_or_not = Rttnon[ ki,  gi,1:-2, repi] >=threshold                     
            if positions.size ==0: 
                onset_uncont[ki,gi,repi] = np.nan
            else:
                diffpos = np.diff(synch_or_not.astype(int)   ) #difference between consecutive steps
                
                diffpos2 = np.zeros(diffpos.size+1) # extend by one step at the beginning to detect 
                diffpos2[1:] = diffpos
                starts = np.argwhere(diffpos2 == 1)
                stops = np.argwhere(diffpos2 == -1)
                #start_stop =[starts, stops - starts]
                if starts.size == 0:
                    onset_uncont[ki,gi,repi] = np.nan
                elif stops.size == 0:
                    onset_uncont[ki,gi,repi] = timegrid[ starts[0] ]
                    counter_uncont[ki,gi] += 1
                    dur = timegrid[ -1 ] - timegrid[ starts[0] ]
                    uncont_synched_durations_m[ki,gi,repi]  = dur
                    uncont_synched_durations_std[ki,gi,repi]  = 0
                    uncont_synched_durations_max[ki,gi,repi]  = dur
                    uncont_synched_durations_sum[ki,gi,repi]  = np.sum(  synch_or_not )/ (timegrid.size-1 - starts[0] )
                else:
                    if stops[0,0] < starts[0,0]:
                        stops = stops[1:,:]
                    durations = stops[:,0] - starts[:stops.size, 0]                            
                    synchned_more_than_50 = np.where( durations>=min_duri )[0]
                    
                    if synchned_more_than_50.size == 0:
                        onset_uncont[ki,gi,repi] = np.nan
                    else:                                
                        onset_uncont[ki,gi,repi] = timegrid[starts[synchned_more_than_50[0]] ]
                        counter_uncont[ki,gi] += 1
                        uncont_synched_durations_m[ki,gi,repi]  = np.mean(  durations )
                        uncont_synched_durations_std[ki,gi,repi]  = np.std(  durations )
                        uncont_synched_durations_max[ki,gi,repi]  = np.max(  durations  )
                        uncont_synched_durations_sum[ki,gi,repi]  = np.sum(  synch_or_not[ starts[synchned_more_than_50[0]][0] :   ] ) /(timegrid.size-1 -starts[synchned_more_than_50[0]] )

#%%

import seaborn as sns
from matplotlib import pyplot as plt


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

fig9.text(0.7, 0.95, r'coupling  $J= 3.0$', ha='center',fontsize=10)
fig9.text(0.95, 0.95, r'noise', ha='center',fontsize=10)
fig9.text(0.95, 0.73, r'$\sigma= 1.0$', ha='center',fontsize=10)
fig9.text(0.95, 0.23, r'$\sigma= 1.5$', ha='center',fontsize=10)



ax02 = fig9.add_subplot(gs1[0:4,0:4 ]) 

#color = next(ax02._get_lines.prop_cycler)['color']
plt.plot(Ks[1:], np.nanmean(np.nanmean(Rttcont[1 :, 0,:-2 ], axis=-2), axis=(-1)), c=cols[5],marker='^',label=r'$\sigma=1.0$',zorder=5 ,lw=2,markersize=6,markeredgecolor='#4f4949')  

#color = next(ax02._get_lines.prop_cycler)['color']
plt.plot(Ks[1:], np.nanmean(np.nanmean(Rttcont[1 :, 1 ,:-2], axis=-2), axis=(-1)) , c=cols[1],marker='.',label=r'$\sigma=1.5$' ,zorder=5,lw=2,markersize=10,markeredgecolor='#4f4949')    
plt.plot(Ks[1:], np.nanmean(np.nanmean(Rttnon[1:, 0,:-2 ], axis=-2), axis=(-1)) ,'--', c=cols2[5],marker='^',label=r'$\sigma=1.0$',lw=2 ,markersize=6,markeredgecolor='#4f4949')                                   
plt.plot(Ks[1:], np.nanmean(np.nanmean(Rttnon[1 :, 1,:-2 ], axis=-2), axis=(-1)) ,'--', c=cols2[1],marker='.' ,label=r'$\sigma=1.5$',lw=2,markersize=10,markeredgecolor='#4f4949')
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
              handletextpad=0.5, columnspacing=0.5,handlelength=1.3, bbox_to_anchor=[0.10, 0.0],
              loc=3, ncol=2, frameon=True,fontsize = 10,shadow=None,framealpha =0,edgecolor ='#0a0a0a')    

plt.setp(plt.gca().get_legend().get_title(), fontsize='10') 
plt.ylabel(r'order  param. $R$') 
plt.xlabel(r'coupling $J$')

#####################################

ax1 = fig9.add_subplot(gs1[0:2,4:6 ] ) 

plt.plot(timegrid[:-2],Rttcont[3,0,:-2,:],c=cols[5],alpha=1),
plt.plot(timegrid[:-2],np.mean(Rttcont[3,0,:-2,:],axis=1), '--',c='#4f4949')

ax1.set_ylim(0.0,1.01)
#plt.yticks([0.5,1.0])
plt.xticks([0.5,1.5])
ax1b = fig9.add_subplot(gs1[0:2,6:8 ] ,  sharey = ax1) 

plt.plot(timegrid[:-2],Rttnon[3,0,:-2,:],c=cols2[5],alpha=0.5)
ax1b.set_ylim(0.0,1.01)
plt.plot(timegrid[:-2],np.mean(Rttnon[3,0,:-2,:],axis=1), '--',c='#4f4949')
plt.plot([0,timegrid[-2]],[1,1], '--',c='silver',zorder=0)
#plt.ylabel(r'order  param. $R$')   
plt.xticks([0.5,1.5])
#plt.yticks([0.5,1.0])
ax2 = fig9.add_subplot(gs1[ 2:4,4:6], sharex = ax1)
ax2.set_ylim(0.0,1.01)
plt.plot(timegrid[:-2],Rttcont[3,1,:-2,:],c=cols[1],alpha=1),
plt.plot(timegrid[:-2],np.mean(Rttcont[3,1,:-2,:],axis=1), '--',c='#4f4949')
#plt.xlabel('time')
#plt.ylabel(r'order  param. $R$')   
#plt.xticks([0.5,1.5])
#plt.yticks([0.5,1.0])
ax2b = fig9.add_subplot(gs1[ 2:4,6:8], sharex = ax1b, sharey = ax2) 
plt.plot(timegrid[:-2],Rttnon[3,1,:-2,:],c=cols2[1],alpha=0.5)
plt.plot(timegrid[:-2],np.mean(Rttnon[3,1,:-2,:],axis=1), '--',c='#4f4949')
plt.plot([0,timegrid[-2]],[1,1], '--',c='silver',zorder=0)
ax2b.set_ylim(0.0,1.01)
#plt.xticks([0.5,1.5])



plt.savefig('systematic_Kuramoto_6N.png', bbox_inches='tight',dpi=300 , pad_inches = 0.)
plt.savefig('systematic_Kuramoto_6N.pdf', bbox_inches='tight',dpi=300,  pad_inches = 0.)



#%%




ax6 = fig9.add_subplot(gs1[0:4,4:8 ]) 

plt.plot(Ks, np.nanmean(onset_cont[:, 0], axis=(-1)) , c=cols[5],marker='^',label=r'$\sigma=0.5$',zorder=5 ,lw=2.8,markersize=6,markeredgecolor='#4f4949') 
plt.plot(Ks, np.nanmean(onset_uncont[:, 0], axis=(-1)),'--', c=cols2[5],marker='^',label=r'$\sigma=0.5$' ,zorder=5,lw=2.8,markersize=6,markeredgecolor='#4f4949')  

plt.plot(Ks, np.nanmean(onset_cont[:, 1], axis=(-1)) , c=cols[1],marker='.',label=r'$\sigma=1.0$',lw=2.8 ,markersize=12,markeredgecolor='#4f4949') 
plt.plot(Ks, np.nanmean(onset_uncont[:, 1], axis=(-1)),'--', c=cols2[0],marker='.' ,label=r'$\sigma=1.0$',lw=2.8,markersize=12,markeredgecolor='#4f4949')   

                   
plt.ylabel(r'onset of synchrony $t^{syn}$') 
plt.xlabel(r'coupling $J$')                



ax6.tick_params(axis='both',which='both',direction='in', length=3, width=1,colors='#4f4949',zorder=3)
ax6.tick_params(bottom=True, top=True, left=True, right=True)
ax6.spines['top'].set_visible(True)
ax6.spines['right'].set_visible(True)
ax6.minorticks_on()
ax6.tick_params(axis='both',which='major',direction='in', length=3.5, width=1,colors='#4f4949',zorder=3)
ax6.tick_params(axis='both',which='minor',direction='in', length=2.5, width=0.5,colors='#4f4949',zorder=3)
ax6.tick_params(bottom=True, top=True, left=True, right=True)
ax6.tick_params(axis='both', which='minor', bottom=True, top=True, left=True, right=True)
####################################

ax7 = fig9.add_subplot(gs1[0:4,8:12 ]) 



plt.plot(Ks, np.nanmean(cont_synched_durations_sum[:, 0], axis=(-1)) , c=cols[5],marker='^',label=r'$\sigma=1.0$',zorder=2 ,lw=2.8,markersize=6,markeredgecolor='#4f4949') 
plt.plot(Ks, np.nanmean(uncont_synched_durations_sum[:, 0], axis=(-1)),'--', c=cols2[5],marker='^',label=r'$\sigma=1.0$' ,zorder=2,lw=2.8,markersize=6,markeredgecolor='#4f4949')  


plt.plot(Ks, np.nanmean(cont_synched_durations_sum[:, 1], axis=(-1)) , c=cols[1],marker='.',label=r'$\sigma=1.5$',lw=2.8 ,markersize=12,markeredgecolor='#4f4949') 
plt.plot(Ks, np.nanmean(uncont_synched_durations_sum[:, 1], axis=(-1)),'--', c=cols2[0],marker='.' ,label=r'$\sigma=1.5$',lw=2.8,markersize=12,markeredgecolor='#4f4949')   

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
    