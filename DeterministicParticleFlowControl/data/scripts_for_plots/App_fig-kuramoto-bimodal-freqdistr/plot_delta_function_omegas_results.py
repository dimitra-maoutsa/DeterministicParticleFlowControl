# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 19:55:28 2022

@author: maout
"""


import numpy as np
import pickle
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
plt.rcParams['xtick.minor.size'] = 10
#plt.rcParams['xtick.major.width'] = 4
plt.rcParams['text.usetex'] = True
#%%

locationw = 'C://Users//maout//Data_Assimilation_stuff//codes//Bridges//kura_revision//kura_wide0//'
location = 'C://Users//maout//Data_Assimilation_stuff//codes//Bridges//kura_revision//kura//'
pali = sns.diverging_palette(145, 300, s=80, as_cmap=True)
cols = [pali(0.99), pali(0.95), pali(0.85), pali(0.80), pali(0.75), pali(0.70)]
cols2 = [pali(0.),pali(0.05),pali(0.1),pali(0.15),pali(0.2),pali(0.25)]

dim = 6
freqs = [0.25, 0.5, 1]
y0 = np.random.normal( loc=3, scale=0.5, size=dim )  
ws = np.ones(dim)
M = 300
Ks   = np.linspace(0,4,5)
N = 3000
reps = 20
t1 = 0
t2 = 0.5
T = t2-t1
h = 0.001 #sim_prec
timegrid = np.arange(0,T+h/2,h)
noises = [0, 1]
brep = 5  ##repetitions with diffferent seed
Rttcont = np.zeros((len(noises), len(freqs), Ks.size, brep, timegrid.size, reps))
Rttnon = np.zeros((len(noises), len(freqs), Ks.size, brep, timegrid.size, reps))

Rttcontw = np.zeros((len(noises), len(freqs), Ks.size, brep, timegrid.size, reps))
Rttnonw = np.zeros((len(noises), len(freqs), Ks.size, brep, timegrid.size, reps))
plt.figure()
    
for noise in noises[:]:

    if noise==0:
        g = 0.5
    elif noise==1:
        g = 1
    
    for ifr, freq in enumerate(freqs):
        plt.subplot(2, 3, 1+ ifr)
        ax = plt.gca()
        ax.title.set_text('Noise: %.1f, freq: %.2f'%(g, freq))
        ws[:round(dim/2)] = freq
        ws[round(dim/2):] = -freq
        
        for ik, K in enumerate(Ks):
            
            for repi in range(brep):
                
                naming =location+ 'Delta%d_N_syst_Kuramoto_inh_k_%d_gi_%d_N_%d_M_%d_repetition_%d_fr_%.3f'%(dim, K,noise,N, M,repi, freq)

                file = open(naming+'.dat','rb')
                to_save = pickle.load(file)                   
                
                
                Rttcont[noise, ifr, ik, repi] = to_save['Rttcont']  
                Rttnon[noise, ifr, ik, repi] = to_save['Rttnon']
                
                namingw =locationw+ 'Delta%d_N_syst_Kuramoto_inh_k_%d_gi_%d_N_%d_M_%d_repetition_%d_fr_%.3f'%(dim, K,noise,N, M,repi, freq)

                file = open(namingw+'.dat','rb')
                to_save = pickle.load(file)                   
                
                
                Rttcontw[noise, ifr, ik, repi] = to_save['Rttcont']  
                Rttnonw[noise, ifr, ik, repi] = to_save['Rttnon']
                
        ax.plot(Ks, np.mean(np.mean(np.mean(Rttcont[noise, ifr, :, :, :], axis=-2), axis=-1), axis=-1) ,'-', c=cols[0])
        ax.plot(Ks, np.mean(np.mean(np.mean(Rttnon[noise, ifr, :, :, :], axis=-2), axis=-1), axis=-1) ,'-', c=cols2[0])
        plt.subplot(2, 3, 4+ ifr)
        ax = plt.gca()
        ax.plot(Ks, np.mean(np.mean(np.mean(Rttcontw[noise, ifr, :, :, :], axis=-2), axis=-1), axis=-1) ,'-', c=cols[0])
        ax.plot(Ks, np.mean(np.mean(np.mean(Rttnonw[noise, ifr, :, :, :], axis=-2), axis=-1), axis=-1) ,'-', c=cols2[0])
        ax.set_ylim(0, 1.01) 
#%%

fig = plt.figure()
#fig.text(0.1, 0.95, r'$\omega^0$', ha='center',fontsize=14) 
fig.text(0.5, 1.03, r'$\omega^0$', ha='center',fontsize=14) 
fig.text(0.09, 1.125, r'$g(\omega)$', ha='center',fontsize=9) 
fig.text(0.20, 0.95, r'$0.25$', ha='center',fontsize=14) 
fig.text(0.50, 0.95, r'$0.5$', ha='center',fontsize=14) 
fig.text(0.80, 0.95, r'$1.0$', ha='center',fontsize=14) 
fig.text(0.95, 0.9, r'$\theta_0$', ha='center',fontsize=14)    
for noise in noises[1:]:

    if noise==0:
        g = 0.5
    elif noise==1:
        g = 1
    
    for ifr, freq in enumerate(freqs):
        plt.subplot(2, 3, 1+ ifr)
        ax = plt.gca()
        #ax.title.set_text('Noise: %.1f, freq: %.2f'%(g, freq))
        ws[:round(dim/2)] = freq
        ws[round(dim/2):] = -freq
        
        #for ik, K in enumerate(Ks):
            
        for repi in range(brep):
            
            ax.plot(Ks, np.mean(np.mean(Rttcont[noise, ifr, :, repi, :-1,:], axis=-2), axis=-1) ,'-', c=cols[0], alpha=0.1)
            ax.plot(Ks, np.mean(np.mean(Rttnon[noise, ifr, :, repi, :-1,:], axis=-2), axis=-1) ,'-', c=cols2[0], alpha=0.1)
            plt.subplot(2, 3, 4+ ifr)
            ax2 = plt.gca()
            ax2.plot(Ks, np.mean(np.mean(Rttcontw[noise, ifr, :, repi, :-1,:], axis=-2), axis=-1) ,'-', c=cols[0], alpha=0.1)
            ax2.plot(Ks, np.mean(np.mean(Rttnonw[noise, ifr, :, repi, :-1,:], axis=-2), axis=-1) ,'-', c=cols2[0], alpha=0.1)
            
                
                
        ax.plot(Ks, np.mean(np.mean(Rttcont[noise, ifr, :, :-1, :], axis=-2), axis=(-1,-2)) ,'-', c=cols[0],marker='^',lw=2,markersize=6,markeredgecolor='#4f4949', label='controlled')
        ax.plot(Ks, np.mean(np.mean(Rttnon[noise, ifr, :, :-1, :], axis=-2), axis=(-1,-2)) ,'-', c=cols2[0],marker='^',lw=2,markersize=6,markeredgecolor='#4f4949', label='uncontrolled')
        if ifr==0:
            ax.annotate('', xy=(0.5, 1.36), xycoords='axes fraction', xytext=(3.55, 1.36), 
            arrowprops=dict(arrowstyle="<-", color='#778899', linewidth=2))
            ax.set_ylabel(r'order  param. $R$')
        ax.set_xlabel(r'coupling $J$',labelpad=-1.5)
        ax.tick_params(axis='both',which='both',direction='out', length=3, width=1,colors='#4f4949',zorder=3)
        ax.tick_params(bottom=True, top=False, left=True, right=False)   
        ax.set_xticks([0,  2,4])
        if ifr==0:
            legend = ax.legend()        
            legend.get_frame().set_linewidth(1.8)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('white')
            handles, labels = ax.get_legend_handles_labels()
            
            if True:
                ax.legend(handles, labels, 
                          handletextpad=0.5, columnspacing=0.3,handlelength=0.5,# bbox_to_anchor=[-0.15, 0.99],
                          loc=3, ncol=1, frameon=True,fontsize = 'small',shadow=None,framealpha =0,edgecolor ='#0a0a0a')    
        
        
        #from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        #if ifr==2:
            #axins = inset_axes(ax, width="30%", height="30%", 
                   #bbox_to_anchor=[20,20,1,1])#, bbox_transform=ax.transAxes)
        
        ax2.plot(Ks, np.mean(np.mean(Rttcontw[noise, ifr, :, :-1, :], axis=-2), axis=(-1, -2)) ,'-', c=cols[0],marker='^',lw=2,markersize=6,markeredgecolor='#4f4949')
        ax2.plot(Ks, np.mean(np.mean(Rttnonw[noise, ifr, :, :-1, :], axis=-2), axis=(-1, -2)) ,'-', c=cols2[0],marker='^',lw=2,markersize=6,markeredgecolor='#4f4949')
        if ifr==0:
            ax2.set_ylabel(r'order  param. $R$')
        ax2.set_xlabel(r'coupling $J$',labelpad=-1.5)
        ax2.tick_params(axis='both',which='both',direction='out', length=3, width=1,colors='#4f4949',zorder=3)
        ax2.tick_params(bottom=True, top=False, left=True, right=False)
        ax2.set_xticks([0,  2,4])    
        ax2.set_ylim(0, 1.01)  
        ax.set_ylim(0, 1.01)  
        
axm = fig.add_axes([0.91, 0.7, 0.07, 0.08])
import scipy.stats as stats
mu = np.pi
sigman = 0.5
x = np.linspace(mu - 3*1, mu + 3*1, 100)
#color_cycle = axm._get_lines.color_cycle
next_color = next(axm._get_lines.prop_cycler)['color']
next_color = next(axm._get_lines.prop_cycler)['color']
axm.plot(x, stats.norm.pdf(x, mu, sigman), c=next_color)
axm.spines['top'].set_color('none')
axm.spines['left'].set_color('none')
axm.spines['right'].set_color('none')
axm.spines['bottom'].set_position('zero')
axm.get_yaxis().set_visible(False)
axm.set_xticks([0, np.pi, 2*np.pi], [r'$0$', r'$\pi$', r'$2\pi$'   ])  # Set text labels.
axm.tick_params(axis='x', which='both', length=0)

axm2 = fig.add_axes([0.91, 0.21, 0.07, 0.08])
sigman = 1
x = np.linspace(mu - 3*sigman, mu + 3*sigman, 100)
next_color = next(axm2._get_lines.prop_cycler)['color']
next_color = next(axm2._get_lines.prop_cycler)['color']
axm2.plot(x, stats.norm.pdf(x, mu, sigman), c=next_color)
axm2.spines['top'].set_color('none')
axm2.spines['left'].set_color('none')
axm2.spines['right'].set_color('none')
axm2.spines['bottom'].set_position('zero')
axm2.get_yaxis().set_visible(False)
axm2.set_xticks([0, np.pi, 2*np.pi], [r'$0$', r'$\pi$', r'$2\pi$'   ])  # Set text labels.
axm2.tick_params(axis='x', which='both', length=0)

###

axm3 = fig.add_axes([0.05, 1.03, 0.08, 0.08])
mu = 3
sigman = 0.05
x = np.linspace(0 - 5*1, 0 + 5*1, 100)
axm3.plot(x, stats.norm.pdf(x, mu, sigman), c='#4f4f4f')
axm3.plot(x, stats.norm.pdf(x, -mu, sigman), c='#4f4f4f')
axm3.spines['top'].set_color('none')
axm3.spines['left'].set_color('none')
axm3.spines['right'].set_color('none')
axm3.spines['bottom'].set_position('zero')
axm3.get_yaxis().set_visible(False)
axm3.set_xticks([-mu-1.5, 0, mu+1.5], [r'$-\omega^0$', r'$0$', r'$\omega^0$'   ])  # Set text labels.
axm3.tick_params(axis='x', which='both', length=0)
plt.savefig('delta_omega_Kuramoto_6N.png', bbox_inches='tight',dpi=300 , pad_inches = 0.15,transparent=False, facecolor=fig.get_facecolor(),)
plt.savefig('delta_omega_Kuramoto_6N.pdf', bbox_inches='tight',dpi=300,  pad_inches = 0.15)
#%%


plt.figure()
    
for noise in noises:

    if noise==0:
        g = 0.5
    elif noise==1:
        g = 1
    
    for ifr, freq in enumerate(freqs):
        plt.subplot(2, 3, 3*(noise)+1+ ifr)
        ax = plt.gca()
        ax.title.set_text('Noise: %.1f, freq: %.2f'%(g, freq))
        ws[:round(dim/2)] = freq
        ws[round(dim/2):] = -freq
        
        #for ik, K in enumerate(Ks):
            
        for repi in range(brep):
            
            ax.plot(timegrid, np.mean(Rttcontw[noise, ifr, 1, repi, :], axis=-1) ,'-', c=cols[0], alpha=0.25)
            ax.plot(timegrid, np.mean(Rttnonw[noise, ifr, 1, repi, :], axis=-1) ,'-', c=cols2[0], alpha=0.25)
        
                
                
        #ax.plot(Ks, np.mean(np.mean(np.mean(Rttcont[noise, ifr, :, :, :], axis=-2), axis=-1), axis=-1) ,'-', c=cols[0])
        #ax.plot(Ks, np.mean(np.mean(np.mean(Rttnon[noise, ifr, :, :, :], axis=-2), axis=-1), axis=-1) ,'-', c=cols2[0])
        
        #ax.plot(Ks, np.mean(np.mean(np.mean(Rttcontw[noise, ifr, :, :, :], axis=-2), axis=-1), axis=-1) ,'--', c=cols[0])
        #ax.plot(Ks, np.mean(np.mean(np.mean(Rttnonw[noise, ifr, :, :, :], axis=-2), axis=-1), axis=-1) ,'--', c=cols2[0])
        ax.set_ylim(0, 1.01) 
        
#%%

noise=1
ifr =1
freq = freqs[ifr]  
ik = 0
K = Ks[ik]   
repi = 3 
namingw =locationw+ 'Delta%d_N_syst_Kuramoto_inh_k_%d_gi_%d_N_%d_M_%d_repetition_%d_fr_%.3f'%(dim, K,noise,N, M,repi, freq)

file = open(namingw+'.dat','rb')
to_save = pickle.load(file)                   


Fcont = to_save['Fcont'] 
Rttcon= to_save['Rttcont']  
Fnon = to_save['Fnon'] 
Rtt = to_save['Rttnon']     
##%%

plt.figure()
for i in range(dim):
    plt.subplot(dim+1, 1, i+1)
    plt.plot(timegrid, Fcont[i], alpha=0.5)                                
plt.subplot(dim+1, 1, dim+1)
plt.plot(timegrid, Rttcon, alpha=0.5)  
plt.plot(timegrid, Rtt, '--', alpha=0.5)  
plt.hlines( np.mean(np.mean(Rttcon[:-1], axis=0)  ),
           xmin=timegrid[0], xmax=timegrid[-1], colors='magenta'    )
plt.hlines( np.mean(np.mean(Rtt[:-1], axis=0)  ),
           xmin=timegrid[0], xmax=timegrid[-1], colors='k'    )





#%%

noise=0
ifr =0
freq = freqs[ifr]  
plt.figure()
for ik, K in enumerate(Ks):
    for repi in range(brep):
        namingw =locationw+ 'Delta%d_N_syst_Kuramoto_inh_k_%d_gi_%d_N_%d_M_%d_repetition_%d_fr_%.3f'%(dim, K,noise,N, M,repi, freq)

        file = open(namingw+'.dat','rb')
        to_save = pickle.load(file)  
        Fcont = to_save['Fcont'] 
        Rttcon= to_save['Rttcont']  
        Fnon = to_save['Fnon'] 
        Rtt = to_save['Rttnon']  
        
        plt.plot( K, np.mean(np.mean(Rttcon[:-1], axis=0)  ) , 'mo'    )
        
        plt.plot( K, np.mean(np.mean(Rtt[:-1], axis=0)  ) , 'ko'    )