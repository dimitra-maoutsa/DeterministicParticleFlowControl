# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 22:31:25 2021

@author: maout
"""





import numpy as np


from matplotlib import pyplot as plt


import pickle


#%%
naming = '6dim_synchronised_instance' #Kuramoto_6N\\
file = open(naming+'.dat','rb')
to_save = pickle.load(file)   
# naming2 = 'kuramot\\2N_systematic_Kuramoto_homogeneous_repb_%d_ki_%d_yi_%d__gi_%d_N_%d_M_%d_UNCONTROLLED'%(bi, ki,yi,gii,N, M)
# file2 = open(naming2+'.dat','rb')
dim = 6
g = to_save['g']
N = to_save['N']
M = to_save['M']
Fcont = to_save['Fcont'] 
Rttcont= to_save['Rttcont']  
Fnon = to_save['Fnon'] 
Rtt = to_save['Rttnon']  
K = to_save['K']                     
y0s= to_save['y0']                
used_us = to_save['used_us'] 
ws = to_save['ws']
timegrid = to_save['timegrid']
tti = 499
dt = 0.001
h = dt
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
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 24
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



def format_exponent(ax, axis='y'):

    # Change the ticklabel format to scientific format
    ax.ticklabel_format(axis=axis, style='sci', scilimits=(-2, 2))

    # Get the appropriate axis
    if axis == 'y':
        ax_axis = ax.yaxis
        x_pos = 0.0
        y_pos = 1.0
        horizontalalignment='left'
        verticalalignment='bottom'
    else:
        ax_axis = ax.xaxis
        x_pos = 1.0
        y_pos = -0.05
        horizontalalignment='right'
        verticalalignment='top'

    # Run plt.tight_layout() because otherwise the offset text doesn't update
    #plt.tight_layout()
    ##### THIS IS A BUG 
    ##### Well, at least it's sub-optimal because you might not
    ##### want to use tight_layout(). If anyone has a better way of 
    ##### ensuring the offset text is updated appropriately
    ##### please comment!

    # Get the offset value
    offset = ax_axis.get_offset_text().get_text()

    if len(offset) > 0:
        # Get that exponent value and change it into latex format
        minus_sign = u'\u2212'
        expo = np.float(offset.replace(minus_sign, '-').split('e')[-1])
        offset_text = r'x$\mathregular{10^{%d}}$' %expo

        # Turn off the offset text that's calculated automatically
        ax_axis.offsetText.set_visible(False)

        # Add in a text box at the top of the y axis
        ax.text(x_pos, y_pos, offset_text, transform=ax.transAxes,
               horizontalalignment=horizontalalignment,
               verticalalignment=verticalalignment)
    return ax

#%%
pali = sns.diverging_palette(145, 300, s=80, as_cmap=True)
cols = [pali(0.99), pali(0.95), pali(0.85), pali(0.80), pali(0.75), pali(0.70) ]
cols2 = [pali(0.),pali(0.05),pali(0.1),pali(0.15),pali(0.2),pali(0.25)]

frecmap = plt.get_cmap( 'plasma')
minw = np.abs(np.min(ws))+0.3
maxw = np.max(ws)
intervalw = maxw +minw +0.4
print([ ( (wsi + np.pi/2)/(np.pi)   ) for wsi in ws       ])
wscols = [ frecmap( (wsi + minw)/(intervalw)   ) for wsi in ws       ]

fig9 = plt.figure(figsize=(7.04, 5.28))#plt.figure(constrained_layout=False)
gs1 = fig9.add_gridspec(nrows=4, ncols=4, wspace=1.1, hspace=1.4)
################################################################
ax1 = fig9.add_subplot(gs1[:2, 0:2])

for di in range(dim):
    plt.plot(timegrid[:tti],Fcont[di,:tti,0],'-',lw=4,c=cols[di],label=r'$\theta_{%d}$'%(di+1))      
plt.xlabel('time',labelpad=-2.5)
plt.ylabel(r'phase $\theta$')

ax1.tick_params(axis='both',which='both',direction='out', length=3, width=1,colors='#4f4949',zorder=3)
ax1.tick_params(bottom=True, top=False, left=True, right=False)

ax1.set_yticks([0,np.pi/2,np.pi, 3*np.pi/2, 2*np.pi])
ax1.set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$', r'${3 \pi}/{2}$', r'$2 \pi$'])
legend = ax1.legend()        
legend.get_frame().set_linewidth(1.8)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('white')
handles, labels = ax1.get_legend_handles_labels()
handles=[]
labels=[]
if g==0.5:
    ax1.legend(handles, labels, title='controlled ',
              handletextpad=0.5, columnspacing=3.2,handlelength=0.8, bbox_to_anchor=[0., 0.7],
              loc=3, ncol=2, frameon=True,fontsize = 'medium',shadow=None,framealpha =0,edgecolor ='#0a0a0a')    
else:
    ax1.legend(handles, labels, title='controlled  ',bbox_to_anchor=[0.00, -0.085],
              handletextpad=0.5, columnspacing=3.2,handlelength=0.8,
              loc=3, ncol=2, frameon=True,fontsize = 'medium',shadow=None,framealpha =0,edgecolor ='#0a0a0a')

######################################################################################
ax2 = fig9.add_subplot(gs1[2:4, 0:2])

for di in range(dim):    
    plt.plot(timegrid[:tti],Fnon[di,:tti,1],'.',lw=3,c=cols2[di],alpha=0.7,zorder=0)
for di in range(dim):
    plt.plot(timegrid[tti],Fnon[di,tti,1],'-',lw=3,c=cols2[di],zorder=0,label=r'$\theta_{%d}$'%(di+1)) #this line is only for the legend entry
    
    
plt.xlabel('time',labelpad=-2.5)
plt.ylabel(r'phase $\theta$')

ax2.tick_params(axis='both',which='both',direction='out', length=3, width=1,colors='#4f4949',zorder=3)
ax2.tick_params(bottom=True, top=False, left=True, right=False)

ax2.set_yticks([0,np.pi/2,np.pi, 3*np.pi/2, 2*np.pi])
ax2.set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$', r'${3 \pi}/{2}$', r'$2 \pi$'])
legend = ax2.legend()        
legend.get_frame().set_linewidth(1.8)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('white')
handles, labels = ax2.get_legend_handles_labels()
handles=[]
labels=[]
if g==0.5:
    ax2.legend(handles, labels, title='uncontrolled',
              handletextpad=0.5, columnspacing=3.2,handlelength=0.8, bbox_to_anchor=[0.00, 0.7],
              loc=3, ncol=2, frameon=True,fontsize = 'small',shadow=None,framealpha =0,edgecolor ='#0a0a0a')    
else:
    ax2.legend(handles, labels, title='uncontrolled',
              handletextpad=0.5, columnspacing=3.2,handlelength=0.8, bbox_to_anchor=[0.00, 0.6],
              loc=3, ncol=2, frameon=True,fontsize = 'small',shadow=None,framealpha =0,edgecolor ='#0a0a0a')


###########
ax7 = fig9.add_subplot(gs1[0:2, 2:4])


plt.plot(timegrid[:tti],Rtt[:tti],lw=2,c= cols2[0],zorder=3,label='uncontrolled',alpha=0.25),
plt.plot(timegrid[:tti],Rttcont[:tti],lw=2.5, c=cols[0],zorder=4,alpha=0.4,label='controlled'),

plt.plot([0,timegrid[tti]], [1/np.sqrt(dim),1/np.sqrt(dim) ],linestyle=(0,(4,3)),lw=2, c='#ee9222',alpha=0.65,label='independent'),
plt.plot([0,timegrid[tti]], [1,1 ],linestyle=(0,(4,3)),c='grey',lw=3,dash_capstyle = "round"),
plt.xlabel('time',labelpad=-2.5)
#plt.ylabel(r'order  param. $R$') 
plt.ylabel(r'order'+'\n'+r'param. $R$', multialignment='center',
           linespacing=0.75)   

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
handles = [ handles[-2],handles[0] , handles[-1]]
labels= [ labels[-2],labels[0] , labels[-1]]
if True:
    ax7.legend(handles, labels, 
              handletextpad=0.5, columnspacing=0.3,handlelength=0.5, 
              bbox_to_anchor=[0.51, 0.9],
              loc=3, ncol=3, frameon=True,fontsize ='medium',shadow=None,
              framealpha =0,edgecolor ='#0a0a0a',
              bbox_transform=fig9.transFigure)    


###########
from matplotlib.ticker import ScalarFormatter

y_formatter = ScalarFormatter(useOffset=True)
ax8 = fig9.add_subplot(gs1[2:4, 2:4])

#for di in range(dim):
#plt.plot(timegrid[:tti],np.linalg.norm(h*g**2*used_us[:,:tti,0],axis=0)**2,lw=4, c=cols[1],alpha=0.75)
plt.plot(timegrid[:tti],np.linalg.norm(h*g**2*used_us[:,:tti,0],axis=0)**2,lw=2.1, c=cols[1],alpha=0.75)
#
#ax8.spines.top=True
#ax8.spines.right = True
#plt.plot(timegrid[:tti],h*g**2*np.linalg.norm(used_u[:,:tti],axis=0),lw=4, c='grey',label=r'$\|u\|$',zorder=0)
plt.xlabel('time')
plt.ylabel(r'control'+'\n'+r'$| u(x,t)|_2^2$', multialignment='center',
           linespacing=0.75)  

ax8.tick_params(axis='both',which='major',direction='in', length=3.5, width=1,colors='#4f4949',zorder=3)
ax8.tick_params(axis='both',which='minor',direction='in', length=2.5, width=0.5,colors='#4f4949',zorder=3)
ax8.tick_params(bottom=True, top=True, left=True, right=True)
ax8.yaxis.set_major_formatter(y_formatter)
ax8 = format_exponent(ax8, axis='y')
ax8.spines['top'].set_visible(True)
ax8.spines['right'].set_visible(True)
ax8.tick_params(axis='both', which='minor', bottom=True, top=True, left=True, right=True)
ax8.minorticks_on()



fig9.subplots_adjust(right=0.985)


plt.savefig("Kuramoto_6instance.png", bbox_inches='tight', transparent='False',  facecolor='white',dpi=300)
plt.savefig("Kuramoto_6instance.pdf", bbox_inches='tight', transparent='False',  facecolor='white',dpi=300)
# legend = ax8.legend()        
# legend.get_frame().set_linewidth(1.8)
# legend.get_frame().set_facecolor('white')
# legend.get_frame().set_edgecolor('white')
# handles, labels = ax8.get_legend_handles_labels()
# ax8.legend(handles, labels, title=None,
#           handletextpad=0.5, columnspacing=0,handlelength=1,
#           loc=4, ncol=1, frameon=True,fontsize = 'small',shadow=None,framealpha =0,edgecolor ='#0a0a0a')




