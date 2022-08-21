# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 23:04:25 2021

@author: maout
"""



##plot non extreme double well - trajectory, control and statistics of control
import pickle  
"""      
to_save = dict()

to_save['Fcont1'] = Fcont1
to_save['Fcont2'] = Fcont2
to_save['Fcont_pice'] = Fcont_pice
to_save['h'] = h
to_save['used_u1'] = used_u1
to_save['used_u2'] = used_u2
to_save['used_u3'] = used_u3
to_save['alph'] = alph 
to_save['N'] = N
to_save['M'] = M
to_save['Npice'] = Npice
to_save['eta'] = eta
to_save['reps'] = reps
to_save['t2'] = t2
to_save['x1'] = x1
to_save['x2'] = x2
to_save['timegrid'] = timegrid
to_save['At'] = At
to_save['iteri'] = iteri
to_save['bridge_reweight.B'] = bridge_reweight.B
to_save['bridge_reweight.Z'] = bridge_reweight.Z
to_save['bridge_reweight.Ztr'] = bridge_reweight.Ztr
#to_save['bridge_reweight.calculate_u'] = bridge_reweight.calculate_u
to_save['bridge_deter.B'] = bridge_deter.B
to_save['bridge_deter.Z'] = bridge_deter.Z
#to_save['bridge_deter.calculate_u'] = bridge_deter.calculate_u
to_save['h'] = h
to_save['g'] = g

filename = 'double_well_end_0_all_3_N_%d_M_%d_for_plotting'
pickle.dump(to_save, open(filename+'.dat', "wb"))
"""
filename = 'double_well_end_1.000_all_3_N_2000_M_100_for_plotting'
file = open(filename +'.dat','rb')
to_save = pickle.load(file)

Fcont1 = to_save['Fcont1']  
Fcont2 = to_save['Fcont2']  
Fcont_pice = to_save['Fcont_pice'] 

used_u1 = to_save['used_u1']  
used_u2 = to_save['used_u2']  
used_u3 = to_save['used_u3'] 

t2 = to_save['t2']
#t1 = to_save['t1']
x2 = to_save['x2']
x1 = to_save['x1']
timegrid = to_save['timegrid']
iteri = to_save['iteri'] 



bridge_reweight_B = to_save['bridge_reweight.B']  
bridge_reweight_Z  = to_save['bridge_reweight.Z']  
bridge_reweight_Ztr  = to_save['bridge_reweight.Ztr']
#to_save['bridge_reweight.calculate_u'] = bridge_reweight.calculate_u
#bridge_deter_B = to_save['bridge_deter.B']  
#bridge_deter_Z = to_save['bridge_deter.Z']

#%%

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 'figure'
plt.rcParams['savefig.facecolor'] = (1,1,1,0)
plt.rcParams['savefig.bbox'] = 'standard'#"tight"
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'     # not available in Colab
plt.rcParams['font.sans-serif'] = 'Verdana'#'Helvetica'  # not available in Colab
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.linewidth'] = 2
plt.rc('axes',edgecolor='#4f4949')
plt.rcParams['figure.frameon'] = False
#plt.rcParams['figure.subplot.hspace'] = 0.05
#plt.rcParams['figure.subplot.wspace'] = 0.05
#plt.rcParams['figure.subplot.left'] = 0.4
#plt.rcParams['figure.subplot.right'] = 0.5            
plt.rcParams['text.usetex'] = True

#%%
t1 = 0
dt = 0.001
def f(x,t=0): 
    return 4*x-4*x*x*x
purple_pal = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)#sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95,  as_cmap=True)#sns.cubehelix_palette(as_cmap=True)#sns.color_palette("flare", as_cmap=True)#
endti = 1000
#potential 
V = lambda x: -2* x**2 + x**4
#points on x axis to evaluate the potential
xis = np.linspace(-1.5,1.5,200)
dx = xis[1]-xis[0]
#one dimentional potential
Vtime = np.zeros((endti,xis.size ))#

fxis = f(xis)
edges = list(xis - dx/2)
edges.append(xis[-1]+dx/2)
for ti in range(endti):
  
  Vtime[ti,:],_ = np.histogram( bridge_reweight_B[:,endti-ti-1], bins=np.array(edges), density=True)
  #Vtime[ti,:] = (fxis + bridge2.grad_log_q[ti](np.atleast_2d(xis).T) -bridge2.ln_roD[ti](np.atleast_2d(xis).T))*dt#(-#/dx#bridge2.psi[ti](np.atleast_2d(xis).T) #
t_start = -0.05
t_stop = 1.05
pad1 = int((t1-t_start)/dt)
#print(pad1)
pad2 = int((t_stop-t2)/dt)
Vtime2 = np.pad(Vtime, ((pad1,pad2),(0,0)),  mode='constant')
#potential for every time point
Vs = V(xis)
Vpot = np.tile( Vs, (endti+pad1+pad2,1))
#print(Vtime.shape)
#%%
##line for potential depth
from mpl_toolkits import mplot3d
#ax = plt.figure().add_subplot(projection='3d')
plt.figure()
plt.plot(Vs*0.25,xis)#,zs=0, zdir='z')
#%%
threedim = np.array([Vs, xis, np.ones(xis.size)])#
threedim2 = np.array([Vs, xis, np.ones(xis.size)*1.005])
threedim3 = np.array([Vs, xis, np.ones(xis.size)*1.01])
threedim4 = np.array([Vs, xis, np.ones(xis.size)*1.0125])
threedim5 = np.array([Vs, xis, np.ones(xis.size)*1.015])
threedim6 = np.array([Vs, xis, np.ones(xis.size)*1.0175])
threedim7 = np.array([Vs, xis, np.ones(xis.size)*1.02])
threedim8 = np.array([Vs, xis, np.ones(xis.size)*1.0225])
thet = 1.5#1.48#45#np.pi/2
rot = np.array([[np.cos(thet) , 0.0, np.sin(thet)], [0, 1, 0.0], [ -np.sin(thet), 0, np.cos(thet)]])
dashed = rot @threedim
dashed2 = rot @threedim2
dashed3 = rot @threedim3
dashed4 = rot @threedim4
dashed5 = rot @threedim5
dashed6 = rot @threedim6
dashed7 = rot @threedim7
dashed8 = rot @threedim8


plt.figure(),
plt.plot(dashed[0],dashed[1], c='#4f4949')
plt.plot(dashed2[0],dashed2[1],c='#4f4949',alpha=0.85)
plt.plot(dashed3[0],dashed3[1], c='#696161',alpha=0.65)
plt.plot(dashed4[0],dashed4[1], c='#696161',alpha=0.55)
plt.plot(dashed5[0],dashed5[1], c='#847979',alpha=0.45)
plt.plot(dashed6[0],dashed6[1], c='#847979',alpha=0.35)
plt.plot(dashed7[0],dashed7[1], c='#9c9494',alpha=0.25)
plt.plot(dashed8[0],dashed8[1], c='#9c9494',alpha=0.15)
plt.plot(threedim[0],threedim[1])
#plt.plot(dashed2[0],dashed2[1])
#%%
ax = plt.figure().add_subplot(projection='3d')
plt.plot(dashed[0],dashed[1],dashed[2])#,zs=0, zdir='z')
plt.xlabel('vs')
plt.ylabel('xs')
ax.view_init(elev=90., azim=90)
#%%
from matplotlib import cm
# Bmean1 = np.mean(bridge_deter.B[:,::-1],axis=0)
# Bstd1 = np.std(bridge_deter.B[:,::-1],axis=0)
Bmean2 = np.mean(bridge_reweight_B[:,::-1],axis=0)
Bstd2 = np.std(bridge_reweight_B[:,::-1],axis=0)


  #'#282828'
##plot
#import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#purple_pal = sns.color_palette('PuRd', as_cmap=True)#('magma_r', as_cmap=True)#sns.cubehelix_palette(as_cmap=True)##sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)#sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95,  as_cmap=True)#
purple_pal = cm.magma_r #sns.color_palette('magma_r', as_cmap=True)
grey_pal = plt.get_cmap('Greys_r')
ds_palette = plt.get_cmap('plasma')#plt.get_cmap('cool')#sns.diverging_palette(145, 300, s=60, as_cmap=True)##sns.diverging_palette(145, 300, s=60, as_cmap=True)
my_mag =  ds_palette(0.33)#ds_palette(0.1)
my_green = ds_palette(0.5)#ds_palette(0.9)
my_mag2 = ds_palette(0.25)
my_green2 = ds_palette(0.75)


fig = plt.figure(figsize= (5,5))
plt.imshow(Vpot.T, origin='lower',interpolation='nearest',cmap=grey_pal,alpha=0.7, extent=[-0.05, 1.05, -1.5,1.5],zorder=0,vmin=-1, vmax=0.4)
plt.imshow(Vtime2.T, origin='lower',interpolation='nearest',cmap=purple_pal,alpha=0.55, extent=[-0.05, 1.05, -1.5,1.5],zorder=2)#,vmin=-150, vmax=100)
##plot potential line
pot_ord = 3 ##level in the plot where potential appears
plt.plot(dashed[0]-0.95,dashed[1], c='#4f4949',zorder=pot_ord,alpha=0.95)
plt.plot(dashed2[0]-0.95,dashed2[1],c='#4f4949',alpha=0.85,zorder=pot_ord)
plt.plot(dashed3[0]-0.95,dashed3[1], c='#696161',alpha=0.65,zorder=pot_ord)
plt.plot(dashed4[0]-0.95,dashed4[1], c='#696161',alpha=0.55,zorder=pot_ord)
plt.plot(dashed5[0]-0.95,dashed5[1], c='#847979',alpha=0.45,zorder=pot_ord)
plt.plot(dashed6[0]-0.95,dashed6[1], c='#847979',alpha=0.35,zorder=pot_ord)
plt.plot(dashed7[0]-0.95,dashed7[1], c='#9c9494',alpha=0.25,zorder=pot_ord)
plt.plot(dashed8[0]-0.95,dashed8[1], c='#9c9494',alpha=0.15,zorder=pot_ord)
##  
 
# plt.plot(timegrid,Bmean1 ,color= my_mag,lw=5,label='$\mu_{{Q}}$',zorder=4)
# plt.plot(timegrid,(Bmean1 +Bstd1 ),'-',color=my_green,lw=5,label='$\sigma_{Q}$',zorder=4)   
# plt.plot(timegrid,(Bmean1 -Bstd1) ,'-',color=my_green,lw=5,zorder=4)  
plt.plot(timegrid,Bmean2,linestyle='-',color=my_mag,lw=5,label='$\mu_t^{\mathrm{gDPF}}$',zorder=5) 
plt.plot(timegrid,(Bmean2+Bstd2),color=my_green,linestyle='--',lw=5,label='$\sigma_t^{\mathrm{gDPF}}$',zorder=5)   
plt.plot(timegrid,(Bmean2-Bstd2),color=my_green,linestyle='--',lw=5,zorder=5) 

plt.plot([0,t2],[x1,x2],'o',markersize=10, color='silver',zorder=6)
plt.tick_params(axis='both', which='major', labelsize=20)
#legend out
# legend = plt.legend(frameon = 1,prop={'size': 15}, loc='upper left',bbox_to_anchor=(1.25, 1))
# frame = legend.get_frame()
# frame.set_facecolor('white')
# frame.set_edgecolor('white')
#legend in
legend = plt.legend(frameon = 1,prop={'size': 12}, loc='upper left',ncol=1,
                    labelspacing=.0,bbox_to_anchor=(0.1, 0.98),
                    markerscale=None,handlelength=0.85,columnspacing=0.45)
frame = legend.get_frame()
frame.set_facecolor('None')
frame.set_edgecolor('None')
#plt.title(r'particles $N=%d$,  intervals $k=%d$, $\sigma=%.2f$, $T=%.6f$, $x_2 = %.3f$' %(N,k,g,T,x2),fontsize=15)
plt.xlabel('time',fontsize=26)
plt.ylabel(r'$X_t$',fontsize=26)

ax2 = plt.gca()
ax2.set_yticks([-1,0,1])
ax2.set_xticks([0,0.5,1])
ax2.set_aspect(0.375)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
#plt.colorbar()
plt.clim(0,4)
ax2.spines['bottom'].set_color('#363636')
ax2.spines['top'].set_color('#363636')
ax2.xaxis.label.set_color('#363636')
ax2.tick_params(axis='x', colors='#363636')
ax2.yaxis.label.set_color('#363636')
ax2.tick_params(axis='y', colors='#363636')



# These are in unitless percentages of the figure size. (0,0 is bottom left)
#left, bottom, width, height = [0.2875, 0.62, 0.25, 0.25]
left, bottom, width, height = [0.54, 0.15, 0.25, 0.25]
ax1 = fig.add_axes([left, bottom, width, height])
#ax1 = inset_axes(ax2, width="35%", height="30%", loc=2)
ax1.plot(bridge_reweight_B[:,::-1].T,'-', color='maroon',zorder=3,alpha=0.1,lw=1)
ax1.plot(bridge_reweight_Z[:,:-1].T, color='grey',alpha=0.82, zorder=2,lw=1)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.patch.set_alpha(0.0)
ax1.set_xlim(left=-150)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_aspect(275)

ax1.plot([0,t2/dt],[x1,x2],'o',markersize=5, color='silver',zorder=5)
ax1.arrow(50,-1.65,700,0, head_width=0.3,head_length=100, capstyle='round',zorder=5, fc='#363636', ec='#363636',joinstyle='round')
ax1.arrow(-30,-1.3,0,2., head_width=100,head_length=0.3, capstyle='round',zorder=1,joinstyle='round',fc='#363636', ec='#363636')
ax1.set_xlabel('time',fontsize=10,labelpad=-1)
ax1.set_ylabel(r'$X_t$',fontsize=10,labelpad=-5)
ax1.xaxis.label.set_color('#363636')
ax1.yaxis.label.set_color('#363636')
#fig.tight_layout()
#ax1.apply_aspect()
#fig =plt.gcf()
##save from gui from proper placement of inset!!!
plt.savefig("app_double_well_extreme_backward_flows.png", bbox_inches='tight',dpi=300, transparent='False',  facecolor='white')
plt.savefig("app_double_well_extreme_backward_flows.pdf", bbox_inches='tight',dpi=300, transparent='False',  facecolor='white')


#%%

### plot two controlled trajectories, and controlled distribtion mean and std for three frameworks
import matplotlib.gridspec as gridspec
##needs Fcont1, Fcont2, Fcont_pice
#mean_Fc1 = np.mean(Fcont1, axis=0)
mean_Fc2 = np.mean(Fcont2, axis=0)
mean_Fcpice = np.mean(Fcont_pice, axis=0)

#std_Fc1 = np.std(Fcont1, axis=0)
std_Fc2 = np.std(Fcont2, axis=0)
std_Fcpice = np.std(Fcont_pice, axis=0)

Vtime2b = Vtime2 < 0

purple_pal = cm.magma_r #sns.color_palette('magma_r', as_cmap=True)
grey_pal = plt.get_cmap('Greys_r')
ds_palette = plt.get_cmap('plasma')#plt.get_cmap('cool')#sns.diverging_palette(145, 300, s=60, as_cmap=True)##sns.diverging_palette(145, 300, s=60, as_cmap=True)
my_mag =  ds_palette(0.33)#ds_palette(0.1)
my_mag2 =  ds_palette(0.45)
#my_green = ds_palette(0.5)#ds_palette(0.9)
my_orange = ds_palette(0.75)
my_orange2 = ds_palette(0.85)


fig = plt.figure(figsize= (12,3.5))
gs = gridspec.GridSpec(nrows=1, ncols=4)
gs01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0:2], wspace=0.3)
ax1 = fig.add_subplot(gs01[0, 0])
#plt.subplot(1,4,1)
############### with background ###########################################################
#""" uncomment this
plt.imshow(Vpot.T, origin='lower',interpolation='nearest',cmap=grey_pal,
           alpha=0.7, extent=[-0.05, 1.05, -1.5,1.5],zorder=0,vmin=-1, 
           vmax=0.4)
plt.imshow(Vtime2b.T, origin='lower',interpolation='nearest',cmap=purple_pal,
           alpha=0.55, extent=[-0.05, 1.05, -1.5,1.5],
           zorder=2)#,vmin=-150, vmax=100)
#"""
############## without background #########################################################
#plt.imshow(Vtime2b.T, origin='lower', interpolation='nearest', cmap=purple_pal,
#           alpha=0., extent=[-0.05, 1.05, -1.5, 1.5], zorder=2)

#,vmin=-150, vmax=100)


##plot potential line
pot_ord = 3 ##level in the plot where potential appears
plt.plot(dashed[0]-0.95,dashed[1], c='#4f4949',zorder=pot_ord,alpha=0.95)
plt.plot(dashed2[0]-0.95,dashed2[1],c='#4f4949',alpha=0.85,zorder=pot_ord)
plt.plot(dashed3[0]-0.95,dashed3[1], c='#696161',alpha=0.65,zorder=pot_ord)
plt.plot(dashed4[0]-0.95,dashed4[1], c='#696161',alpha=0.55,zorder=pot_ord)
plt.plot(dashed5[0]-0.95,dashed5[1], c='#847979',alpha=0.45,zorder=pot_ord)
plt.plot(dashed6[0]-0.95,dashed6[1], c='#847979',alpha=0.35,zorder=pot_ord)
plt.plot(dashed7[0]-0.95,dashed7[1], c='#9c9494',alpha=0.25,zorder=pot_ord)
plt.plot(dashed8[0]-0.95,dashed8[1], c='#9c9494',alpha=0.15,zorder=pot_ord)
##  

plt.plot([0,t2], [x1,x2],'o',markersize=10, color='silver',zorder=6)

#plt.plot(timegrid, Fcont1[0], color= my_mag,lw=2,zorder=7,label='$X_t^{DPF}$', solid_capstyle='round')
plt.plot(timegrid, Fcont2[0], color= my_orange,lw=2,zorder=7,label='$X_t^{\mathrm{gDPF}}$', solid_capstyle='round')
plt.plot(timegrid, Fcont_pice[0], color= '#4f4949',lw=2,zorder=6,label='$X_t^{\mathrm{pice}}$', solid_capstyle='round')
plt.tick_params(axis='both', which='major', labelsize=20) 
legend = plt.legend(frameon = 1,prop={'size': 16},loc='upper right',ncol=2,
                    labelspacing=.3500,bbox_to_anchor=(0.285, 0.915), handletextpad=0.2,
                    markerscale=None,handlelength=0.75,columnspacing=0.35,
                    bbox_transform=fig.transFigure)
frame = legend.get_frame()
frame.set_facecolor('None')
frame.set_edgecolor('None')



legend.get_lines()[0].set_linewidth(4)
legend.get_lines()[1].set_linewidth(4)

#plt.title(r'particles $N=%d$,  intervals $k=%d$, $\sigma=%.2f$, $T=%.6f$, $x_2 = %.3f$' %(N,k,g,T,x2),fontsize=15)
plt.xlabel('time',fontsize=26)
plt.ylabel(r'x',fontsize=26,labelpad = -10)

ax2 = plt.gca()
ax2.set_yticks([-1,0,1])
ax2.set_xticks([0,0.5,1])
ax2.set_aspect(0.375)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
#plt.colorbar()
plt.clim(0,4)
ax2.spines['bottom'].set_color('#363636')
ax2.spines['top'].set_color('#363636')
ax2.xaxis.label.set_color('#363636')
ax2.tick_params(axis='x', colors='#363636')
ax2.yaxis.label.set_color('#363636')
ax2.tick_params(axis='y', colors='#363636')
##############################################################################
ax2 = fig.add_subplot(gs01[0, 1])
#plt.subplot(1,4,2)
############### with background ###############################################
#""" uncomment this
plt.imshow(Vpot.T, origin='lower',interpolation='nearest',cmap=grey_pal,
           alpha=0.7, extent=[-0.05, 1.05, -1.5,1.5],
           zorder=0,vmin=-1, vmax=0.4)
plt.imshow(Vtime2b.T, origin='lower', interpolation='nearest', cmap=purple_pal,
           alpha=0.55, extent=[-0.05, 1.05, -1.5,1.5], zorder=2)
#,vmin=-150, vmax=100)
#"""
############## without background #############################################
#plt.imshow(Vtime2b.T, origin='lower', interpolation='nearest', cmap=purple_pal,
#           alpha=0., extent=[-0.05, 1.05, -1.5,1.5], zorder=2)
##plot potential line
pot_ord = 3 ##level in the plot where potential appears
plt.plot(dashed[0]-0.95,dashed[1], c='#4f4949',zorder=pot_ord,alpha=0.95)
plt.plot(dashed2[0]-0.95,dashed2[1],c='#4f4949',alpha=0.85,zorder=pot_ord)
plt.plot(dashed3[0]-0.95,dashed3[1], c='#696161',alpha=0.65,zorder=pot_ord)
plt.plot(dashed4[0]-0.95,dashed4[1], c='#696161',alpha=0.55,zorder=pot_ord)
plt.plot(dashed5[0]-0.95,dashed5[1], c='#847979',alpha=0.45,zorder=pot_ord)
plt.plot(dashed6[0]-0.95,dashed6[1], c='#847979',alpha=0.35,zorder=pot_ord)
plt.plot(dashed7[0]-0.95,dashed7[1], c='#9c9494',alpha=0.25,zorder=pot_ord)
plt.plot(dashed8[0]-0.95,dashed8[1], c='#9c9494',alpha=0.15,zorder=pot_ord)
##  
 
#plt.plot(timegrid,mean_Fc1 ,color= my_mag,lw=4,label='$\mu_{{Q_{DPF}}}$',zorder=4, solid_capstyle='round')
#plt.plot(timegrid,(mean_Fc1 +std_Fc1 ),'-',color=my_mag2,lw=4,label='$\sigma_{Q_{DPF}}$',zorder=4,solid_capstyle='round')   
#plt.plot(timegrid,(mean_Fc1 -std_Fc1) ,'-',color=my_mag2,lw=4,zorder=4,solid_capstyle='round')  
plt.plot(timegrid,mean_Fc2,linestyle='-',color=my_orange,lw=4,label='$\mu_t^{\mathrm{gDPF}}$',zorder=5,dash_capstyle='round') 
plt.plot(timegrid,(mean_Fc2+std_Fc2),color=my_orange2,linestyle='-',lw=4,label='$\sigma_t^{\mathrm{gDPF}}$',zorder=5,dash_capstyle='round')   
plt.plot(timegrid,(mean_Fc2-std_Fc2),color=my_orange2,linestyle='-',lw=4,zorder=5,dash_capstyle='round') 

plt.plot(timegrid,mean_Fcpice,linestyle=(0, (3, 8)),color='#282828',lw=4,label='$\mu_t^{\mathrm{pice}}$',zorder=5,dash_capstyle='round') 
plt.plot(timegrid,(mean_Fcpice+std_Fcpice),color='#282828',linestyle=(0, (0.5, 3)),lw=4,label='$\sigma_t^{\mathrm{pice}}$',zorder=5,dash_capstyle='round')   
plt.plot(timegrid,(mean_Fcpice-std_Fcpice),color='#282828',linestyle=(0, (0.5, 3)),lw=4,zorder=5,dash_capstyle='round') 


plt.plot([0,t2],[x1,x2],'o',markersize=10, color='silver',zorder=6)



plt.tick_params(axis='both', which='major', labelsize=20)
#legend out
# legend = plt.legend(frameon = 1,prop={'size': 15}, loc='upper left',bbox_to_anchor=(1.25, 1))
# frame = legend.get_frame()
# frame.set_facecolor('white')
# frame.set_edgecolor('white')
#legend in #,bbox_to_anchor=(1, 0.95)\
legend = plt.legend(frameon = 1,prop={'size': 16},  loc='upper right',ncol=2,
                    labelspacing=.35 ,bbox_to_anchor=(0.475, 1.021),
                    handletextpad=0.2,
                    markerscale=None,handlelength=0.75,columnspacing=0.6,
                    bbox_transform=fig.transFigure)
frame = legend.get_frame()
frame.set_facecolor('None')
frame.set_edgecolor('None')
#plt.title(r'particles $N=%d$,  intervals $k=%d$, $\sigma=%.2f$, $T=%.6f$, $x_2 = %.3f$' %(N,k,g,T,x2),fontsize=15)
plt.xlabel('time',fontsize=26)
plt.ylabel(r'x',fontsize=26,labelpad = -10)

ax2 = plt.gca()
ax2.set_yticks([-1,0,1])
ax2.set_xticks([0,0.5,1])
ax2.set_aspect(0.375)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
#plt.colorbar()
plt.clim(0,4)
ax2.spines['bottom'].set_color('#363636')
ax2.spines['top'].set_color('#363636')
ax2.xaxis.label.set_color('#363636')
ax2.tick_params(axis='x', colors='#363636')
ax2.yaxis.label.set_color('#363636')
ax2.tick_params(axis='y', colors='#363636')
#################################################################################################################
##needs #used_u1[:,ti] ,        used_u2[:,ti] ,        used_u3[:,ti]
#plt.subplot(1,4,3)
gs02 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 2:4], wspace=0.6)
ax3 = fig.add_subplot(gs02[0, 0])
#plt.imshow(Vpot.T, origin='lower',interpolation='nearest',cmap=grey_pal,alpha=0.7, extent=[-0.05, 1.05, -1.5,1.5],zorder=0,vmin=-1, vmax=0.4)
T = t2

# plt.plot(timegrid[:-1], h*used_u1[1,:-1]**2,color= my_mag, lw=3)
# plt.plot(timegrid[:-1], h*used_u2[1,:-1]**2,color= my_orange,lw=3, alpha= 0.65)
# plt.plot(timegrid[:-1], h*used_u3[1,:-1]**2,color='#4f4949',lw=3, alpha= 0.65)


# plt.plot(timegrid[:-1],h* np.mean(used_u1[:,:-1]**2,axis=0),color= my_mag, lw=3, zorder = 1, alpha= 0.65)
# plt.plot(timegrid[:-1], h*np.mean(used_u2[:,:-1]**2,axis=0),'--',color= my_orange,lw=3, zorder=2)
# plt.plot(timegrid[:-1], h*np.mean(used_u3[:,:-1]**2,axis=0),color='#4f4949',lw=3, zorder = 1)
#plt.yscale('log')
#ax3 = plt.gca()
# ax2.set_yticks([-1,0,1])
# ax2.set_xticks([0,0.5,1])
#ax3.set_aspect(0.4)

my_palette = sns.color_palette( [my_orange , '#666666' ])
my_mag3 = (my_mag2[0], my_mag2[1],my_mag2[2],0.15)
my_orange3 = (my_orange2[0], my_orange2[1],my_orange2[2],0.15)
my_palette2 = sns.color_palette( [my_orange3 , '#939393' ]) 


controls = np.array([np.nansum(np.power(used_u1[:,:-1],2), axis=1)/(T/dt), np.nansum(np.power(used_u2[:,:-1],2), axis=1)/(T/dt),np.nansum(np.power(used_u3[:,:-1],2), axis=1)/(T/dt)])
import pandas as pd
df = pd.DataFrame({ r'gDPF': controls[1, :], r'pice': controls[2, :]})

sns.violinplot( data=df, palette=my_palette2, alpha=0.65,saturation=0.81)#color="0.8")
sns.stripplot( data=df,jitter=0.15,alpha=0.5, palette=my_palette,edgecolor='#363636',linewidth=0.25,size=3)

# distance across the "X" or "Y" stipplot column to span, in this case 30%
mean_width = 0.5
ax = plt.gca()
for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
    sample_name = text.get_text()  # "X" or "Y"
    # calculate the median value for all replicates of either X or Y
    mean_val = df[sample_name].mean()
    # plot horizontal lines across the column, centered on the tick
    ax.plot([tick-mean_width/2, tick+mean_width/2], [mean_val, mean_val],
            lw=4, color='silver',solid_capstyle='round',zorder=3)
sns.despine(trim=True,  top=True, right=True, bottom=True,ax=ax)
#sns.despine(offset=10, trim=True)
plt.ylabel(r'Control energy $\| u(x,t) \|^2$',fontsize=20)
ax.spines['bottom'].set_color('#363636')
ax.spines['top'].set_color('#363636')
ax.xaxis.label.set_color('#363636')
ax.tick_params(axis='x', colors='#363636')
ax.yaxis.label.set_color('#363636')
ax.tick_params(axis='y', colors='#363636')    
plt.tick_params(axis='y', which='major', labelsize=26) 
plt.tick_params(axis='x', which='major', labelsize=19) 
ax.xaxis.set_tick_params(width=0)


ax3 = fig.add_subplot(gs02[0, 1])
#plt.subplot(1,4,4)    
#end_cost1 = 1*np.sqrt((Fcont1[:,-2]-x2)**2)
end_cost2 = 1*np.sqrt((Fcont2[:,-2]-x2)**2)
end_cost3 = 1*np.sqrt((Fcont_pice[:,-2]-x2)**2)


df_end = pd.DataFrame({r'gDPF': end_cost2, r'pice': end_cost3})

sns.violinplot(  data=df_end, palette=my_palette2, alpha=0.65,saturation=0.81)#color="0.8")

sns.stripplot( data=df_end,jitter=0.15,alpha=0.5, palette=my_palette,edgecolor='#363636',linewidth=0.25,size=3)
mean_width = 0.5
ax = plt.gca()
for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
    sample_name = text.get_text()  # "X" or "Y"
    # calculate the median value for all replicates of either X or Y
    mean_val = df_end[sample_name].mean()
    # plot horizontal lines across the column, centered on the tick
    ax.plot([tick-mean_width/2, tick+mean_width/2], [mean_val, mean_val],
            lw=4, color='silver',solid_capstyle='round',zorder=3)
sns.despine(trim=True,  top=True, right=True,bottom=True,ax=ax)    
plt.ylabel(r'Terminal error $(x^*- X_T)^2$',fontsize=20)
ax.spines['bottom'].set_color('#363636')
ax.spines['top'].set_color('#363636')
ax.xaxis.label.set_color('#363636')
ax.tick_params(axis='x', colors='#363636')

ax.yaxis.label.set_color('#363636')
ax.tick_params(axis='y', colors='#363636')       
plt.tick_params(axis='y', which='major', labelsize=26) 
plt.tick_params(axis='x', which='major', labelsize=19) 
ax.xaxis.set_tick_params(width=0)
plt.subplots_adjust(wspace=0.52)
#plt.savefig("Adouble_well_extreme_backward_flows-without-background.png", bbox_inches='tight',dpi=300, transparent='False',  facecolor='white')
#plt.savefig("Adouble_well_extreme_backward_flows-without-background.pdf", bbox_inches='tight',dpi=300, transparent='False',  facecolor='white')

plt.savefig("Adouble_well_extreme_backward_flows.png", bbox_inches='tight',dpi=300, transparent='False',  facecolor='white')
plt.savefig("Adouble_well_extreme_backward_flows.pdf", bbox_inches='tight',dpi=300, transparent='False',  facecolor='white')