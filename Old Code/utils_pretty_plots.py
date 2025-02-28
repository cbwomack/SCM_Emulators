def plot_box_model2(T, t, experiments, regions, colors, soln_type, coupled=False, ensemble=False, T2=None):

  n_exp = len(experiments)
  n_reg = len(regions)
  fig, ax = plt.subplots(2, 2, figsize=(12,12), sharex=True, sharey=True, constrained_layout=True)

  for i, exp in enumerate(experiments):
    if i == 0:
      r, c = 0, 0
    elif i == 1:
      r, c = 0, 1
    elif i == 2:
      r, c = 1, 0
    else:
      r, c = 1, 1

    for j, reg in enumerate(regions):
      if ensemble is False:
        if coupled:
          T_temp = np.array(T[exp])[j,:]
          T_temp_2 = np.array(T2[exp])[j,:]
        else:
          T_temp = T[exp][reg]
        n=15
        ax[r,c].plot(t, T_temp, c=brewer2_light(j), label=regions[j], lw=4)
        ax[r,c].plot(t[::n], T_temp_2[::n], 'o', c=brewer2_light(j), lw=4, markerfacecolor='white')

      else:
        T_temp = T[exp][:,j,:]
        T_mean = np.mean(T_temp, axis=0)
        T_std = np.std(T_temp, axis=0)

        ax[i].plot(t, T_mean, c=colors[j], label='Ensemble Mean', linewidth=3)
        ax[i].fill_between(t, T_mean - T_std, T_mean + T_std, color=colors[j], alpha=0.5, label='Ensemble Spread (±1 std)')
        ax[i].fill_between(t, T_mean - 2*T_std, T_mean + 2*T_std, color=colors[j], alpha=0.2, label='Ensemble Spread (±2 std)')

    ax[r,c].set_title(f'{experiments[i]}',fontsize=36)
    ax[r,c].tick_params(axis='both', which='major', labelsize=28)

  if not ensemble:
    custom_marker = Line2D(
    [0], [0],
    marker='o',
    color='black',             # Line color (ignored for the marker)
    markerfacecolor='white',   # White marker face
    markersize=10,
    linestyle='None'           # No line
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(custom_marker)
    labels.append('Emulator')
    ax[0,0].legend(handles,labels,fontsize=28,loc='upper left')

  #fig.suptitle('Experimental Overview',fontsize=32)

  ax[1,0].set_xlabel('Year',fontsize=36)
  ax[1,1].set_xlabel('Year',fontsize=36)
  ax[0,0].set_ylabel(r'$\Delta T$ [$^\circ$C]',fontsize=36)
  ax[1,0].set_ylabel(r'$\Delta T$ [$^\circ$C]',fontsize=36)

  return

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(gamma,lw=4,c=brewer2_light(2))
ax.tick_params(axis='both', which='major', labelsize=28)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.set_xlabel(r'Position ($\mathbf{x}$)',fontsize=32)
ax.set_ylabel(r'Magnitude',fontsize=32)
ax.set_title(r'Spatial Forcing Profile ($\gamma(\mathbf{x})$)',fontsize=32)
plt.tight_layout()
plt.savefig('fig3_b.pdf',dpi=500)

def plot_2D2(T, t_mesh, x_mesh, experiments, soln_type, T_2 = None):

  if T_2 is not None:
    import copy
    T_sub = {}
    for exp in experiments:
      T_sub[exp] = T[exp] - T_2[exp]
    T = copy.copy(T_sub)

  n_exp = len(experiments)
  fig, ax = plt.subplots(2, 2, figsize=(12,12), sharey=True, sharex=True, constrained_layout=True)

  # Compute the global min and max across all experiments
  global_min = np.min([np.min(T[exp]) for exp in experiments])
  global_max = np.max([np.max(T[exp]) for exp in experiments])
  colorbars = []

  for i, exp in enumerate(experiments):
    if i == 0:
      r, c = 0, 0
    elif i == 1:
      r, c = 0, 1
    elif i == 2:
      r, c = 1, 0
    else:
      r, c = 1, 1
    if T_2 is not None:
      cf = ax[r,c].contourf(x_mesh, t_mesh, T[exp], vmin=global_min, vmax=global_max, cmap='RdBu')
    else:
      cf = ax[r,c].contourf(x_mesh, t_mesh, T[exp], vmin=global_min, vmax=global_max)

    colorbars.append(cf)
    ax[r,c].set_title(f'{experiments[i]}',fontsize=36)
    ax[r,c].tick_params(axis='both', which='major', labelsize=24)
    ax[r,c].get_xaxis().set_ticks([])
    ax[r,c].get_yaxis().set_ticks([])
  
  ax[0,0].set_ylabel(r'Time $(t)$',fontsize=36)
  ax[1,0].set_ylabel(r'Time $(t)$',fontsize=36)

  ax[1,0].set_xlabel(r'Position ($\mathbf{x}$)',fontsize=36)
  ax[1,1].set_xlabel(r'Position ($\mathbf{x}$)',fontsize=36)

  cbar = fig.colorbar(colorbars[2], ax=ax, orientation='vertical', fraction=0.075, pad=0.02)
  cbar.ax.tick_params(labelsize=28) 

  cbar.set_label(r'$\Delta T$ [$^\circ$C]',fontsize=36)

  #plt.savefig('fig3.pdf',dpi=500)
  return

from matplotlib.patches import Patch

years = np.linspace(0, 400, 400)

atmosphere = 2.5 * np.exp(-((years - 200) / 100)**2)
shallow_ocean = 1.2 * np.exp(-((years - 210) / 100)**2)
deep_ocean = 0.9 * np.exp(-((years - 300) / 140)**2)
global_avg = (atmosphere*1/6 + shallow_ocean*1/6 + deep_ocean*4/6)

fill_start = 200
fill_end_1 = 210
fill_mask_1 = (years >= fill_start) & (years <= fill_end_1)
fill_end_2 = 300
fill_mask_2 = (years >= fill_start) & (years <= fill_end_2)

fig, ax = plt.subplots(figsize=(8,8),constrained_layout=True)
ax.plot(atmosphere,lw=3,label='Atmosphere',c=utils_general.brewer2_light(0))
#ax.plot(shallow_ocean,lw=3,label='Shallow Ocean',c=utils_general.brewer2_light(1))
#ax.plot(deep_ocean,lw=3,label='Deep Ocean',c=utils_general.brewer2_light(2))
#ax.plot(global_avg,lw=3,label='Global Avg.',c=utils_general.brewer2_light(3))
ax.axvline(x=200,lw=3,ls='--',c='k',label='Peak Forcing')

#ax.fill_between(years, shallow_ocean, where=fill_mask_1, color=utils_general.brewer2_light(1), alpha=0.4)
#ax.fill_between(years, deep_ocean, where=fill_mask_2, color=utils_general.brewer2_light(2), alpha=0.4)

#gray_patch = Patch(color='gray', alpha=0.3, label='Delayed Warming')

handles, labels = ax.get_legend_handles_labels()
#handles.append(gray_patch)
#labels.append('Delayed Warming')

ax.set_ylabel(r'$\Delta T$ [$^\circ$C]',fontsize=24)
ax.set_xlabel('Year',fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend(handles=handles, labels=labels,loc='upper left',fontsize=24)

plt.savefig('warming_1.png',dpi=500)

G_1 = L_to_G(L_raw_method_2['2xCO2'],utils_coup.t)
G_2 = L_to_G(L_raw_method_2['4xCO2'],utils_coup.t)
G_3 = L_to_G(L_raw_method_2['High Emissions'],utils_coup.t)
G_4 = L_to_G(L_raw_method_2['Overshoot'],utils_coup.t)

i=0
stop = 30

t_p=np.arange(0,stop)

fig, ax = plt.subplots(figsize=(8,4), constrained_layout=True)
ax.fill_between(t_p,G_1[i].T[0:stop],G_4[i].T[0:stop],color=utils_general.brewer2_light(0),alpha=0.4)
ax.plot(G_1[i].T[0:stop],lw=3,c=utils_general.brewer2_light(0))
ax.plot(G_4[i].T[0:stop],lw=3,c=utils_general.brewer2_light(0))
ax.set_title('Atmosphere Response Function Spread from DMD',fontsize=24)
ax.set_ylabel('Magnitude',fontsize=24)
ax.set_xlabel('Year',fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=20)

plt.savefig('spread_atm.png',dpi=500)

G_2 = G_raw_method_3['2xCO2']
fig, ax = plt.subplots(figsize=(8,4), constrained_layout=True)
ax.plot(G_2[0],lw=3,c=utils_general.brewer2_light(0),label='Atmosphere')
ax.plot(G_2[1],lw=3,c=utils_general.brewer2_light(1),label='Shallow Ocean')
ax.plot(G_2[2],lw=3,c=utils_general.brewer2_light(2),label='Deep Ocean')
ax.set_title('Response Functions from Deconvolution',fontsize=24)
ax.set_ylabel('Magnitude',fontsize=24)
ax.set_xlabel('Year',fontsize=24)
ax.legend(fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)

plt.savefig('G_deconv.png',dpi=500)