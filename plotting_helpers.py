import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import dataSettings
import numpy as np

label_map={'zipfit_etempfit_rho': r'$T_e$',
           'zipfit_itempfit_rho': r'$T_i$',
           'zipfit_edensfit_rho': r'$n_e$',
           'zipfit_trotfit_rho': r'$\Omega$',
           'zeff_rho': r'$Z_{eff}$',
           'TE_astrainterpretive': r'$T_e$',
           'TI_astrainterpretive': r'$T_i$',
           'NE_astrainterpretive': r'$n_e$',
           'NI_astrainterpretive': r'$n_i$',
           'ANGF_astrainterpretive': r'$\Omega$',
           'qpsi_EFIT01': r'$q$',
           'pinj': r'$P_{NBI}$',
           'tinj': r'$T_{NBI} (N m)$',
           'ech_pwr_total': r'$P_{ECH}$',
           'P_AUXILIARY': r'$P_{NBI}+P_{ECH}$',
           'ip': r'$I_p$',
           'bt': r'$B_t$',
           'Ar_tot': 'Ar',
           'D_tot': 'D',
           'H_tot': 'H',
           'He_tot': 'He',
           'N_tot': 'N',
           'Ne_tot': 'Ne',
           'li_EFIT01': 'li',
           'aminor_EFIT01': r'$a$',
           'rmaxis_EFIT01': r'$R$',
           'tribot_EFIT01': r'$\delta_l$',
           'tritop_EFIT01': r'$\delta_u$',
           'kappa_EFIT01': r'$\kappa$',
           'volume_EFIT01': 'V',
           'ipsiptargt': r'$I_p^{target}$',
           'dssdenest': r'$<n_e>$'}

def modelRollout_plot(predicted_means, predicted_stds, predicted_times,
                      denormalized_true_dic, true_times,
                      plotted_profiles, plotted_parameters, plotted_actuators,
                      rho_ind, title):
    plot_filename=f'{title}.svg'
    NSTEPS_PLOTTED=2
    num_columns = 3
    if len(plotted_parameters)>0:
        num_columns = 4
    fig,axes=plt.subplots(max(len(plotted_profiles),len(plotted_parameters),len(plotted_actuators)),num_columns, sharex='col', figsize=(8,5))
    plt.subplots_adjust(hspace=0, wspace=1)
    colors=cm.viridis(np.linspace(0,1,NSTEPS_PLOTTED+1))
    nwarmup=len(true_times)-len(predicted_times)
    # get evenly spaced predictde points
    time_inds_predicted_for_nsteps=np.array([int(t) for t in np.linspace(int(len(predicted_times)/NSTEPS_PLOTTED),
                                                                         len(predicted_times)-1,
                                                                         NSTEPS_PLOTTED,
                                                                         endpoint=True)])
    x=np.linspace(0,1,dataSettings.nx)
    #plotted_profiles[0], plotted_profiles[3] = plotted_profiles[3], plotted_profiles[0]
    #plotted_actuators.insert(0, plotted_actuators.pop())
    for i,profile in enumerate(plotted_profiles):
        axes[i,1].errorbar(predicted_times, predicted_means[profile][:,rho_ind], yerr=predicted_stds[profile][:,rho_ind],
                           label='predicted', c='k', alpha=0.1)
        axes[i,1].plot(true_times, denormalized_true_dic[profile][:,rho_ind],
                       label='real', c='k', linestyle='--')
        axes[i,1].set_ylabel(label_map[profile])
        # plot the initial experimental step always
        axes[i,2].plot(x, denormalized_true_dic[profile][nwarmup],
                       alpha=0.5, c='k', label=f'{true_times[nwarmup]}ms')
        for step_ind, time_ind in enumerate(time_inds_predicted_for_nsteps):
            axes[i,2].plot(x, predicted_means[profile][time_ind], c=colors[step_ind],
                           label=f'{predicted_times[time_ind]}ms')
            axes[i,2].plot(x, denormalized_true_dic[profile][time_ind+nwarmup],
                           linestyle='--', c=colors[step_ind])
    for i,actuator in enumerate(plotted_actuators):
        axes[i,0].plot(true_times, denormalized_true_dic[actuator]
                       label='real', c='k', linestyle='--')
        axes[i,0].set_ylabel(label_map[actuator])
    if len(plotted_parameters)>0:
        for i,parameter in enumerate(plotted_parameters):
            axes[i,3].errorbar(predicted_times, predicted_means[parameter][:, 0], yerr=predicted_stds[parameter][:, 0],
                           label='predicted', c='k', alpha=0.1)
            axes[i,3].plot(true_times, denormalized_true_dic[parameter],
                           label='real', c='k', linestyle='--')
            axes[i,3].set_ylabel(label_map[parameter])
    axes[0, 0].text(0.5, 1.05, 'Actuators over time', transform=axes[0, 0].transAxes, fontsize=10, ha='center')
    axes[0, 1].text(0.5, 1.05, 'Predictions over time', transform=axes[0, 1].transAxes, fontsize=10, ha='center')
    axes[0, 2].text(0.5, 1.05, 'Predicted profiles', transform=axes[0, 2].transAxes, fontsize=10, ha='center')
    axes[0, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
    axes[1, 2].legend(fontsize=6)
    fig.suptitle(f'{title}')
    plt.savefig(plot_filename)
    plt.show()
