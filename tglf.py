# TODO:
#    - convert q and pressure to rho basis (currently psi)
#    - add Zeff
#    - add mass
#    - add shafranov shift
#    - add 1D kappa
#    - add 1D triangularity
#    - add 1D R
#    - add 1D a
#    - add 1D rho (unnormed) for BUNIT

from scipy.io import netcdf
import numpy as np
import os
import h5py as h5
import matplotlib.pyplot as plt
import orso_nn_helpers

# CGS units
# note 1 is electrons, 2 is D, 3 is C
# densities by 1e13 to get them to #/cm^3, temp by 1e3 to get to eV
# k0 to scale eV to erg
# 1e4 to scale T to G
modelDir='DIIID_ion_stiffness_60_rotation'
ensemble_info=orso_nn_helpers.get_ensemble_info(modelDir)
filename='test.h5' #'test.h5'

tglf_inputs=['VEXB_SHEAR', 'XNUE', 'VPAR_1', 'VPAR_SHEAR_1', 'BETAE', 'ZEFF', 'DRMAJDX_LOC','TAUS_2','RMIN_LOC','RMAJ_LOC',
             'AS_2', 'AS_3', 'Q_LOC', 'Q_PRIME_LOC', 'P_PRIME_LOC', 'DELTA_LOC', 'KAPPA_LOC',
             'RLNS_1','RLNS_2','RLNS_3', 'RLTS_1','RLTS_2', 'S_KAPPA_LOC']
#test_densities=[1,3,5,7]
with h5.File(filename,'a') as f:
    database_times=f['times'][:]
    rho_arr=f['spatial_coordinates'][:]
    drho=1./(len(rho_arr)-1)
    def get_derivative(arr,rho_ind, offset=1):
        left_offset=min(offset,rho_ind)
        right_offset=min(offset,arr.shape[1]-1-rho_ind)
        return (arr[:,rho_ind+right_offset]-arr[:,rho_ind-left_offset])/((left_offset+right_offset)*drho)
    k0=1.6e-12 #erg/ev
    mp=1.67e-24 #proton mass (g)

    shot='174911'
    dic=f[shot]
    # for shot
    m0=mp*2 # in future replace 2 w/ mass based on gas used
    ip_sign=np.sign(dic['ip'][()])
    kappa_0=(dic['kappa_EFIT01'][:]+1)*0.5
    kappa_1d=np.outer((dic['kappa_EFIT01'][:]-kappa_0),rho_arr**2)+kappa_0[:,np.newaxis] # very rough approximation, should adapt
    trian_1d=(dic['tritop_EFIT01'][()]+dic['tribot_EFIT01'][()])/2
    aminor=1e2*dic['aminor_EFIT01'][()]
    etemp=1e3*dic['zipfit_etempfit_rho'][()]
    itemp=1e3*dic['zipfit_itempfit_rho'][()]
    edens=1e13*dic['zipfit_edensfit_rho'][()]
    Zimp=6
    impdens=0.02*edens #replace w/ measured impurity density from CER later
    idens=edens-Zimp*impdens
    pres=edens*etemp+(idens+impdens)*itemp #+pfast+0.5*(pblon+pbper)

    input_dic={key: [] for key in tglf_inputs}
    for rho_ind in range(33): #(1,33-1):
        rho=rho_arr[rho_ind]
        TI=itemp[:,rho_ind]
        cs0=np.sqrt(k0*etemp[:,rho_ind]/m0)
        tria=rho**2*trian_1d
        a0=rho*aminor # length scale in cm
        R0=1e2*dic['rmaxis_EFIT01'][()]
        shift_constant=0.3 #see Wesson's Tokamaks book, Shafranov shift chapter
        shift=shift_constant*aminor**2/R0 #shafranov shift
        R=R0-shift*np.square(rho)
        dRdx=-2*rho*shift/aminor # analytically wrote out derivative of the above by eye
        kappa=kappa_1d[:,rho_ind]
        # see Waltz Miller 1999 ITG simulations
        # BUNIT = B0 * (rho drho) / (r dr)~ btor * kappa, since rho~a*sqrt(kappa)
        BUNIT=1e4*dic['bt'][()]*kappa
        q=dic['qpsi_EFIT01'][:,rho_ind]

        # Bmod=sqrt(Btor**2+Bpol**2)
        #     Bmod=Btor sqrt(1+(Btor/Bpol)**2)
        #     Bmod=Btor sqrt(1+(a/qR)**2)
        # Er =
        # vexb2 = -Er / Bmod
        # 1e2 vexb2 / cs0

        lnlambda=24-0.5*np.log(idens[:,rho_ind])+np.log(itemp[:,rho_ind])
        taue=3.44e5*(itemp[:,rho_ind])**1.5 / (idens[:,rho_ind]*lnlambda)
        input_dic['XNUE'].append(0.75*np.sqrt(np.pi)*aminor/(taue*cs0))

        par_component=1/np.sqrt(1+np.square(a0/(q*R)))
        drot=get_derivative(dic['zipfit_trotfit_rho'],rho_ind)
        input_dic['VPAR_1'].append(ip_sign*par_component*R*1e3*dic['zipfit_trotfit_rho'][:,rho_ind] / cs0)
        input_dic['VPAR_SHEAR_1'].append(-ip_sign*par_component*R*1e3*drot/cs0)

        perp_component=1/np.sqrt(1+np.square(q*R/a0))
        dvexb=1e3*R*perp_component*drot
        input_dic['VEXB_SHEAR'].append(-ip_sign*dvexb/cs0)
        #input_dic['VEXB_SHEAR'].append(-ip_sign*a0*dvexb/cs0/q)

        input_dic['RLNS_1'].append(-get_derivative(edens,rho_ind)/edens[:,rho_ind])
        input_dic['RLNS_2'].append(-get_derivative(idens,rho_ind)/idens[:,rho_ind])
        input_dic['RLNS_3'].append(-get_derivative(impdens,rho_ind)/impdens[:,rho_ind])
        input_dic['RLTS_1'].append(-get_derivative(etemp,rho_ind)/etemp[:,rho_ind])
        input_dic['RLTS_2'].append(-get_derivative(itemp,rho_ind)/itemp[:,rho_ind])

        input_dic['S_KAPPA_LOC'].append(rho*get_derivative(kappa_1d,rho_ind)/kappa)
        input_dic['BETAE'].append(8*np.pi*k0*edens[:,rho_ind]*etemp[:,rho_ind]/BUNIT**2)
        input_dic['ZEFF'].append(2*np.ones_like(dic['ip'][()]))
        input_dic['DRMAJDX_LOC'].append(dRdx)
        input_dic['TAUS_2'].append(itemp[:,rho_ind]/etemp[:,rho_ind])
        input_dic['RMIN_LOC'].append(a0/aminor)
        input_dic['RMAJ_LOC'].append(R/aminor)
        input_dic['AS_2'].append(idens[:,rho_ind]/edens[:,rho_ind])
        input_dic['AS_3'].append(impdens[:,rho_ind]/edens[:,rho_ind])
        input_dic['Q_LOC'].append(q)
        #input_dic['Q_PRIME_LOC'].append(q**2*aminor/(a0**2)*get_derivative(dic['qpsi_EFIT01'],rho_ind))
        input_dic['Q_PRIME_LOC'].append(q*get_derivative(dic['qpsi_EFIT01'],rho_ind))
        #input_dic['P_PRIME_LOC'].append(q*aminor**2/(a0*BUNIT**2)*get_derivative(dic['pres_EFIT01'],rho_ind))
        input_dic['P_PRIME_LOC'].append(k0/BUNIT**2 * q/rho * get_derivative(pres,rho_ind))
        input_dic['DELTA_LOC'].append(tria)
        input_dic['KAPPA_LOC'].append(kappa)

for key in tglf_inputs:
    input_dic[key]=np.array(input_dic[key]).T
    #print(f'{key}: {input_dic[key][30]}')

# testing (temporary)
import matplotlib.pyplot as plt
import os
# equ is tglfTest, ran 5.75 to 5.81
dirname='../../Downloads/tglf_inputs_174911_5800/'
def add_signal(dic,key,value):
    if key not in dic:
        dic[key]=[]
    else:
        dic[key].append(value)
def make_float(value):
    try:
        return float(value)
    except:
        return value
astra_dic={}
for i in range(1,51):
    with open(os.path.join(dirname,f'input.tglf_{i}'), 'r') as f:
        for line in f:
            candidate=line.split('=')
            if len(candidate)==2:
                add_signal(astra_dic,candidate[0].strip(),make_float(candidate[1]))
time_ind=232

cdfFilename='../../Downloads/174911Q95INPUTtglfTest.CDF'
with netcdf.netcdf_file(cdfFilename) as f:
    time_ind=-1
    #f.variables['TIME'].data
    def get_sig(sig):
        return f.variables[sig].data[time_ind]
    # in keV * 10^19/m^2 / s
    QE_tglf=get_sig('HE') * -np.diff(get_sig('TE'),append=0)/np.diff(get_sig('RHO'),append=get_sig('RHO')[-1]) * get_sig('G11')*get_sig('NE')
    QI_tglf=get_sig('XI') * -np.diff(get_sig('TI'),append=0)/np.diff(get_sig('RHO'),append=get_sig('RHO')[-1]) * get_sig('G11')*get_sig('NI')
    k0=1.6e-12 #erg/ev
    mp=1.67e-24 #proton mass (g)
    e=4.8e-10 #statcoulomb
    m0=mp*2
    cs=np.sqrt(k0*1e3*get_sig('TE')/m0)
    drhodr=np.diff(get_sig('RHO'),prepend=0)/np.diff(get_sig('AMETR'),prepend=0)
    BUNIT=1e4*get_sig('BTOR')*get_sig('ELON') #**drhodr*get_sig('RHO')/get_sig('AMETR')
    c=3e10 #speed of light, cm/s
    omega=e*BUNIT/(m0*c)
    rhoGyro=cs/omega
    rhoStar=rhoGyro/(1e2*get_sig('AMETR'))
    # in keV * 10^19/m^2 / s
    Q_gyrobohm=get_sig('NE') * get_sig('TE') * 1e-2*cs * rhoStar**2

if False:
    plotted_sigs=tglf_inputs #['VPAR_1','VPAR_SHEAR_1','VEXB_SHEAR']
    # needs rho basis(?): ['Q_LOC','Q_PRIME_LOC']
    # bad: ['BETAE','VEXB_SHEAR', 'VPAR_1','VPAR_SHEAR_1']
    # need kappa: ['BETAE','KAPPA_LOC', 'S_KAPPA_LOC']
    # note BETAE would do even better w/ full rho
    # need triang: ['DELTA_LOC']
    # needs shafranov shift: ['DRMAJDX_LOC','RMAJ_LOC']
    # bad (understandably): [, 'DRMAJDX_LOC','RMAJ_LOC']
    # good: ['RLTS_1', 'RLTS_2', 'RLNS_1', 'TAUS_2']
    # needs ZEFF: ['XNUE','RLNS_2','RLNS_3','AS_2','AS_3']
    limits={sig: (None, None) for sig in plotted_sigs}
    limits['XNUE']=(0,2)
    #fig,axes=plt.subplots(len(plotted_sigs),sharex=True)
    #axes=np.atleast_1d(axes)
    fig,axes=plt.subplots(6,4,sharex=True)
    axes=np.ndarray.flatten(axes)
    for sig_ind,sig in enumerate(plotted_sigs):
        axes[sig_ind].plot(rho_arr,input_dic[sig][time_ind], label='me')
        axes[sig_ind].plot(np.linspace(0,1,49),astra_dic[sig], label='astra')
        axes[sig_ind].set_ylabel(sig)
        axes[sig_ind].set_ylim(limits[sig])
    axes[0].legend()
    plt.show()

rho_inds=range(49)#np.arange(1,49)
num_outputs=4
ensemble_means=np.zeros((len(rho_inds),num_outputs))
for rho_ind in rho_inds:
    inputs=[]
    for sig in ensemble_info['input_names']:
        inputs.append(astra_dic[sig][rho_ind])
    inputs=np.array(inputs)
    ensemble_mean,ensemble_std=orso_nn_helpers.evaluate_model(inputs,ensemble_info)
    ensemble_means[rho_ind,:]=ensemble_mean
fig,axes=plt.subplots(num_outputs+3,sharex=True)
output_labels=[r'$Q_e$',r'$Q_i$',r'$\Gamma_e$',r'$\Pi_i$']
for i in range(num_outputs):
    axes[i].plot(np.linspace(0,1,len(ensemble_means)),ensemble_means[:,i],label='tglf (nn)')
    axes[i].set_ylabel(output_labels[i])
axes[-2].plot(np.linspace(0,1,len(QE_tglf)),QE_tglf,label='tglf (astra)')
axes[-1].plot(np.linspace(0,1,len(QI_tglf)),QI_tglf)
# axes[-3].plot(np.linspace(0,1,len(Q_tglf)),Q_tglf)
# axes[-3].set_ylabel('Q_tglf')
# axes[-2].plot(np.linspace(0,1,len(Q_tglf)),Q_gyrobohm)
# axes[-2].set_ylabel('Q_Gyrobohm')
# axes[-1].plot(np.linspace(0,1,len(Q_tglf)),Q_tglf/Q_gyrobohm)
# axes[-1].set_ylabel('Q_tglf/Q_Gyrobohm')
axes[0].legend()
plt.show()

if False:
    shots = list(f.keys())
    shots.remove('times')
    shots.remove('spatial_coordinates')
    for shot in shots:
        # add neped estimate
        if 'zipfit_edensfit_rho' in f[shot]:
            if 'neped_joe' in f[shot]:
                del f[shot]['neped_joe']
            rho_ind=26
            f[shot]['neped_joe']=f[shot]['zipfit_edensfit_rho'][:,rho_ind]
        ensemble_means=[]
        inputs=[]
        try:
            for input_name in ensemble_info['input_names']:
                inputs.append(get_sig(f[shot],input_name))
            all_sigs_available=True
        except:
            all_sigs_available=False
        if all_sigs_available:
            inputs=np.array(inputs).T
            ensemble_means=np.zeros(len(inputs))
            for input_ind, test_input in enumerate(inputs):
                ensemble_mean,ensemble_std=orso_nn_helpers.evaluate_model(test_input,ensemble_info)
                ensemble_means[input_ind]=ensemble_mean[0]
            if 'epedHeight' in f[shot]:
                del f[shot]['epedHeight']
            f[shot]['epedHeight']=ensemble_means
            if 'eped_te_prediction' in f[shot]:
                del f[shot]['eped_te_prediction']
            # unit cnonversion from OMFIT's EPED module scripts
            # *1e3, /1.6e-19, *1e19, /2 (the 2 is for electron/ion split I think)
            f[shot]['eped_te_prediction']=f[shot]['epedHeight'][:] / f[shot]['neped_joe'][:] *1e3/1.6/2

        else:
            print(f'{shot} eped failed')
