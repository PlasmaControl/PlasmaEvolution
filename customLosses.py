import torch
import dataSettings

def calculate_taue(W, dWdt, P):
    # W in J, P in W, dWdt in W
    return torch.clamp(W / (P - dWdt),
                       min=dataSettings.taueMin,
                       max=dataSettings.taueMax)

# get stored energy in J from profile info and volume
def calculate_W(etemp, itemp, edens, volume):
    # etemp, itemp, edens, and volume normalized values coming in
    etemp=dataSettings.denormalize(etemp, 'zipfit_etempfit_rho')
    itemp=dataSettings.denormalize(itemp, 'zipfit_itempfit_rho')
    edens=dataSettings.denormalize(edens, 'zipfit_edensfit_rho')
    volume=dataSettings.denormalize(volume, 'volume_EFIT01')
    # assume in dataset etemp/itemp keV, edens 1e19 m^-3
    energy_density=3./2 * (etemp + (1-dataSettings.IMPURITY_Z*dataSettings.IMPURITY_FRACTION)*itemp)*edens
    energy_density*=dataSettings.KEV_PER_1019_TO_J
    # vol ~ (rho/rho_lim)^2 Vol
    # dVol ~ 2 (rho/rho_lim) Vol d(rho/rho_lim)
    # (exact for low aspect ratio, low-pressure)
    rhon=torch.linspace(0,1,dataSettings.nx)
    # note this is an average diff, not np.diff (which would be 1/(nx-1) )
    drhon=1./dataSettings.nx
    dVol=2*torch.outer(volume,rhon)*drhon
    return torch.sum(energy_density*dVol,axis=-1)

class myMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, profiles_predicted_tensor,
                profiles_tensor, actuators_tensor, parameters_tensor,
                profiles, actuators, parameters):
        lookahead=actuators_tensor.shape[1]-parameters_tensor.shape[1] #present timestep -lookahead-1
        return torch.nn.MSELoss(reduction='mean')(profiles_predicted_tensor,
                                                  profiles_tensor[:,-lookahead:,:,:])

class simpleMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, profiles_predicted_tensor, profiles_tensor, *args):
        return torch.nn.MSELoss(reduction='mean')(profiles_predicted_tensor,
                                                  profiles_tensor)

class combinedLoss(torch.nn.Module):
    def __init__(self, energyWeight=0):
        super().__init__()
        self.energyWeight=energyWeight
    def forward(self, predicted_profiles_tensor,
                profiles_tensor, actuators_tensor, parameters_tensor,
                profiles, actuators, parameters):
        lookahead=actuators_tensor.shape[1]-parameters_tensor.shape[1] #present timestep -lookahead-1
        batch_size=len(predicted_profiles_tensor)
        total_loss=torch.mean((predicted_profiles_tensor-profiles_tensor[:,-lookahead:,:,:])**2)

        etemp_ind=profiles.index('zipfit_etempfit_rho')
        itemp_ind=profiles.index('zipfit_itempfit_rho')
        edens_ind=profiles.index('zipfit_edensfit_rho')
        volume_ind=parameters.index('volume_EFIT01')
        pinj_ind=actuators.index('pinj')
        W_now=calculate_W(profiles_tensor[:,-lookahead-1,etemp_ind,:],
                          profiles_tensor[:,-lookahead-1,itemp_ind,:],
                          profiles_tensor[:,-lookahead-1,edens_ind,:],
                          parameters_tensor[:,-1,volume_ind])
        W_prev=calculate_W(profiles_tensor[:,-lookahead-2,etemp_ind,:],
                           profiles_tensor[:,-lookahead-2,itemp_ind,:],
                           profiles_tensor[:,-lookahead-2,edens_ind,:],
                           parameters_tensor[:,-2,volume_ind])
        dWdt=(W_now-W_prev)/dataSettings.DT
        # add small epsilon of ohmic power, mostly so no-beam cases still converge ok
        P_rollout=1.e3*dataSettings.denormalize(actuators_tensor[:,-lookahead-1:,pinj_ind],'pinj') + dataSettings.ohmicPower
        #(actuators_tensor[:,-lookahead:,pinj_ind]+actuators_tensor[:,-lookahead-1:-1,pinj_ind])/2.
        W=W_now #(W_now+W_prev)/2
        taue_now=calculate_taue(W, dWdt, P_rollout[:,0])
        W_rollout=torch.empty(batch_size, lookahead)
        for time_ind in range(0,lookahead):
            # use most recent volume, in future if predict volume could add it here
            W_rollout[:,time_ind] = calculate_W(predicted_profiles_tensor[:,time_ind,etemp_ind,:],
                                                predicted_profiles_tensor[:,time_ind,itemp_ind,:],
                                                predicted_profiles_tensor[:,time_ind,edens_ind,:],
                                                parameters_tensor[:,-1,volume_ind])
        dWdtRollout=torch.empty(batch_size, lookahead)
        dWdtRollout[:,0] = W_rollout[:,0] - W_now
        dWdtRollout[:,1:]=W_rollout[:,1:] - W_rollout[:,:-1]
        dWdtRollout /= dataSettings.DT
        energyError = dWdtRollout - (-W_rollout/taue_now.expand(lookahead,batch_size).T + P_rollout[:,1:])
        energyError /= 1.e6 # in MW
        total_loss+=self.energyWeight*torch.mean(torch.square(energyError))

        return total_loss
