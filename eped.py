import torch
import numpy as np
import copy
import os
import h5py as h5
import matplotlib.pyplot as plt

def get_ensemble_info(modelDir):
    ensemble_weight_matrices=[]
    ensemble_bias_arrays=[]
    ensemble_steepnesses=[]
    ensemble_activations=[]
    for filename in os.listdir(modelDir):
        with open(os.path.join(modelDir, filename),'r') as f:
            params={}
            for line in f.readlines()[1:]:
                (variable,data)=line.split('=')
                params[variable]=data
        num_layers=int(params['num_layers'])
        def get_arr_from_str(string, dtype=float):
            ret=[dtype(elem) for elem in string.split()]
            if dtype is str:
                ret=[key.replace("'","") for key in ret]
            return ret
        def get_2darr_from_str(string):
            arr=[]
            for line in string.split('(')[1:]:
                line=line.split(')')[0]
                arr.append([float(elem) for elem in line.split(',')])
            return arr
        layer_sizes=get_arr_from_str(params['layer_sizes'],dtype=int)
        input_names=get_arr_from_str(params['input_names'],dtype=str)
        scale_mean_in=get_arr_from_str(params['scale_mean_in'])
        scale_mean_out=get_arr_from_str(params['scale_mean_out'])
        scale_deviation_in=get_arr_from_str(params['scale_deviation_in'])
        scale_deviation_out=get_arr_from_str(params['scale_deviation_out'])
        neurons=get_2darr_from_str(params['neurons (num_inputs, activation_function, activation_steepness)'])
        connections=get_2darr_from_str(params['connections (connected_to_neuron, weight)'])
        neur=np.array(neurons)
        conn=np.array(connections)
        weight_matrices=[]
        bias_arrays=[]
        steepnesses=[]
        activations=[]
        connection_ind=0
        neuron_ind=layer_sizes[0]
        for i in range(len(layer_sizes)-1):
            layer_size=layer_sizes[i]
            # libfann documentation: "There will be a bias neuron in each layer (except the output layer),
            # and this bias neuron will be connected to all neurons in the next layer.
            # When running the network, the bias nodes always emits 1."
            next_layer_size=layer_sizes[i+1]-1
            next_connection_ind=connection_ind+next_layer_size*layer_size
            relevant_connections=conn[connection_ind:next_connection_ind,1].reshape((next_layer_size,layer_size))
            activations.append(neur[neuron_ind,1])
            steepnesses.append(neur[neuron_ind,2])
            neuron_ind+=layer_sizes[i+1]
            connection_ind=next_connection_ind
            weight_matrices.append(relevant_connections[:,:-1])
            bias_arrays.append(relevant_connections[:,-1])
        ensemble_weight_matrices.append(weight_matrices)
        ensemble_bias_arrays.append(bias_arrays)
        ensemble_activations.append(activations)
        ensemble_steepnesses.append(steepnesses)
    ensemble_info={'input_names': input_names,'scale_mean_in': scale_mean_in, 'scale_mean_out': scale_mean_out,
                   'scale_deviation_in': scale_deviation_in, 'scale_deviation_out': scale_deviation_out,
                   'ensemble_weight_matrices': ensemble_weight_matrices, 'ensemble_bias_arrays': ensemble_bias_arrays,
                   'ensemble_activations': ensemble_activations, 'ensemble_steepnesses': ensemble_steepnesses}
    return ensemble_info

def get_sig(dic,key):
    name_map={'a':'aminor_EFIT01','betan':'betan_EFIT01','bt':'bt','ip':'ip','kappa':'kappa_EFIT01','r':'rmaxis_EFIT01','neped':'neped_joe'}
    if key in name_map:
        ret=dic[name_map[key]][()]
        if key=='ip':
            ret*=1e-6
        return ret
    elif key=='delta':
        return (dic['tritop_EFIT01'][()]+dic['tribot_EFIT01'][()])/2
    # elif key=='neped':
    #     num_rho_points=dic['zipfit_edensfit_rho'].shape[-1]
    #     return dic['zipfit_edensfit_rho'][:,int(num_rho_points*0.8)]
    elif key=='zeffped':
        return 2*np.ones_like(dic['ip'][()])
    elif key=='m':
        return 2*np.ones_like(dic['ip'][()])
    else:
        return np.nan*np.ones_like(dic['ip'][()])

def evaluate_model(test_input, ensemble_info):
    norm_input=(test_input-ensemble_info['scale_mean_in'])/ensemble_info['scale_deviation_in']
    all_norm_outputs=[]
    #https://github.com/libfann/fann/blob/master/src/include/fann_data.h#L204
    def linear_activation(arr, steepness):
        return steepness*arr
    def tanh_activation(arr,steepness):
        return np.tanh(steepness*arr)
    activation_function_mapping={0: linear_activation, 5: tanh_activation}
    for which_model in range(len(ensemble_info['ensemble_weight_matrices'])):
        h=copy.deepcopy(norm_input)
        for which_layer in range(len(ensemble_info['ensemble_weight_matrices'][which_model])):
            activation_function=activation_function_mapping[ensemble_info['ensemble_activations'][which_model][which_layer]]
            h=np.matmul(ensemble_info['ensemble_weight_matrices'][which_model][which_layer],h)+ensemble_info['ensemble_bias_arrays'][which_model][which_layer]
            h=activation_function(h,ensemble_info['ensemble_steepnesses'][which_model][which_layer])
        all_norm_outputs.append(h)
    # "'OUT_p_E1_0' 'OUT_p_E1_2' 'OUT_wid_E1_0' 'OUT_wid_E1_2'\n"
    all_norm_outputs=np.array(all_norm_outputs)*ensemble_info['scale_deviation_out']+ensemble_info['scale_mean_out']
    return np.mean(all_norm_outputs,axis=0), np.std(all_norm_outputs,axis=0)

modelDir='multiroot'
ensemble_info=get_ensemble_info(modelDir)
filename='aug_data.h5' #aug_data
#test_densities=[1,3,5,7]
with h5.File(filename,'a') as f:
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
                ensemble_mean,ensemble_std=evaluate_model(test_input,ensemble_info)
                ensemble_means[input_ind]=ensemble_mean[0]
            if 'epedHeight' in f[shot]:
                del f[shot]['epedHeight']
            f[shot]['epedHeight']=ensemble_means
            if 'eped_te_prediction' in f[shot]:
                del f[shot]['eped_te_prediction']
            # unit cnonversion from OMFIT's EPED module scripts
            # *1e3, /1.6e-19, *1e19, /2 (the 2 is for electron/ion split I think)
            f[shot]['eped_te_prediction']=f[shot]['epedHeight'][:] / f[shot]['neped_joe'][:] *1e3/1.6/2

            # ensemble_means=np.zeros((len(inputs),len(test_densities)))
            # density_ind=ensemble_info['input_names'].index('neped')
            # for test_density_ind, test_density in enumerate(test_densities):
            #     for input_ind, test_input in enumerate(inputs):
            #         test_input[density_ind]=test_density
            #         ensemble_mean,ensemble_std=evaluate_model(test_input,ensemble_info)
            #         ensemble_means[input_ind,test_density_ind]=ensemble_mean[0]
            # for test_density_ind, test_density in enumerate(test_densities):
            #     if f'epedHeightForNe{test_density}' in f[shot]:
            #         del f[shot][f'epedHeightForNe{test_density}']
            #     f[shot][f'epedHeightForNe{test_density}']=ensemble_means[:,test_density_ind]
        else:
            print(f'{shot} eped failed')

# if True:
#     density_ind=-3
#     test_densities=np.arange(0,17,1)
#     test_input=np.array([0.552,1.278,1.692,0.5417,1.3,1.855,2,3.62,1.7,2.073])
#     for density in test_densities:
#         test_input[density_ind]=density
#         ensemble_mean,ensemble_std=evaluate_model(test_input,ensemble_info)
#         ensemble_means.append(ensemble_mean)
#         ensemble_stds.append(ensemble_std)
#     title="density scan (shot 153523.3745, the OMFIT example)"
#     xlabel="neped"
#     xaxis=test_densities
# else:
#     filename='../PlasmaEvolution/test.h5'
#     inputs=[]
#     with h5.File(filename,'r') as f:
#         shot='153523'
#         for input_name in ensemble_info['input_names']:
#             inputs.append(get_sig(f[shot],input_name))
#     inputs=np.array(inputs).T
#     for test_input in inputs:
#         ensemble_mean,ensemble_std=evaluate_model(test_input,ensemble_info)
#         ensemble_means.append(ensemble_mean)
#         ensemble_stds.append(ensemble_std)
#     title=f"shot {shot}"
#     xlabel="time (s)"
#     xaxis=np.arange(len(inputs))*0.025
# ensemble_means=np.array(ensemble_means).T
# ensemble_stds=np.array(ensemble_stds).T
# if False:
#     fig,axes=plt.subplots(2,sharex=True)
#     def plot_this(plot_ind,output_ind,label,color):
#         avg=ensemble_means[output_ind]
#         std=ensemble_stds[output_ind]
#         axes[plot_ind].plot(xaxis,avg,label=label,c=color)
#         axes[plot_ind].fill_between(xaxis,avg-std,avg+std,color=color,alpha=0.2)
#     axes[0].set_title('p_ped_height')
#     axes[1].set_title('p_ped_width')
#     plot_this(0,0,'H','r')
#     plot_this(0,1,'superH','b')
#     plot_this(1,2,'H','r')
#     plot_this(1,3,'superH','b')
#     axes[0].legend()
#     axes[0].set_ylim((0,None))
#     axes[1].set_ylim((0,None))
#     fig.suptitle(title)
#     axes[-1].set_xlabel(xlabel)
#     plt.show()
# axes[0].plot(test_densities,ensemble_outputs[:,:,1]);
# axes[1].plot(test_densities,ensemble_outputs[:,:,2]);
# axes[1].plot(test_densities,ensemble_outputs[:,:,3]);
# denormalize output

# class brainfuse(torch.nn.Module):
#     def __init__(self, parameters):
#         super().__init__()
#         self.

#     def forward(self, inputs):
#         self.DB
