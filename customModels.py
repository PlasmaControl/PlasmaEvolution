from dataSettings import nx
import copy
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class IanMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=100):
        super().__init__()
        self.mlp=torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, padded_input):
        return self.mlp(padded_input)

class IanGRU(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=100):
        super().__init__()
        self.encoder=torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            #torch.nn.ReLU()
        )
        # batch_size x time_length x input_dim
        self.rnn=torch.nn.LSTM(
            hidden_dim, hidden_dim,
            batch_first=True
        )
        self.decoder=torch.nn.Sequential(
            #torch.nn.Linear(hidden_dim, hidden_dim),
            #torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, padded_input): #, padded_lens):
        embedding=self.encoder(padded_input)
        embedding_evolved,_=self.rnn(embedding)
        # this doesn't work with data parallelism and is annoying with just computation
        #    time as a benefit, skipping for now
        #total_length = embedding.size(1) # get the max sequence length...
        #packed_embedding=pack_padded_sequence(embedding, padded_lens, batch_first=True)
        #packed_embedding_evolved,_=self.rnn(packed_embedding)
        #embedding_evolved,_=pad_packed_sequence(packed_embedding_evolved, batch_first=True)
                                                #total_length=total_length) # ...and put it here
        padded_output=self.decoder(embedding_evolved)
        return padded_output

# simple mapping, given just actuators over time try to predict profiles
# I imagine lookback=0 is most sensible
class ProfilesFromActuatorsAdvanced(torch.nn.Module):
    def __init__(self, profiles, actuators):
        super().__init__()
        self.nprofiles = len(profiles)
        self.nactuators = len(actuators)
        self.c = torch.nn.Parameter(torch.randn((1,self.nprofiles*nx,self.nactuators),
                                                requires_grad=True, dtype=torch.float))
    def forward(self, input_profiles, input_actuators, input_parameters):
        # Computes the outputs / predictions
        # batch_size, nx*len(profiles), lookahead
        this_batch_size=input_actuators.shape[0]
        pseudo_profiles_over_time = torch.bmm(self.c.repeat(this_batch_size,1,1),
                                              input_actuators)
        # sum over the lookahead
        pseudo_profiles = pseudo_profiles_over_time.sum(dim=-1).reshape(this_batch_size,self.nprofiles,nx)
        return pseudo_profiles

# reproducing 2021 paper
# dataset should be built w/ just last timestep as output
class PlasmaConv2D(torch.nn.Module):
    def __init__(self, profiles, actuators, parameters):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(len(profiles),10,2,padding='same'),
            torch.nn.Conv1d(10,20,4,padding='same'),
            torch.nn.Conv1d(20,40,6,padding='same'),
            torch.nn.Conv1d(40,80,8,padding='same'),
            torch.nn.ReLU()
        )
        self.actuatorPreRNN = torch.nn.Sequential(
            torch.nn.Linear(len(actuators),10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,20),
            torch.nn.ReLU(),
            torch.nn.Linear(20,40),
            torch.nn.ReLU()
        )
        self.parameterPreRNN = torch.nn.Sequential(
            torch.nn.Linear(len(parameters),10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,20),
            torch.nn.ReLU(),
            torch.nn.Linear(20,40),
            torch.nn.ReLU()
        )
        # remember we'll just take the latest output
        self.actuatorRNN = torch.nn.LSTM(40,80,batch_first=True)
        self.parameterRNN = copy.deepcopy(self.actuatorRNN)
        self.actuatorPostRNN = torch.nn.Sequential(
            torch.nn.Linear(1,8),
            torch.nn.ReLU(),
            torch.nn.Linear(8,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,33),
            torch.nn.ReLU()
        )
        self.parameterPostRNN = copy.deepcopy(self.actuatorPostRNN)
        self.deconv = torch.nn.Sequential(
            torch.nn.Conv1d(80,40,8,padding='same'),
            torch.nn.Conv1d(40,20,4,padding='same'),
            torch.nn.Conv1d(20,10,4,padding='same'),
            torch.nn.Conv1d(10,len(profiles),2,padding='same'),
            torch.nn.ReLU(),
        )
    def forward(self, profiles_tensor, input_actuators, input_parameters):
        lookahead=input_actuators.shape[1]-input_parameters.shape[1] #present timestep -lookahead-1
        present_profiles=profiles_tensor[:,-lookahead-1,:,:]

        preAddProfiles=self.conv(present_profiles) #input_profiles)
        preAddActuators=self.actuatorPreRNN(input_actuators)
        _, (preAddActuators, _)=self.actuatorRNN(preAddActuators)
        preAddActuators=preAddActuators.permute([1,2,0])
        preAddActuators=self.actuatorPostRNN(preAddActuators)
        preAddParameters=self.parameterPreRNN(input_parameters)
        _, (preAddParameters, _)=self.parameterRNN(preAddParameters)
        preAddParameters=preAddParameters.permute([1,2,0])
        preAddParameters=self.parameterPostRNN(preAddParameters)
        pseudoProfiles=preAddProfiles+preAddActuators+preAddParameters
        outputProfiles=self.deconv(pseudoProfiles)
        return outputProfiles

# simplest RNN possible
# dataset should be built with all timesteps output
class PlasmaGRU(torch.nn.Module):
    def __init__(self, profiles, actuators, parameters):
        super().__init__()
        self.nprofiles=len(profiles)
        self.recurrent = torch.nn.GRU(len(actuators),len(profiles)*nx,batch_first=True)
    def forward(self, profiles_tensor, actuators_tensor, parameters_tensor):
        lookahead=actuators_tensor.shape[1]-parameters_tensor.shape[1] #present timestep -lookahead-1
        present_profile=profiles_tensor[:,-lookahead-1,:,:]
        hiddenProfiles,_=self.recurrent(actuators_tensor[:,-lookahead:,:],
                                        torch.flatten(present_profile,start_dim=1)[None,:])
        outputProfiles=hiddenProfiles.reshape(*hiddenProfiles.shape[:-1],self.nprofiles,nx)
        return outputProfiles

class ProfilesFromActuators(torch.nn.Module):
    def __init__(self, profiles, actuators, nProfilePoints, hidden_size=30):
        super().__init__()
        self.mlp= torch.nn.Sequential(
            torch.nn.Linear(len(actuators), hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, len(profiles)*nProfilePoints)
        )
    def forward(self, profiles_tensor, actuators_tensor):
        return self.mlp(actuators_tensor)
