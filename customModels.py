from dataSettings import nx
import copy
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class IanMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim,
                 hidden_dim=100, extra_layers=1):
        super().__init__()
        self.mlp=torch.nn.Sequential()
        self.mlp.append(torch.nn.Linear(input_dim, hidden_dim))
        self.mlp.append(torch.nn.ReLU())
        for i in range(extra_layers):
            self.mlp.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(torch.nn.ReLU())
        self.mlp.append(torch.nn.Linear(hidden_dim, output_dim))
    def forward(self, padded_input):
        return self.mlp(padded_input)

class IanRNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim,
                 encoder_dim=100, encoder_extra_layers=1,
                 rnn_dim=100,
                 decoder_dim=100, decoder_extra_layers=1
                 ):
        super().__init__()
        self.encoder = torch.nn.Sequential()
        self.encoder.append(torch.nn.Linear(input_dim, encoder_dim))
        self.encoder.append(torch.nn.ReLU())
        for i in range(encoder_extra_layers):
            self.encoder.append(torch.nn.Linear(encoder_dim, encoder_dim))
            self.encoder.append(torch.nn.ReLU())
        # batch_size x time_length x input_dim
        self.rnn=torch.nn.LSTM(
            encoder_dim, rnn_dim,
            batch_first=True
        )
        self.decoder = torch.nn.Sequential()
        self.decoder.append(torch.nn.Linear(rnn_dim, decoder_dim))
        self.decoder.append(torch.nn.ReLU())
        for i in range(decoder_extra_layers):
            self.decoder.append(torch.nn.Linear(decoder_dim, decoder_dim))
            self.decoder.append(torch.nn.ReLU())
        self.decoder.append(torch.nn.Linear(decoder_dim, output_dim))
    def forward(self, padded_input):
        embedding=self.encoder(padded_input)
        embedding_evolved,_=self.rnn(embedding)
        padded_output=self.decoder(embedding_evolved)
        return padded_output

class InverseLeakyReLU(torch.nn.Module):
    def __init__(self, slope=0.01):
        super(InverseLeakyReLU, self).__init__()
        self.slope = slope

    def forward(self, x):
        return torch.where(x < 0, x / self.slope, x)

class HiroLinear(torch.nn.Module):
    def __init__(self, input_dim, output_dim,
                 encoder_extra_layers=1,
                 decoder_extra_layers=1
                 ):
        super().__init__()


        self.input_dim = input_dim
        self.output_dim = output_dim
        state_dim = output_dim

        self.encoder = torch.nn.Sequential()
        self.encoder.append(torch.nn.Linear(state_dim, state_dim))
        self.encoder.append(torch.nn.LeakyReLU(negative_slope=0.01))
        for i in range(encoder_extra_layers):
            self.encoder.append(torch.nn.Linear(state_dim, state_dim))
            self.encoder.append(torch.nn.LeakyReLU(negative_slope=0.01))

        # linear A and B matrices
        self.A = torch.nn.Linear(state_dim, state_dim)
        actuator_length = (input_dim - state_dim) // 2 # divide by 2 cuz input has u_t and u_t+1
        self.B = torch.nn.Linear(actuator_length, state_dim)

        self.decoder = torch.nn.Sequential()
        self.decoder.append(InverseLeakyReLU(slope=0.01))
        self.decoder.append(torch.nn.Linear(state_dim, state_dim))
        for i in range(decoder_extra_layers):
            self.decoder.append(InverseLeakyReLU(slope=0.01))
            self.decoder.append(torch.nn.Linear(state_dim, state_dim))

    def forward(self, padded_input):

        state_dim = self.output_dim

        x_t = padded_input[:state_dim]
        u_t = padded_input[state_dim:]

        z_t = self.encoder(x_t)

        z_t1 = self.A(z_t) + self.B(u_t)

        x_t1 = self.decode(z_t1)

        return x_t1

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
