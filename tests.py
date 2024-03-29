import unittest
import torch
import os
from customDatasetMakers import get_state_indices_dic, state_to_dic, dic_to_state, \
    preprocess_data
from dataSettings import get_denormalized_dic, get_normalized_dic
from customModels import IanRNN, HiroLinear
from train_helpers import get_state_mask, get_sample_time_state_mask, masked_loss
import numpy as np

# takes ~90 seconds the first time then faster after (I think h5 unravels itself like DNA / histones)
@unittest.SkipTest
class TestPreprocessedData(unittest.TestCase):
    def test_ech_exclusion(self):
        data_filename='/projects/EKOLEMEN/profile_predictor/joe_hiro_models/diiid_data.h5'
        profiles=['zipfit_etempfit_rho']
        scalars=['ech_pwr_total']
        # 152621 is an ECH shot, 163303 is not
        shots=[152621,163303]
        preprocessed_data=preprocess_data(None,
                                          data_filename,profiles,scalars,
                                          shots=shots,lookahead=1,
                                          exclude_ech=False,
                                          zero_fill_signals=['ech_pwr_total'])
        returned_shots=np.unique(preprocessed_data['shotnum'])
        for shot in shots:
            self.assertIn(shot,returned_shots)
        self.assertTrue(np.allclose(preprocessed_data['ech_pwr_total'][preprocessed_data['shotnum']==163303],0))
        self.assertFalse(np.allclose(preprocessed_data['ech_pwr_total'][preprocessed_data['shotnum']==152621],0))
        preprocessed_data=preprocess_data(None,
                                          data_filename,profiles,scalars,
                                          shots=shots,lookahead=1,
                                          exclude_ech=True,
                                          zero_fill_signals=['ech_pwr_total'])
        returned_shots=np.unique(preprocessed_data['shotnum'])
        self.assertNotIn(152621,returned_shots)
        self.assertIn(163303,returned_shots)
        preprocessed_data=preprocess_data(None,
                                          data_filename,profiles,scalars,
                                          shots=shots,lookahead=1,
                                          exclude_ech=False)
        returned_shots=np.unique(preprocessed_data['shotnum'])
        self.assertIn(152621,returned_shots)
        self.assertIn(163303,returned_shots)

class TestStateDicConversions(unittest.TestCase):
    def assert_numpy_dictionaries_equal(self, first_dic, second_dic):
        # ensures lists have same number of elements regardless of order
        self.assertCountEqual(first_dic.keys(),second_dic.keys())
        for sig in first_dic:
            self.assertEqual(np.shape(second_dic[sig]),np.shape(first_dic[sig]))
            self.assertTrue(np.allclose(second_dic[sig],first_dic[sig]))
    def test_get_state_indices(self):
        profiles=['one', 'two']
        parameters=['three']
        actuators=['four','five']
        calculations=[]
        self.assertDictEqual(get_state_indices_dic(profiles, parameters, calculations, actuators, nx=3),
                             {'one': [0,1,2], 'two': [3,4,5], 'three': 6, 'four': [7,9], 'five': [8,10]})
        profiles=['one']
        parameters=[]
        actuators=[]
        calculations=[]
        self.assertDictEqual(get_state_indices_dic(profiles,parameters,calculations,actuators),
                             {'one': list(range(33))})
        profiles=['one']
        parameters=[]
        calculations=['two']
        actuators=['three']
        nx=3
        result=get_state_indices_dic(profiles,parameters,calculations,actuators,nx=nx)
        self.assertDictEqual(result,
                             {'one': [0,1,2], 'two': [3,4,5], 'three': [6,7]})
    def test_state_dic_conversions(self):
        state=torch.arange(11)
        profiles=['one','two']
        parameters=['three']
        calculations=[]
        actuators=['four']
        dic=state_to_dic(state,profiles,parameters,calculations,actuators,nx=4)
        true_dic={'one': np.arange(4), 'two': np.arange(4,8), 'three': 8, 'four': np.arange(9,11)}
        self.assert_numpy_dictionaries_equal(true_dic, dic)
        states=torch.zeros((3,68)) # 3 timesteps, 2 profiles, 1 actuator
        states[-1,-2:]=torch.tensor([2,3])
        profiles=['one','two']
        parameters=[]
        calculations=[]
        actuators=['three']
        dic=state_to_dic(states,profiles,parameters,calculations,actuators)
        true_dic={'one': np.zeros((3,33)), 'two': np.zeros((3,33)), 'three': np.array([[0,0],[0,0],[2,3]])}
        self.assert_numpy_dictionaries_equal(true_dic, dic)
    def test_inversion(self):
        profiles=['one']
        parameters=['two']
        actuators=['four']
        calculations=[]
        start_dic={'one': [[1,2,3],[2,2,3]], 'two': [1,2], 'four': [[3,3],[3,3]]}
        state=dic_to_state(start_dic,
                           profiles,parameters,calculations,actuators,nx=3)
        end_dic=state_to_dic(state,profiles,parameters,calculations,actuators,nx=3)
        self.assert_numpy_dictionaries_equal(start_dic, end_dic)
        start_state=np.arange(9)
        profiles=['one','two']
        parameters=['three']
        actuators=['four']
        dic=state_to_dic(start_state,
                         profiles,parameters,calculations,actuators,nx=3)
        end_state=dic_to_state(dic,
                               profiles,parameters,calculations,actuators,nx=3)
        self.assertTrue(np.allclose(start_state,end_state))

class TestNormalizations(unittest.TestCase):
    def assert_numpy_dictionaries_equal(self, first_dic, second_dic):
        # ensures lists have same number of elements regardless of order
        self.assertCountEqual(first_dic.keys(),second_dic.keys())
        for sig in first_dic:
            self.assertEqual(np.shape(second_dic[sig]),np.shape(first_dic[sig]))
            self.assertTrue(np.allclose(second_dic[sig],first_dic[sig]))
    def test_denormalization(self):
        dic={'zipfit_etempfit_rho': [0.5,0.5], 'PETOT_astrainterpretive': [0.5,0.5],
             'qpsi': [2,2],
             'bt': 1, 'ip': 1e-6}
        denormed_dic=get_denormalized_dic(dic)
        true_dic={'zipfit_etempfit_rho': [1,1], 'PETOT_astrainterpretive': [1,1], 'qpsi': [0.5,0.5],
                  'bt': 1, 'ip': 1}
        self.assert_numpy_dictionaries_equal(denormed_dic, true_dic)
        dic={'zipfit_etempfit_rho': [[0.5,0.5],[0.5,0.5]],
             'PETOT_astrainterpretive': [[0.5,0.5],[0.5,0.5]],
             'qpsi': [[2,2],[2,2]],
             'bt': [1,1], 'ip': [1e-6,1e-6]}
        denormed_dic=get_denormalized_dic(dic)
        true_dic={'zipfit_etempfit_rho': [[1,1],[1,1]], 'PETOT_astrainterpretive': [[1,1],[1,1]],
                  'qpsi': [[0.5,0.5],[0.5,0.5]],
                  'bt': [1,1], 'ip': [1,1]}
        self.assert_numpy_dictionaries_equal(denormed_dic, true_dic)
    def test_normalization(self):
        dic={'zipfit_etempfit_rho': [2,2], 'PETOT_astrainterpretive': [2,2],
             'qpsi': [0.5,0.5],
             'bt': 1, 'ip': 1e6}
        normed_dic=get_normalized_dic(dic)
        true_dic={'zipfit_etempfit_rho': [1,1], 'PETOT_astrainterpretive': [1,1], 'qpsi': [2,2],
                  'bt': 1, 'ip': 1}
        self.assert_numpy_dictionaries_equal(normed_dic, true_dic)
        dic={'zipfit_etempfit_rho': [[2,2],[2,2]],
             'PETOT_astrainterpretive': [[2,2],[2,2]],
             'qpsi': [[0.5,0.5],[0.5,0.5]],
             'bt': [1,1], 'ip': [1e6,1e6]}
        normed_dic=get_normalized_dic(dic)
        true_dic={'zipfit_etempfit_rho': [[1,1],[1,1]], 'PETOT_astrainterpretive': [[1,1],[1,1]],
                  'qpsi': [[2,2],[2,2]],
                  'bt': [1,1], 'ip': [1,1]}
        self.assert_numpy_dictionaries_equal(normed_dic, true_dic)
    def test_gyrobohm_normalization(self):
        dic={'zipfit_trotfit_rho': [200,100], 'zipfit_edensfit_rho': [2,2], 'volume_EFIT01': 10, 'rmaxis_EFIT01': 1, 'aminor_EFIT01': 1, 'ip': 1e6,
             'times': 10, 'shotnum': 10}
        normed_dic=get_normalized_dic(dic, use_fancy_normalization=True)
        true_dic={'zipfit_trotfit_rho': [1,0.5], 'zipfit_edensfit_rho': [1,1], 'volume_EFIT01': 1, 'rmaxis_EFIT01': 1, 'aminor_EFIT01': 1, 'ip': 1,
                  'times': 10, 'shotnum': 10}
        self.assert_numpy_dictionaries_equal(normed_dic, true_dic)
        identity_dic=get_denormalized_dic(normed_dic,use_fancy_normalization=True)
        self.assert_numpy_dictionaries_equal(dic, identity_dic)
        dic={'zipfit_trotfit_rho': [[200,100],[200,100]],
             'zipfit_edensfit_rho': [[2,2],[2,2]],
             'volume_EFIT01': [10,10],
             'rmaxis_EFIT01': [1,1],
             'ip': [1e6,1e6],
             'aminor_EFIT01': [1,1]}
        normed_dic=get_normalized_dic(dic,use_fancy_normalization=True)
        true_dic={'zipfit_trotfit_rho': [[1,0.5],[1,0.5]],
                  'zipfit_edensfit_rho': [[1,1],[1,1]],
                  'volume_EFIT01': [1,1],
                  'rmaxis_EFIT01': [1,1],
                  'ip': [1,1],
                  'aminor_EFIT01': [1,1]}
        self.assert_numpy_dictionaries_equal(normed_dic, true_dic)
        identity_dic=get_denormalized_dic(normed_dic,use_fancy_normalization=True)
        self.assert_numpy_dictionaries_equal(dic, identity_dic)
        # 3 samples, 2 timesteps, 4 rho points -- for normalization of preprocessed data
        dic={'zipfit_trotfit_rho': [[[200,100,100,100],[200,100,100,100]],
                                    [[200,100,100,100],[200,100,100,100]],
                                    [[400,100,100,100],[200,100,100,100]]],
             'zipfit_edensfit_rho': [[[2,2,2,2],[2,2,2,2]],
                                     [[2,2,2,2],[2,2,2,2]],
                                     [[2,2,2,2],[2,2,2,2]]],
             'volume_EFIT01': [[20,20],
                               [20,20],
                               [20,20]],
             'rmaxis_EFIT01': [[1,1],
                               [1,1],
                               [1,1]],
             'aminor_EFIT01': [[2,2],
                               [2,2],
                               [2,2]],
             'pinj': [[8000,8000],
                      [8000,8000],
                      [8000,8000]],
             'ip': [[2e6,2e6],
                    [2e6,2e6],
                    [2e6,2e6]]}
        normed_dic=get_normalized_dic(dic,use_fancy_normalization=True)
        true_dic={'zipfit_trotfit_rho': [[[2,1,1,1],[2,1,1,1]],
                                         [[2,1,1,1],[2,1,1,1]],
                                         [[4,1,1,1],[2,1,1,1]]],
                  'zipfit_edensfit_rho': [[[2,2,2,2],[2,2,2,2]],
                                          [[2,2,2,2],[2,2,2,2]],
                                          [[2,2,2,2],[2,2,2,2]]],
                  'volume_EFIT01': [[2,2],
                                    [2,2],
                                    [2,2]],
                  'rmaxis_EFIT01': [[1,1],
                                    [1,1],
                                    [1,1]],
                  'aminor_EFIT01': [[2,2],
                                    [2,2],
                                    [2,2]],
                  'pinj': [[2,2],
                           [2,2],
                           [2,2]],
                  'ip': [[2,2],
                         [2,2],
                         [2,2]]}
        self.assert_numpy_dictionaries_equal(normed_dic, true_dic)
        identity_dic=get_denormalized_dic(normed_dic,use_fancy_normalization=True)
        self.assert_numpy_dictionaries_equal(dic, identity_dic)

class TestTrainHelpers(unittest.TestCase):
    def test_state_mask(self):
        profiles=['one','two']
        parameters=['three','four']
        actuators=['onion'] # this doesn't even matter
        mask=get_state_mask(profiles,parameters,
                            masked_outputs=['two','three'], rho_bdry_index=3,
                            nx=4)
        truth=torch.Tensor([1,1,1,0,
                            0,0,0,0,
                            0,
                            1])
        self.assertTrue(torch.allclose(truth,mask))
        # testing default of rho_bdry = None
        mask=get_state_mask(profiles,parameters,
                            masked_outputs=['two','three'],
                            nx=4)
        truth=torch.Tensor([1,1,1,1,
                            0,0,0,0,
                            0,
                            1])
        self.assertTrue(torch.allclose(truth,mask))
    def test_mask(self, use_gpu=True):
        if use_gpu and torch.cuda.is_available():
            device='cuda'
            print(f"Using {torch.cuda.device_count()} GPU(s)")
        else:
            device='cpu'
        lengths=[6,4]
        nwarmup=2
        state_mask=torch.Tensor([0,1,0])
        truth=torch.Tensor([[[0,0,0], #first sample
                             [0,0,0],   #timesteps in sample
                             [0,1,0],      #state elements in timestep
                             [0,1,0],
                             [0,1,0],
                             [0,1,0]],
                            [[0,0,0], #second sample
                             [0,0,0],
                             [0,1,0],
                             [0,1,0],
                             [0,0,0],
                             [0,0,0]]])
        mask=get_sample_time_state_mask(state_mask, truth.size(),
                                        lengths=lengths, nwarmup=nwarmup)
        self.assertTrue(np.allclose(truth,mask))
        output=torch.Tensor([[[1,2,3], #first sample
                              [1,2,3],   #timesteps in sample
                              [1,10,3],      #state elements in timestep
                              [1,10,3],
                              [1,10,3],
                              [1,10,3]],
                             [[1,2,3], #second sample
                              [1,2,3],
                              [1,10,3],
                              [1,10,3],
                              [1,2,3],
                              [1,2,3]]])
        target=torch.zeros_like(output)
        output=output.to(device)
        target=target.to(device)
        mask=mask.to(device)
        loss=masked_loss(torch.nn.MSELoss(reduction='sum'),
                         output,target,
                         mask)
        self.assertEqual(loss,100)
        loss=masked_loss(torch.nn.MSELoss(reduction='sum'),
                         output,target,
                         mask)
        self.assertEqual(loss,100)

        # this is a full-stack test of the model going through to loss
        model=IanRNN(input_dim=3, output_dim=3,
                     encoder_dim=1, encoder_extra_layers=0,
                     rnn_dim=1, rnn_num_layers=1,
                     decoder_dim=1, decoder_extra_layers=0,
                     rnn_type='linear')
        model.to(device)
        # by setting all weights to 0 and biases to 1 it's the identity map
        # [1,...,1]*x + 1 "encoder" layer
        # 1*x + 1 "rnn" (really linear here) layer
        # 1*x + 1 "decoder" layer
        # [1,...,1]*x + [1,...,1] final layer (also part of "decoder")
        for name, param in model.named_parameters():
            # Just an example
            if 'weight' in name:
                param.data = torch.ones_like(param)
            elif 'bias' in name:
                param.data = torch.zeros_like(param)
        new_output=model(output,reset_probability=1)
        new_output=new_output.to(device)
        loss=masked_loss(torch.nn.MSELoss(reduction='sum'),
                         new_output,target,
                         mask)


class TestModels(unittest.TestCase):
    def test_ian_rnn(self, use_gpu=True):
        state_length=2
        actuator_length=1
        # 4 total inputs going in
        if use_gpu and torch.cuda.is_available():
            device='cuda'
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            print(f"Using {torch.cuda.device_count()} GPU(s)")
        model=IanRNN(input_dim=state_length+2*actuator_length, output_dim=state_length,
                     encoder_dim=1, encoder_extra_layers=0,
                     rnn_dim=1, rnn_num_layers=1,
                     decoder_dim=1, decoder_extra_layers=0,
                     rnn_type='linear')
        # by setting all weights and biases to 1, should be like
        # [1,...,1]*x + 1 "encoder" layer
        # 1*x + 1 "rnn" (really linear here) layer
        # 1*x + 1 "decoder" layer
        # [1,...,1]*x + [1,...,1] final layer (also part of "decoder")
        for name, param in model.named_parameters():
            # Just an example
            if 'weight' in name:
                param.data = torch.ones_like(param)
            elif 'bias' in name:
                param.data = torch.ones_like(param)
        test_input=torch.ones((2,2,4))
        test_input[:,-1,-1]=2
        desired_output=torch.ones((2,2,2)) # [8,8]
        desired_output[:,0,:]*=8
        desired_output[:,1,:]*=9
        model_output=model(test_input,reset_probability=1)
        self.assertTrue(torch.allclose(model_output,desired_output))
        desired_output=torch.ones((2,2,2))
        desired_output[:,0,:]*=8
        desired_output[:,1,:]*=23
        model_output=model(test_input,reset_probability=0)
        self.assertTrue(torch.allclose(model_output,desired_output))
        # test rnn works
        model=IanRNN(input_dim=state_length+2*actuator_length, output_dim=state_length,
                     encoder_dim=10, encoder_extra_layers=0,
                     rnn_dim=12, rnn_num_layers=1,
                     decoder_dim=13, decoder_extra_layers=0,
                     rnn_type='lstm')
        # check that lstm works at all (don't have a careful test for output correctness)
        model(test_input,reset_probability=0)
        model(test_input,reset_probability=1)
    def test_HiroLinear(self, use_gpu=True):
        state_length=2
        actuator_length=1
        # 4 total inputs going in
        if use_gpu and torch.cuda.is_available():
            device='cuda'
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            print(f"Using {torch.cuda.device_count()} GPU(s)")
        model=HiroLinear(input_dim=state_length+2*actuator_length, output_dim=state_length,
                     encoder_extra_layers=1,
                        )
        '''for name, param in model.named_parameters():
            # Just an example
            if 'weight' in name:
                param.data = torch.ones_like(param)
            elif 'bias' in name:
                param.data = torch.zeros_like(param)'''
        test_input=torch.ones((2,3,4))
        test_input[:, 0, -1]=2
        test_input[:,-1,-2]=2
        test_input[:,-1,-1]=3
        desired_output=torch.ones((2,2,2)) # [8,8]
        desired_output[:,0,:]*=19
        desired_output[:,1,:]*=21
        '''import pdb; pdb.set_trace()
        for layer in model.encoder:
            print(layer[0].weight)'''
        model_output=model(test_input,reset_probability=1)
        #print(model_output)
        #self.assertTrue(torch.allclose(model_output,desired_output))
        desired_output=torch.ones((2,4,2))
        desired_output[:,0,:]*=19
        desired_output[:,1,:]*=45
        model_output=model(test_input,reset_probability=0)
        #print(model)
        #print(model_output)
        #import pdb; pdb.set_trace()
        #self.assertTrue(torch.allclose(model_output,desired_output))
    '''def test_HiroLinear_invertibility(self, use_gpu=True):
        # to check invertibility, we set Az+Bu=z ie B is zeros, A is identity 
        state_length=2
        actuator_length=1
        # 4 total inputs going in
        if use_gpu and torch.cuda.is_available():
            device='cuda'
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            print(f"Using {torch.cuda.device_count()} GPU(s)")
        model=HiroLinear(input_dim=state_length+2*actuator_length, output_dim=state_length,
                     encoder_extra_layers=1,
                     decoder_extra_layers=1
                        )
        for name, param in model.named_parameters():
            # Just an example
            if 'weight' in name:
                param.data = torch.ones_like(param)
            elif 'bias' in name:
                param.data = torch.ones_like(param)
        print(model.named_parameters())
        #linear_layer.weight.data = torch.eye(10)
        test_input=torch.ones((2,2,4))
        test_input[:, 0, -1]=2
        test_input[:,-1,-2]=2
        test_input[:,-1,-1]=3
        desired_output=torch.ones((2,2,2)) # [8,8]
        desired_output[:,0,:]*=19
        desired_output[:,1,:]*=21
        model_output=model(test_input,reset_probability=1)
        print(model_output)
        #self.assertTrue(torch.allclose(model_output,desired_output))
        desired_output=torch.ones((2,2,2))
        desired_output[:,0,:]*=19
        desired_output[:,1,:]*=45
        model_output=model(test_input,reset_probability=0)
        print(model_output)'''
if __name__ == '__main__':
    unittest.main()
