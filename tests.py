import unittest
import torch
from customDatasetMakers import get_state_indices_dic, state_to_dic, dic_to_state
from train_helpers import get_mask, masked_loss
import numpy as np

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
        self.assertDictEqual(get_state_indices_dic(profiles, parameters, actuators, nx=3),
                             {'one': [0,1,2], 'two': [3,4,5], 'three': 6, 'four': [7,9], 'five': [8,10]})
        profiles=['one']
        parameters=[]
        actuators=[]
        self.assertDictEqual(get_state_indices_dic(profiles, parameters),
                             {'one': list(range(33))})
    def test_state_dic_conversions(self):
        state=torch.arange(11)
        dic=state_to_dic(state,['one','two'],['three'],['four'],nx=4)
        true_dic={'one': np.arange(4), 'two': np.arange(4,8), 'three': 8, 'four': np.arange(9,11)}
        self.assert_numpy_dictionaries_equal(true_dic, dic)
        states=torch.zeros((3,68)) # 3 timesteps, 2 profiles, 1 actuator
        states[-1,-2:]=torch.tensor([2,3])
        dic=state_to_dic(states,['one','two'],[],['three'])
        true_dic={'one': np.zeros((3,33)), 'two': np.zeros((3,33)), 'three': np.array([[0,0],[0,0],[2,3]])}
        self.assert_numpy_dictionaries_equal(true_dic, dic)
    def test_inversion(self):
        profiles=['one']
        parameters=['two']
        actuators=['four']
        start_dic={'one': [[1,2,3],[2,2,3]], 'two': [1,2], 'four': [[3,3],[3,3]]}
        state=dic_to_state(start_dic,
                           profiles,parameters,actuators,nx=3)
        end_dic=state_to_dic(state, profiles, parameters, actuators,nx=3)
        self.assert_numpy_dictionaries_equal(start_dic, end_dic)
        start_state=np.arange(9)
        profiles=['one','two']
        parameters=['three']
        actuators=['four']
        dic=state_to_dic(start_state,
                         profiles,parameters,actuators,nx=3)
        end_state=dic_to_state(dic,
                               profiles,parameters,actuators,nx=3)
        print(start_state)
        print(end_state)
        self.assertTrue(np.allclose(start_state,end_state))

class TestTrainHelpers(unittest.TestCase):
    def test_mask(self):
        shape=(2,6,3)
        lengths=[6,4]
        nwarmup=2
        masked_indices=[0,2]
        mask=get_mask(shape, lengths, nwarmup, masked_indices=masked_indices)
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
        loss=masked_loss(torch.nn.MSELoss(reduction='sum'),
                         output,target,
                         lengths=lengths,
                         nwarmup=nwarmup,
                         masked_indices=masked_indices)
        self.assertEqual(loss,100)

if __name__ == '__main__':
    unittest.main()
