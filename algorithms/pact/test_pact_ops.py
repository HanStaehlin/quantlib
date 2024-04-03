import unittest
import torch
from quantlib.algorithms.pact.pact_ops import _PACTActivation, PACTIntegerConcat

class TestPACTActivation(unittest.TestCase):
    def setUp(self):
        self.activation = _PACTActivation(n_levels=256, init_clip='max', learn_clip=True, act_kind='relu', leaky=0.1)

    def test_activation_forward(self):
        input = torch.randn(10, 10)
        output = self.activation(input)
        self.assertEqual(output.shape, input.shape)

    def test_tqt_quantization(self):
        self.activation.tqt = True
        self.activation.noisy = torch.tensor(False)
        self.activation.learn_clip = True
        self.activation.symm = True

        input = torch.randn(10, 10)
        output = self.activation(input)

        self.assertEqual(output.shape, input.shape)

    
    def test_updateHistogram(self):
        # Create a mock `_PACTActivation` object with `init_clip` set to "percentile"
        pact_act = _PACTActivation(n_levels=256, init_clip="percentile", act_kind='relu',learn_clip=True)


        data = torch.randn(10, 10)
        pact_act.updateHistogram(data)

        self.assertGreater(pact_act.histogram.sum(), 0)

    def test_updateClipBounds(self):
        # Create a mock `_PACTActivation` object with `init_clip` set to "percentile"
        pact_act = _PACTActivation(n_levels=256, init_clip="percentile", act_kind='relu',learn_clip=True)

        # Simulate histogram update (assuming you have a way to mock it)
        pact_act.histogram = torch.ones(pact_act.num_bins)
        pact_act.histogram[pact_act.num_bins // 2] = 10  # Set a peak in the middle

        # Call the updateClipBounds function
        pact_act.updateClipBounds()
        print(pact_act.min, pact_act.max)
        self.assertGreater(pact_act.min.item(),-0.9905)
        self.assertGreater(0.9995,pact_act.max.item())
    
    def test_forward_unstarted(self):
        # Create a mock `_PACTActivation` object that is not started
        pact_act = _PACTActivation(n_levels=256, init_clip="percentile", act_kind='relu',learn_clip=True)

        # Create some input data
        x = torch.randn(10, 10)

        # Call the forward function
        result = pact_act(x)

        self.assertIsInstance(result, torch.Tensor)

class TestPACTIntegerConcat(unittest.TestCase):
    def setUp(self):
        force_out_eps = False
        self.concat = PACTIntegerConcat(num_args=3, dim=1, stack_flag=False, signed=True, n_levels=256, init_clip='max', learn_clip=True, act_kind='relu', leaky=0.1)

    def test_reassign_epsilons(self):
        self.concat.reassign_epsilons()
        self.assertTrue(self.concat.epsilons[0].requires_grad)

    def test_forward(self):
        input1 = torch.randn(10, 10)
        input2 = torch.randn(10, 10)
        input3 = torch.randn(10, 10)
        output = self.concat(input1, input2, input3)
        self.assertEqual(output.shape, (10, 30))

if __name__ == '__main__':
    unittest.main()