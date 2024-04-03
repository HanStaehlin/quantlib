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
        # Create PACTIntegerSoftmax object with mock acts
        self.pact_int_softmax = PACTIntegerConcat(num_args=3, dim=1, stack_flag=False, signed=True, n_levels=256, init_clip='max', learn_clip=True, act_kind='relu', leaky=0.1)

    def test_reassign_epsilons(self):
        # Call the reassign_epsilons function
        self.pact_int_softmax.acts[0].clip_lo.data = torch.tensor([-2.0])
        self.pact_int_softmax.acts[1].clip_lo.data = torch.tensor([1.0])
        self.pact_int_softmax.acts[2].clip_lo.data = torch.tensor([-1.0])

        self.pact_int_softmax.acts[0].clip_hi.data = torch.tensor([3.0])
        self.pact_int_softmax.acts[1].clip_hi.data = torch.tensor([0.5])
        self.pact_int_softmax.acts[2].clip_hi.data = torch.tensor([-0.5])

        self.pact_int_softmax.reassign_epsilons()

        # Assert that all clip_lo and clip_hi values have the same epsilon difference
        expected_eps = (self.pact_int_softmax.acts[0].clip_hi.data - self.pact_int_softmax.acts[0].clip_lo.data) 
        for act in self.pact_int_softmax.acts:
            self.assertEqual(abs(act.clip_hi - act.clip_lo), expected_eps)

        # Assert that symm is set correctly based on clip_lo value
        for act in self.pact_int_softmax.acts:
            if act.clip_lo.item() < 0:
                self.assertTrue(act.symm)
            else:
                self.assertFalse(act.symm)

if __name__ == '__main__':
    unittest.main()