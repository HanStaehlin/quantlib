import unittest
import torch
from quantlib.algorithms.pact.pact_ops import _PACTActivation

class TestPACTActivation(unittest.TestCase):
    def setUp(self):
        self.activation = _PACTActivation(n_levels=256, init_clip='max', learn_clip=True, act_kind='relu', leaky=0.1)

    def test_activation_forward(self):
        input = torch.randn(10, 10)
        output = self.activation(input)
        self.assertEqual(output.shape, input.shape)

    def test_clipping_params(self):
        self.activation.started = torch.tensor(True)
        self.activation.max.data = torch.tensor(5.0)
        self.activation.min.data = torch.tensor(-5.0)
        self.activation.running_mean.data = torch.tensor(0.0)
        self.activation.running_var.data = torch.tensor(1.0)

        #self.activation.update_clipping_params()

        self.assertAlmostEqual(self.activation.clip_hi.item(), 5.0)
        self.assertAlmostEqual(self.activation.clip_lo.item(), -5.0)

    def test_tqt_quantization(self):
        self.activation.tqt = True
        self.activation.noisy = torch.tensor(False)
        self.activation.learn_clip = True
        self.activation.symm = True

        input = torch.randn(10, 10)
        output = self.activation(input)

        self.assertEqual(output.shape, input.shape)

    def test_init_clip_percentile(self):
        self.activation.init_clip = 'percentile'
        self.activation.started = torch.tensor(True)
        self.activation.histogram = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.activation.prevEdges.data = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.activation.truemax.data = torch.tensor(10.0)
        self.activation.truemin.data = torch.tensor(0.0)
        self.activation.ready.data = torch.tensor(True)

        #self.activation.update_clipping_params()

        self.assertAlmostEqual(self.activation.clip_hi.item(), 10.0)
        self.assertAlmostEqual(self.activation.clip_lo.item(), 0.0)

if __name__ == '__main__':
    unittest.main()