import unittest
import torch
import torch.fx as fx
from quantlib.editing.fx.passes.eps import AnnotateEpsPass

class TestAnnotateEpsPass(unittest.TestCase):
    def test_run_pass_no_eps(self):
        # Create a mock `AnnotateEpsPass` object with no eps_in provided
        annotate_eps_pass = AnnotateEpsPass(eps_in=None, n_levels_in=256, accumulator_levels=2**32, signed_in=True, prop_eps=True, prop_n_levels=True, prop_sign=True, verbose=False)

        # Create a mock `fx.GraphModule` object
        gm = fx.GraphModule(torch.nn.Module(), fx.Graph())

        # Call the run_pass function
        result = annotate_eps_pass.run_pass(gm)

        # Assert that the result is equal to the input gm
        self.assertEqual(result, gm)

    def test_run_pass_with_eps(self):
        # Create a mock `AnnotateEpsPass` object with eps_in provided
        eps_in = torch.tensor(0.1)
        annotate_eps_pass = AnnotateEpsPass(eps_in=eps_in, n_levels_in=256, accumulator_levels=2**32, signed_in=True, prop_eps=True, prop_n_levels=True, prop_sign=True, verbose=False)

        # Create a mock `fx.GraphModule` object
        gm = fx.GraphModule(torch.nn.Module(), fx.Graph())

        # Call the run_pass function
        result = annotate_eps_pass.run_pass(gm)

        # Assert that the result is equal to the input gm
        self.assertEqual(result, gm)

if __name__ == '__main__':
    unittest.main()