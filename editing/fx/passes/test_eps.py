import unittest
import torch
import torch.fx as fx
import torch.nn as nn
from unittest.mock import patch
from quantlib.editing.fx.passes.eps import AnnotateEpsPass, eps_conversion_pact_linears, eps_conversion_pact_acts, eps_conversion_invalid, n_levels_out_pact_linears, n_levels_out_pact_acts, n_levels_out_invalid, signed_out_pact_wrap, signed_out_or_in_signed, has_signed_attr
from quantlib.algorithms.pact import PACTAsymmetricAct, PACTWrapModule
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

class TestNLevelsOut(unittest.TestCase):


    def test_n_levels_out_pact_linears(self):
        in_levels = [256, 256]
        accumulator_levels = 2**32
        m = nn.Linear(10, 5)
        result = n_levels_out_pact_linears(m, in_levels, accumulator_levels)
        self.assertEqual(result, accumulator_levels)


    def test_n_levels_out_invalid(self):
        in_levels = [256]
        accumulator_levels = 2**32
        m = nn.Conv2d(3, 64, kernel_size=3)
        with self.assertRaises(AssertionError):
            n_levels_out_invalid(m, in_levels, accumulator_levels)


class TestSignedOut(unittest.TestCase):

    def test_signed_out_or_in_signed(self):
        si = [True, False, True]
        m = nn.MaxPool2d(2)
        result = signed_out_or_in_signed(m, si)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()