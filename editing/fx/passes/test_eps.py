import unittest
import torch
import torch.fx as fx
import torch.nn as nn
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

class TestEpsConversion(unittest.TestCase):
    def test_eps_conversion_pact_linears(self):
        eps_in = torch.tensor(0.1)
        m = nn.Linear(10, 5)
        result = eps_conversion_pact_linears(m, eps_in)
        self.assertEqual(result, eps_in.type_as(eps_in))

    def test_eps_conversion_pact_acts(self):
        eps_in = torch.tensor(0.1)
        m = PACTAsymmetricAct()
        result = eps_conversion_pact_acts(m, eps_in)
        self.assertEqual(result, m.get_eps().type_as(eps_in))

    def test_eps_conversion_invalid(self):
        m = nn.Conv2d(3, 64, kernel_size=3)
        with self.assertRaises(AssertionError):
            eps_conversion_invalid(m)

    # Add more test cases for other conversion functions...

class TestNLevelsOut(unittest.TestCase):


    def test_n_levels_out_pact_linears(self):
        in_levels = [256, 256]
        accumulator_levels = 2**32
        m = nn.Linear(10, 5)
        result = n_levels_out_pact_linears(m, in_levels, accumulator_levels)
        self.assertEqual(result, accumulator_levels)

    def test_n_levels_out_pact_acts(self):
        in_levels = [256]
        accumulator_levels = 2**32
        m = PACTAsymmetricAct()
        result = n_levels_out_pact_acts(m, in_levels, accumulator_levels)
        self.assertEqual(result, m.n_levels)

    def test_n_levels_out_invalid(self):
        in_levels = [256]
        accumulator_levels = 2**32
        m = nn.Conv2d(3, 64, kernel_size=3)
        with self.assertRaises(AssertionError):
            n_levels_out_invalid(m, in_levels, accumulator_levels)

    # Add more test cases for other n_levels_out functions...

class TestSignedOut(unittest.TestCase):
    def test_signed_out_pact_wrap(self):
        si = [True]
        m = PACTWrapModule()
        result = signed_out_pact_wrap(m, si)
        self.assertEqual(result, m.statTracker.get_eps())

    def test_signed_out_or_in_signed(self):
        si = [True, False, True]
        m = nn.MaxPool2d(2)
        result = signed_out_or_in_signed(m, si)
        self.assertTrue(result)

    def test_has_signed_attr(self):
        si = [True, False, True]
        m = nn.BatchNorm1d(10)
        result = has_signed_attr(m, si)
        self.assertTrue(result)

    # Add more test cases for other signed_out functions...

if __name__ == '__main__':
    unittest.main()