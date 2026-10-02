"""M0 full-region regression coverage, including the MC08 curvature failure."""
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from m0_sensory_v1.model.bayesian import (
    LogOddsCoefficients, bayesian_choice_probability, category_variances,
    log_odds_coefficients, solve_positive_decision_region,
)
from m0_sensory_v1.model.m0 import predict_m0_trials_with_parameters
from m0_sensory_v1.model.m1_bayesian import integrate_general_decision_region, m1_bayesian_choice_probability
from m0_sensory_v1.model.parameters import decode_raw_parameters, encode_constrained_parameters
from m0_sensory_v1.registry import load_model_registry
from _resume_staleness import result_registry_is_stale

DTYPE = torch.float64


class M0FullRegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_model_registry()
        cls.moments = cls.registry.task_moments

    def parameters(self, w=0.0001, kappa=1.35):
        raw = encode_constrained_parameters(torch.tensor([w, .492, .16, .46, kappa], dtype=DTYPE))
        raw.requires_grad_(True)
        return raw, decode_raw_parameters(raw)

    def test_mc08_negative_curvature_is_valid_and_matches_context_readout(self):
        raw, parameters = self.parameters()
        durations = torch.tensor([.2, .5, .8, 1.7, 2., 2.3], dtype=DTYPE)
        category = (durations > 1).to(DTYPE)
        coefficients = log_odds_coefficients(category_variances(parameters, self.moments), self.moments, raw.new_tensor(.5), parameters.log_kappa)
        self.assertLess(float(coefficients.a2.detach()), 0)
        with self.assertRaisesRegex(ValueError, 'not an upper long-choice threshold'):
            solve_positive_decision_region(coefficients)
        prediction = predict_m0_trials_with_parameters(durations, category, torch.zeros_like(durations), parameters, self.moments)
        expected = torch.stack([m1_bayesian_choice_probability(s, parameters, self.moments, raw.new_tensor(.5)).probability_long for s in durations])
        torch.testing.assert_close(prediction.non_lapse_probability_long, expected, rtol=0, atol=0)
        expected_lapse = torch.where(category == 1, parameters.ell_long, parameters.ell_short)
        torch.testing.assert_close(prediction.probability_long, (1-expected_lapse)*expected + expected_lapse/2, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(prediction.non_lapse_probability_long.sum(), raw, retain_graph=True)[0]
        reference_grad = torch.autograd.grad(expected.sum(), raw)[0]
        torch.testing.assert_close(actual_grad, reference_grad, atol=1e-10, rtol=1e-10)
        self.assertTrue(torch.isfinite(actual_grad).all())

    def test_all_regions_match_independent_gaussian_integrals_and_scalar_gradients(self):
        # Roots at -1, 2; both linear directions; constant/tie; no and double roots.
        cases = [(1.,-1.,-2.),(-1.,1.,2.),(0.,1.,-1.),(0.,-1.,1.),
                 (0.,0.,1.),(0.,0.,-1.),(0.,0.,0.),(1.,0.,1.),
                 (-1.,0.,-1.),(1.,-2.,1.),(-1.,2.,-1.)]
        durations = torch.tensor([.2,.8,1.7,2.3],dtype=DTYPE)
        for values in cases:
            with self.subTest(coefficients=values):
                c = torch.tensor(values,dtype=DTYPE,requires_grad=True)
                sd = torch.tensor(.7,dtype=DTYPE,requires_grad=True)
                q, region = integrate_general_decision_region(durations, sd, LogOddsCoefficients(*c.unbind()))
                self.assertEqual(q.shape,durations.shape)
                scalar = torch.stack([integrate_general_decision_region(d,sd,LogOddsCoefficients(*c.unbind()))[0] for d in durations])
                torch.testing.assert_close(q,scalar,rtol=0,atol=0)
                grads = torch.autograd.grad(q.sum(),(c,sd),retain_graph=True,allow_unused=True)
                scalar_grads = torch.autograd.grad(scalar.sum(),(c,sd),allow_unused=True)
                for a,b in zip(grads,scalar_grads):
                    if a is None or b is None:
                        self.assertIs(a,b)
                    else:
                        self.assertTrue(torch.isfinite(a).all())
                        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
                # Integrate each connected region, independently of the shared CDF code.
                a2,a1,a0=values
                roots = sorted(np.roots([a2,a1,a0] if a2 else [a1,a0]).real.tolist()) if (a2 and a1*a1-4*a2*a0 >= 0) or (not a2 and a1) else []
                bounds = [-np.inf,*roots,np.inf]
                expected=[]
                for duration in durations.tolist():
                    mass=0.
                    for lo,hi in zip(bounds[:-1],bounds[1:]):
                        x = (lo+hi)/2 if np.isfinite(lo) and np.isfinite(hi) else (hi-1 if np.isfinite(hi) else lo+1 if np.isfinite(lo) else 0)
                        if a2*x*x+a1*x+a0 > 0:
                            mass += quad(lambda z: np.exp(-.5*((z-duration)/.7)**2)/(.7*np.sqrt(2*np.pi)),lo,hi,epsabs=1e-12)[0]
                    expected.append(mass)
                torch.testing.assert_close(q,torch.tensor(expected,dtype=DTYPE),rtol=1e-10,atol=1e-10)
                self.assertEqual(region.negative_root_policy,'integrate_all_real_measurement_regions')

    def test_batched_prior_preserves_individual_values_and_gradients(self):
        raw,p = self.parameters(w=.2)
        duration = torch.tensor([[.3,.7],[1.8,2.2]],dtype=DTYPE)
        prior = torch.tensor([[.1,.3],[.6,.9]],dtype=DTYPE,requires_grad=True)
        q = bayesian_choice_probability(duration,p,self.moments,prior).probability_long
        ref = torch.stack([m1_bayesian_choice_probability(d,p,self.moments,pi).probability_long for d,pi in zip(duration.reshape(-1),prior.reshape(-1))]).reshape(duration.shape)
        torch.testing.assert_close(q,ref,rtol=0,atol=0)
        g = torch.autograd.grad(q.sum(),(raw,prior),retain_graph=True)
        gr = torch.autograd.grad(ref.sum(),(raw,prior))
        for actual,expected in zip(g,gr): torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)

    def test_mixed_batched_regions_preserve_gradients_without_per_trial_loop(self):
        values = [(1.,-1.,-2.),(-1.,1.,2.),(0.,1.,-1.),(0.,-1.,1.),
                  (0.,0.,1.),(0.,0.,-1.),(0.,0.,0.),(1.,0.,1.),
                  (-1.,0.,-1.),(1.,-2.,1.),(-1.,2.,-1.)]
        c = torch.tensor(values * 20,dtype=DTYPE,requires_grad=True)
        s = torch.linspace(.2,2.3,len(c),dtype=DTYPE,requires_grad=True)
        sd = torch.full_like(s,.7,requires_grad=True)
        from m0_sensory_v1.model.m1_bayesian import _integrate_homogeneous_decision_region
        with patch('m0_sensory_v1.model.m1_bayesian._integrate_homogeneous_decision_region',
                   wraps=_integrate_homogeneous_decision_region) as kernel:
            q,region=integrate_general_decision_region(s,sd,LogOddsCoefficients(*c.unbind(-1)))
        self.assertLessEqual(kernel.call_count,10)
        scalar = [integrate_general_decision_region(d,n,LogOddsCoefficients(*row.unbind()))
                  for d,n,row in zip(s,sd,c)]
        expected = torch.stack([v for v,_ in scalar])
        torch.testing.assert_close(q,expected,rtol=0,atol=0)
        self.assertEqual(region.regime,tuple(r.regime for _,r in scalar))
        for field in ('lower_boundary_s','upper_boundary_s'):
            torch.testing.assert_close(getattr(region,field),torch.stack([getattr(r,field) for _,r in scalar]),equal_nan=True)
        gradients=torch.autograd.grad(q.sum(),(c,s,sd),retain_graph=True)
        expected_gradients=torch.autograd.grad(expected.sum(),(c,s,sd))
        for actual,reference in zip(gradients,expected_gradients):
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual,reference,rtol=1e-12,atol=1e-12)
        empty=torch.empty((0,2),dtype=DTYPE)
        q,region=integrate_general_decision_region(empty,empty,LogOddsCoefficients(empty,empty,empty))
        self.assertEqual(q.shape,(0,2))
        self.assertEqual(region.regime,())

    def test_mixed_dtype_batch_promotes_probabilities_boundaries_and_gradients(self):
        s=torch.tensor([.2,.7,1.7,2.3],dtype=torch.float32,requires_grad=True)
        sd=torch.full_like(s,.6,requires_grad=True)
        c=torch.tensor([(1.,-1.,-2.),(-1.,1.,2.),(0.,1.,-1.),(1.,0.,1.)],
                       dtype=torch.float64,requires_grad=True)
        q,region=integrate_general_decision_region(s,sd,LogOddsCoefficients(*c.unbind(-1)))
        references=[integrate_general_decision_region(d.double(),n.double(),LogOddsCoefficients(*row.unbind()))
                    for d,n,row in zip(s,sd,c)]
        expected=torch.stack([p for p,_ in references])
        self.assertEqual(q.dtype,torch.float64)
        torch.testing.assert_close(q,expected,rtol=0,atol=0)
        for field in ('lower_boundary_s','upper_boundary_s'):
            self.assertEqual(getattr(region,field).dtype,torch.float64)
            torch.testing.assert_close(getattr(region,field),torch.stack([getattr(r,field) for _,r in references]),equal_nan=True)
        gradients=torch.autograd.grad(q.sum(),(s,sd,c),retain_graph=True)
        expected_gradients=torch.autograd.grad(expected.sum(),(s,sd,c))
        for actual,reference in zip(gradients,expected_gradients):
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual,reference,rtol=0,atol=0)

    def test_m0_masks_and_full_region_diagnostics(self):
        _,p = self.parameters()
        s=torch.tensor([.5,float('nan'),2.,.7],dtype=DTYPE)
        category=torch.tensor([0.,0.,1.,0.],dtype=DTYPE)
        choices=torch.tensor([0.,1.,1.,float('nan')],dtype=DTYPE)
        prediction=predict_m0_trials_with_parameters(s,category,choices,p,self.moments)
        for field in ('probability_long','decision_lower_boundary_s','decision_upper_boundary_s','positive_threshold_s'):
            self.assertTrue(torch.isnan(getattr(prediction,field)[[1,3]]).all())
        self.assertEqual(prediction.decision_regime[1],'not_evaluated_invalid_choice')
        torch.testing.assert_close(prediction.positive_threshold_s,prediction.decision_upper_boundary_s,equal_nan=True)
        flipped=predict_m0_trials_with_parameters(s,category,1-choices,p,self.moments)
        torch.testing.assert_close(prediction.probability_long,flipped.probability_long,equal_nan=True)
        for size in (0,3):
            x=torch.full((size,),float('nan'),dtype=DTYPE)
            with patch('m0_sensory_v1.model.m0.bayesian_choice_probability',side_effect=AssertionError('must skip')):
                missing=predict_m0_trials_with_parameters(x,x,x,p,self.moments)
            self.assertEqual(len(missing.decision_regime),size)
            self.assertTrue(torch.isnan(missing.probability_long).all())

    def test_m0_calls_one_vectorized_integration(self):
        _,p=self.parameters()
        s=torch.linspace(.2,2.3,1000,dtype=DTYPE)
        with patch('m0_sensory_v1.model.m1_bayesian.integrate_general_decision_region',wraps=integrate_general_decision_region) as solver:
            predict_m0_trials_with_parameters(s,(s>1).to(DTYPE),torch.zeros_like(s),p,self.moments)
        self.assertEqual(solver.call_count,1)
        self.assertEqual(solver.call_args.args[2].a0.ndim,0)

    def test_mc08_curvature_boundary_gradients_match_finite_differences(self):
        raw,p=self.parameters()
        s=torch.tensor([.2,.8,1.7,2.3],dtype=DTYPE)
        c=(s>1).to(DTYPE); y=torch.tensor([0.,1.,0.,1.],dtype=DTYPE)
        def objective(r):
            q=predict_m0_trials_with_parameters(s,c,y,decode_raw_parameters(r),self.moments).probability_long
            return -(y*q.log()+(1-y)*torch.log1p(-q)).sum()
        g=torch.autograd.grad(objective(raw),raw)[0]
        h=1e-5; fd=[]
        for i in range(5):
            step=torch.zeros_like(raw); step[i]=h
            fd.append((objective(raw.detach()+step)-objective(raw.detach()-step))/(2*h))
        torch.testing.assert_close(g,torch.stack(fd),rtol=2e-4,atol=2e-5)

    def test_m0_registries_declare_new_rule_and_reject_old_identity(self):
        for name in ('m0_sensory_v1.json','m0_sensory_v1_mc06.json','m0_sensory_v1_mc07.json'):
            r=load_model_registry(ROOT/'config'/name)
            self.assertFalse(r.positive_root_only)
            config=json.loads(r.config_path.read_text())
            self.assertEqual(config['decision_rule'],'long if posterior_log_odds > 0')
            self.assertEqual(config['negative_root_policy'],'integrate_all_real_measurement_regions')
            self.assertTrue(result_registry_is_stale({'registry_config_sha256':'historical-positive-root-fit'},r))

if __name__=='__main__': unittest.main()
