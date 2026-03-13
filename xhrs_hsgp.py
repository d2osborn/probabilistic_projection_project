"""
The HSGP implementation of modeling xHRs with Exit Velocity (EV) and Launch Angle (LA) in NumpyRo and using the Matern 5/2 Kernel function
"""

import pandas as pd
import numpy as np

## modeling libraries
from sklearn.model_selection import train_test_split

from optax import linear_onecycle_schedule

import jax
from jax import random
import jax.numpy as jnp
import jax.nn

import numpyro
from numpyro import distributions as dist
from numpyro.infer import Predictive
from numpyro.infer.elbo import Trace_ELBO
from numpyro.optim import Adam
from numpyro.infer.svi import SVI
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_median
from numpyro.contrib.hsgp.approximation import hsgp_matern


def hsgp_model_run():
    seed = 42
    numpyro.set_host_device_count(6)
    jax.config.update("jax_enable_x64", True) ## makes the arrays a little more precise

    batted_ball_events = pd.read_parquet('data/batted_ball_events.parquet')

    X = jnp.asarray(batted_ball_events[['launch_speed', 'launch_angle']].to_numpy(dtype=float))
    y = jnp.asarray(batted_ball_events['is_HR'].to_numpy(dtype=int))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_mean = jnp.mean(X_train, axis=0)
    X_train_std = jnp.std(X_train, axis=0)
    ## standardizing the continous features --> ensures numerical stability
    X_train_scaled = (X_train - X_train_mean) / X_train_std

    ## used to calculate the boundary limits of the HSGP
    L1 = abs(X_train_scaled[:, 0].max())
    L2 = abs(X_train_scaled[:, 1].max())

    empirical_hr_prob = batted_ball_events['is_HR'].value_counts(normalize=True).iloc[1]
    log_odds_baseline = np.log(empirical_hr_prob / (1 - empirical_hr_prob))

    def fit_svi(seed: int, model: callable, guide: callable, num_steps: int = 5_000, peak_lr: float = 0.01, **model_kwargs,):
        """
        Trains the HSGP with SVI --> approximates the posteriors with optimization rather than the exact posterior (i.e., MCMC)
        SVI uses stochastic optimization: mini batches + stochastic gradient ascent
        Source: https://www.geeksforgeeks.org/data-science/stochastic-variational-inference-svi/
        """
        lr_scheduler = linear_onecycle_schedule(num_steps, peak_lr) ## maps out the lr at steps 1, 2,..., 5000
        svi = SVI(model, guide, Adam(lr_scheduler), Trace_ELBO()) ## Adam updates the weights inside the guide 
        return svi.run(random.PRNGKey(seed), num_steps, progress_bar=False, **model_kwargs)

    # the HSGP model --> Matern 5/2 kernel function
    @jax.tree_util.register_pytree_node_class
    class HSGP_matern52_Model:
        def __init__(self, m: list[int], L: list[float]):
            self.m = m ## num of basis funcs
            self.L = L ## boundary limits

        def model(self, X: jax.Array, y: jax.Array | None = None):
            amplitude = numpyro.sample("amplitude", dist.LogNormal(0, 1)) ## A ~ LogNormal(0, 1) --> 0 is in the log-odds, so 50%
            length = numpyro.sample("lengthscale", dist.Exponential(jnp.ones(2))) ## l ~ Exponential([1, 1]) --> a length scale for each feature in R^2
            f_centered = hsgp_matern(X, alpha=amplitude, length=length, ell=self.L, m=self.m, nu=2.5) ## nu = 2.5 makes it matern 5/2 kernel
            f = numpyro.deterministic("f_star", f_centered + log_odds_baseline) ## we're setting the m(x) at -3 (in log odds) ≈ 0.0474 HR%
            site = "y" if y is not None else "y_test"
            numpyro.sample(site, dist.Bernoulli(logits=f), obs=y) ## Y ~ Bernoulli(theta), theta = logit(*)

        def tree_flatten(self):
                children = ()  
                aux_data = (self.L, self.m) 
                return (children, aux_data)

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            return cls(*children, **aux_data)

    hsgp_m = HSGP_matern52_Model(m=[15, 15], L=[L1 * 1.25, L2 * 1.25])
    hsgp_guide = AutoNormal(hsgp_m.model, init_loc_fn=init_to_median(num_samples=25))
    hsgp_res = fit_svi(seed, hsgp_m.model, hsgp_guide, X=X_train_scaled, y=y_train, num_steps=1_000)

    ## applying the hsgp model to our entire dataset of batted balls
    X_new_raw = jnp.asarray(batted_ball_events[['launch_speed', 'launch_angle']].to_numpy(dtype=float))
    X_new_scaled = (X_new_raw - X_train_mean) / X_train_std

    predictive = Predictive(hsgp_m.model, guide=hsgp_guide, params=hsgp_res.params, num_samples=500)
    post_preds = predictive(random.PRNGKey(seed), X=X_new_scaled)
    mean_f_star_new = post_preds["f_star"].mean(axis=0)
    mean_probs_new = jax.nn.sigmoid(mean_f_star_new) ## log odds -> [0, 1]

    batted_ball_events['xHR'] = jnp.array(mean_probs_new) ## xHRs is just the probability that the batted ball was a HR!

    batted_ball_events.to_parquet('data/batted_balls_w_xhrs.parquet') ## adds the xHR probability to the batted ball data from 2018-2025

if __name__ == "__main__":
    hsgp_model_run()
    print('run complete!')
    