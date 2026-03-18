"""
Our hierarchical Bayesian model implementation
Predicts the HR outcomes for each player projected to play in a held out set
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

## modeling libraries
import jax
import jax.numpy as jnp
import jax.random as random
from jax.scipy.special import logit, expit
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.mcmc import MCMC
import patsy
import arviz as az
from scipy.stats import pearsonr

blue_color = '#1E90FF'
red_color = '#EF3E42'

seed = 42
key = jax.random.key(seed)
numpyro.set_host_device_count(6)
jax.config.update("jax_enable_x64", True) ## makes the arrays a little more precise

def create_model_dict(df):
    """
    helper function that creates the data dictionary to be put into a numpyro model
    """
    return {'events': jnp.array(df['events'].values), 
            'HR': jnp.array(df['HR'].values), ## our target
            'park_factor': jnp.array(df['park_factor'].values),
            'pos_idx': jnp.array(df['pos_idx'].values),
            'age_splines': jnp.array(np.stack(df['age_splines'].values)),
            ## season j-1
            'events_lag1': jnp.array(df['events_lag1'].values),
            'xHR_lag1': jnp.array(df['xHR_lag1'].values),
            'pos_idx_lag1': jnp.array(df['pos_idx_lag1'].values),
            ## season j-2
            'events_lag2': jnp.array(df['events_lag2'].values),
            'xHR_lag2': jnp.array(df['xHR_lag2'].values),
            'pos_idx_lag2': jnp.array(df['pos_idx_lag2'].values),
            ## season j-3
            'events_lag3': jnp.array(df['events_lag3'].values),
            'xHR_lag3': jnp.array(df['xHR_lag3'].values), 
            'pos_idx_lag3': jnp.array(df['pos_idx_lag3'].values),
            }

def hr_projection_model(events, park_factor, pos_idx, age_splines, 
                        events_lag1, xHR_lag1, pos_idx_lag1, 
                        events_lag2, xHR_lag2, pos_idx_lag2, 
                        events_lag3, xHR_lag3, pos_idx_lag3, 
                        HR=None):
    """
    bayesian hierarchical model for HR projections as discussed in the methodology Section 3.2 in the paper
    """
    n_obs = events.shape[0]
    n_pos = 9
    n_bases = 6 ## basis functions for the spline

    ## hyper priors for positional prior distributon --> our global population baseline
    mu = numpyro.sample("mu", dist.Normal(-3.0, 0.75))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))

    ## we model alpha_k as an expit function of z_k, which is our positional baseline
    ## we want to treat alpha_k as a random variable rather than a fixed one
    with numpyro.plate("positions", n_pos):
        z = numpyro.sample("z", dist.Normal(mu, sigma))
        alpha = numpyro.deterministic("alpha", expit(z))

    ## our stabilization point and acts as our shrinkage parameter
    ## represents the ratio of variances in the partial pooling equation
    M = numpyro.sample("M", dist.LogNormal(jnp.log(75.0), 0.25))

    ## our lagged weights for each previous season j-l, l = 1, 2, 3
    ## we hypothesize that the most recent season should take most of the weight
    beta = numpyro.sample("beta", dist.Dirichlet(jnp.array([6.0, 3.0, 1.0])))

    ## our age curve --> we used a cubic b-spline
    with numpyro.plate("pos_splines", n_pos, dim=-2):
        with numpyro.plate("bases", n_bases, dim=-1):
            ## our prior distribution on the basis function coefficients
            gamma = numpyro.sample("gamma", dist.Normal(0.0, 0.375)) 
    ## calculates the positional age effect for each player-season 
    f_k = jnp.sum(gamma[pos_idx] * age_splines, axis=-1)

    ## this serves as our adjusted estimate for a player's true talent estimated xHR total from each of the previous 3 seasons
    ## this adjustment is just a regression to the mean, in which players with small to zero N will fall towards alpha_k
    p_hat_1 = (xHR_lag1 + alpha[pos_idx_lag1] * M) / (events_lag1 + M)
    p_hat_2 = (xHR_lag2 + alpha[pos_idx_lag2] * M) / (events_lag2 + M)
    p_hat_3 = (xHR_lag3 + alpha[pos_idx_lag3] * M) / (events_lag3 + M)

    ## this applies each of the weights to the adjusted xHR estimate from season j-l
    ## this captures information of the player's previous seasons
    p_hist = beta[0] * p_hat_1 + beta[1] * p_hat_2 + beta[2] * p_hat_3

    ## we've done everything in the log odds (unbounded space) --> we want real probs [0, 1] space
    ## we use the expit AKA inverse-logit function to do this
    theta_logit = logit(p_hist) + f_k
    theta = numpyro.deterministic("theta", expit(theta_logit))

    ## park factor is already scaled as (BF + 1)/2
    ## adjusts theta, our latent true talent param, with park factor --> we already made the adjustment to park factor, so it's just a product
    p_adj = theta * park_factor

    ## our likelihood function --> we used a binomial process
    with numpyro.plate("data", n_obs):
        numpyro.sample("Y", dist.Binomial(total_count=events, probs=p_adj), obs=HR)

def bayesian_proj_model(train_dict, num_chains=4, num_warmup=1000, num_samples=2000):
    ## initalizes the NUTS sampler
    nuts_kernel = NUTS(hr_projection_model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    ## runs MCMC with the training set dictionary (and seed 42)
    mcmc.run(random.PRNGKey(seed), **train_dict)

    ## grabs the ArviZ idata
    idata = az.from_numpyro(mcmc)

    ## checks to make sure our sampler converged with the r_hat and ESS measures
    summary_df = az.summary(idata)
    print(summary_df['r_hat'].max())
    print(summary_df['ess_bulk'].min())

    ## grabs the posterior samples (all 8,000)
    posterior_samples = mcmc.get_samples()

    return idata, posterior_samples

def posterior_plots(idata, train_df, posterior_samples, hold_out_df, hold_out_dict, design_info, num_chains=4, num_samples=2000):
    """
    plots the posterior predictions for every player in the hold-out set
    plots the posterior distributions for each parameter
    """
    ## just makes sure the predictions don't just echo the observations
    test_inputs = {k: v for k, v in hold_out_dict.items() if k != "HR"}
    ## creates the predictions for the validation set
    predictive = Predictive(hr_projection_model, posterior_samples)
    predictions_2026 = predictive(jax.random.PRNGKey(seed), **test_inputs)
    Y_pred_samples = np.array(predictions_2026["Y"]) ## predicted HRs per sample

    ## grabs each player's HR samples and their metadata --> use in the streamlit dashboard
    flattened_data = []    
    for i, (_, row) in enumerate(hold_out_df.iterrows()):
        player_sims = Y_pred_samples[:, i]
        flattened_data.append({'name': row['name'].title(),
                               'team': row['team'],
                               'position': row['primary_pos'],
                               'projected_pa': row['PA'],
                               'projected_events': row['events'],
                               'simulated_hrs': player_sims
                               })
    export_df = pd.DataFrame(flattened_data)
    export_df.to_parquet('data/2026_hr_posterior_samples.parquet')

    # the forest plot of each player's posterior predictive interval (95% HDIs)
    sample_preds = Y_pred_samples
    sample_names = hold_out_df['name'].values

    pred_idata = az.from_dict(posterior_predictive={"HR_proj": np.expand_dims(sample_preds, axis=0)}, 
                              coords={"player": sample_names}, 
                              dims={"HR_proj": ["player"]}
                              )
    num_players = len(sample_names)
    dynamic_height = max(5, num_players * 0.15)

    ## the forest plot
    fig, ax = plt.subplots(figsize=(8, dynamic_height))
    axes = az.plot_forest(pred_idata.posterior_predictive, var_names=["HR_proj"], combined=True, colors=blue_color, hdi_prob=0.95, ax=ax)
    axes[0].set_title("Posterior Predictive HR Intervals (95% HDI)")
    axes[0].set_xlabel("Home Run Total")
    # axes[0].set_yticklabels(sample_names[::-1])
    axes[0].set_xlim(0, 70)
    axes[0].legend()
    plt.tight_layout()
    # plt.savefig("plots/fig_2026_all_intervals.png", format="png", dpi=300)
    plt.show()

    # age curve
    ages = np.linspace(20, 42, 100)
    age_df = pd.DataFrame({'Age': ages})
    ## spline basis with the same knots from the data
    spline_grid = patsy.build_design_matrices([design_info], age_df)[0]
    spline_grid = np.asarray(spline_grid)
    ## the estimated gamma coeficients for each sample 
    gamma_samples = posterior_samples['gamma']
    ## 3x3 grid plot for each position
    pos_labels = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
    fig, axes = plt.subplots(3, 3, figsize=(7, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    for k in range(9):
        ## samples * 100
        age_effect_k = gamma_samples[:, k, :] @ spline_grid.T
        age_effect_k = np.array(age_effect_k)
        ## posterior mean and 95% for the age curve
        age_curve_mean = age_effect_k.mean(axis=0)
        reshaped_effects = age_effect_k.reshape(num_chains, num_samples, -1)
        age_curve_hdi = az.hdi(reshaped_effects, hdi_prob=0.95)
        
        ax = axes[k]
        ax.plot(ages, age_curve_mean, color=blue_color, lw=2)
        ax.fill_between(ages, age_curve_hdi[:, 0], age_curve_hdi[:, 1], color=blue_color, alpha=0.3)
        ax.axhline(0, color=red_color, linestyle='--', alpha=0.5) 
        ax.set_title(pos_labels[k], fontsize=12, fontweight='bold')
        if k >= 6:
            ax.set_xlabel("Age")
        if k == 3:
            ax.set_ylabel(r"Age Effect on True Talent xHR% (Logit Scale)")
    plt.suptitle("Position-Specific Age Curves for True Talent xHR%", fontsize=16, y=1.02)
    plt.tight_layout()
    # plt.savefig("plots/fig_age_curves_2026.png", format="png", bbox_inches="tight")
    plt.show()

    # M - shrinkage param
    fig, ax = plt.subplots()
    az.plot_posterior(idata, var_names=["M"], hdi_prob=0.95, point_estimate="mean", color=blue_color, ax=ax)
    ax.set_title("95% HDI Posterior Distribution of $M$")
    ax.set_xlabel("BIPs")
    plt.tight_layout()
    # plt.savefig("plots/fig_M_posterior_2026.png", format="png", dpi=300)
    plt.show()

    # lag beta params --> our dirichlet prior params
    lag_labels = ["Season j-1 (Lag 1)", "Season j-2 (Lag 2)", "Season j-3 (Lag 3)"]
    fig, ax = plt.subplots()
    axes = az.plot_forest(idata, var_names=["beta"], combined=True, colors=blue_color, hdi_prob=0.95, ax=ax)
    axes[0].set_yticklabels(lag_labels[::-1])
    axes[0].set_title("95% HDI Posterior Distributions of Lag Weights ($\\beta$)")
    axes[0].set_xlim(0, 1)
    plt.tight_layout()
    # plt.savefig("plots/fig_beta_weights_2026.png", format="png", dpi=300)
    plt.show()

    # alpha_k --> the positional baseline 
    pos_labels = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]

    fig, ax = plt.subplots()
    axes = az.plot_forest(idata, var_names=["alpha"], colors=blue_color, combined=True, hdi_prob=0.95, ax=ax)
    axes[0].set_yticklabels(pos_labels[::-1])
    axes[0].set_title(r"95% HDI Posterior Distributions of $\alpha_k$")
    axes[0].set_xlabel("xHR Probability")
    axes[0].legend(loc='upper right')
    plt.tight_layout()
    # plt.savefig("plots/fig_alpha_k_2026.png", format="png", dpi=300)
    plt.show()

if __name__ == "__main__":
    test_year = 2026
    # sets up the train and test set data dictionaries (holds our inputs for the model)
    panel_data = pd.read_parquet('data/player_data_lagged_2026.parquet').copy()
    ## splitting the data into a training and test set
    train_df = panel_data[panel_data['game_year'] < test_year].copy()
    test_df = panel_data[panel_data['game_year'] == test_year].copy()
    ## generates the cubic B-spline matrix for modeling age curve
    spline_matrix = patsy.dmatrix("0 + bs(Age, df=6, degree=3)", train_df, return_type='dataframe')
    train_df['age_splines'] = list(spline_matrix.values)
    test_spline = patsy.build_design_matrices([spline_matrix.design_info], test_df)[0]
    test_df['age_splines'] = list(np.asarray(test_spline))

    ## the data dictionaries for each set
    master_train = create_model_dict(train_df)
    master_test = create_model_dict(test_df)

    idata, posterior_samples = bayesian_proj_model(master_train)
    posterior_plots(idata, train_df, posterior_samples, test_df, master_test, spline_matrix.design_info)
    
    print('run complete!')

