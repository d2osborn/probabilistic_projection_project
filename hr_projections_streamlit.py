import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from scipy.special import logit, expit
import json

st.set_page_config(page_title="Bayesian 2026 HR Projections Dashboard", layout="wide")

MLB_COLORS = {
    'ARI': '#A71930', 'ATL': '#CE1141', 'BAL': '#DF4601', 'BOS': '#BD3039',
    'CHC': '#0E3386', 'CHW': '#27251F', 'CIN': '#C6011F', 'CLE': '#E31937',
    'COL': '#333366', 'DET': '#0C2340', 'HOU': '#EB6E1F', 'KCR': '#004687',
    'LAA': '#BA0021', 'LAD': '#1E90FF', 'MIA': '#00A3E0', 'MIL': '#FFC52F',
    'MIN': '#002B5C', 'NYM': '#FF5910', 'NYY': '#003087', 'ATH': '#003831',
    'PHI': '#E81828', 'PIT': '#FDB827', 'SDP': '#2F241D', 'SFG': '#FD5A1E',
    'SEA': '#0C2C56', 'STL': '#C41E3A', 'TBR': '#092C5C', 'TEX': '#003278',
    'TOR': '#134A8E', 'WSN': '#AB0003'
}

@st.cache_data
def load_data():
    df = pd.read_parquet('data/2026_hr_posterior_samples.parquet')
    df['mean_hr'] = df['simulated_hrs'].apply(np.mean).round(1)
    df['hdi_lower'] = df['simulated_hrs'].apply(lambda x: np.percentile(x, 2.5)).astype(int)
    df['hdi_upper'] = df['simulated_hrs'].apply(lambda x: np.percentile(x, 97.5)).astype(int)
    rostered_df = df[df['team'] != '--']    
    team_samples = rostered_df.groupby('team')['simulated_hrs'].apply(lambda x: np.sum(np.vstack(x), axis=0))    
    team_df = pd.DataFrame({'team': team_samples.index, 
                            'team_sims': team_samples.values
                            })
    team_df['mean_hr'] = team_df['team_sims'].apply(np.mean).round(1)
    team_df['hdi_lower'] = team_df['team_sims'].apply(lambda x: np.percentile(x, 2.5)).astype(int)
    team_df['hdi_upper'] = team_df['team_sims'].apply(lambda x: np.percentile(x, 97.5)).astype(int)
    team_df = team_df.sort_values('mean_hr', ascending=False).reset_index(drop=True)
    team_df.index += 1
    return df, team_df
df, team_df = load_data()

# NAVIGATION
st.sidebar.title("⚾ 2026 MLB HR Projections")
st.sidebar.markdown("Modeled with a hierarchical Bayesian framework in NumPyro & JAX")
page = st.sidebar.radio("Navigation", ["Player Projections", 
                                       "Player HR Leaderboard", 
                                       "Team HR Leaderboard", 
                                       "Sandbox Projections", 
                                       "Methodology", 
                                       "About",
                                       ])

st.sidebar.markdown("---")

# PLAYER PROJECTIONS
if page == "Player Projections":
    st.title("2026 Player Posterior Distributions for Projected HRs")
    st.markdown("Explore the full posterior of each player's projected HR count for the 2026 season. \
                You can filter by the team they play on and what position they play. \
                You can also select multiple players at once and see how their projected distributions compare.")

    col1, col2 = st.columns(2)
    with col1:
        team_filter = st.selectbox("Filter by Team", ["All Teams"] + sorted(df['team'].dropna().unique().tolist()))
    with col2:
        pos_filter = st.selectbox("Filter by Position", ["All Positions"] + sorted(df['position'].dropna().unique().tolist()))

    filtered_df = df.copy()
    if team_filter != "All Teams":
        filtered_df = filtered_df[filtered_df['team'] == team_filter]
    if pos_filter != "All Positions":
        filtered_df = filtered_df[filtered_df['position'] == pos_filter]

    selected_players = st.multiselect("Select Players to Compare:", 
                                      options=filtered_df['name'].tolist(), 
                                      default=[filtered_df['name'].iloc[0]] if not filtered_df.empty else []
                                      )

    if selected_players:
            fig = go.Figure()            
            distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            for i, player in enumerate(selected_players):
                player_data = df[df['name'] == player].iloc[0]
                sims = player_data['simulated_hrs']
                team = player_data['team']
                mean_hr = player_data['mean_hr']
                
                if team_filter == "All Teams":
                    fill_color = MLB_COLORS.get(team, '#1E90FF') 
                else:
                    fill_color = distinct_colors[i % len(distinct_colors)] 
                
                kde = gaussian_kde(sims, bw_method=0.3)
                x_vals = np.linspace(0, max(sims) + 5, 200)
                y_vals = kde(x_vals)

                fig.add_trace(go.Scatter(x=x_vals, 
                                         y=y_vals, 
                                         fill='tozeroy', 
                                         mode='lines', 
                                         line=dict(color=fill_color, width=3), 
                                         name=f"{player} ({team})", 
                                         hovertemplate="<b>HRs:</b> %{x:.0f}<br><b>Density:</b> %{y:.3f}<extra></extra>"
                                         ))
                
                mean_density_height = kde([mean_hr])[0]

                fig.add_trace(go.Scatter(x=[mean_hr, mean_hr], 
                                         y=[0, mean_density_height], 
                                         mode='lines', 
                                         line=dict(color=fill_color, width=2, dash='dash'), 
                                         opacity=0.8, 
                                         showlegend=False, 
                                         hoverinfo='skip'
                                         ))

                hdi_lower = player_data['hdi_lower']
                hdi_upper = player_data['hdi_upper']

                lower_density_height = kde([hdi_lower])[0]
                upper_density_height = kde([hdi_upper])[0]

                fig.add_trace(go.Scatter(x=[hdi_lower, hdi_lower], 
                                         y=[0, lower_density_height], 
                                         mode='lines', 
                                         line=dict(color=fill_color, width=1.5, dash='dot'), 
                                         opacity=0.6, 
                                         showlegend=False, 
                                         hoverinfo='skip'
                                         ))

                fig.add_trace(go.Scatter(x=[hdi_upper, hdi_upper], 
                                         y=[0, upper_density_height],
                                         mode='lines',
                                         line=dict(color=fill_color, width=1.5, dash='dot'),
                                         opacity=0.6,
                                         showlegend=False,
                                         hoverinfo='skip'
                                         ))
                
            fig.update_layout(title="Kernel Density Plot of Projected HRs for 2026",
                              xaxis_title="Projected # of Home Runs",
                              yaxis_title="Probability Density",
                              hovermode="x unified",
                              template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white", 
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                              )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("### Player Results")
            cols = st.columns(len(selected_players))
            for i, player in enumerate(selected_players):
                with cols[i]:
                    p_data = df[df['name'] == player].iloc[0]
                    st.info(f"**{p_data['name']}** (Age: {p_data['age']} | {p_data['position']} - {p_data['team']})")
                    spread = p_data['hdi_upper'] - p_data['hdi_lower']
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Projected HRs", f"{p_data['mean_hr']}")
                    with m2:
                        st.metric("95% HR HDI", f"{p_data['hdi_lower']}-{p_data['hdi_upper']}")
                    with m3:
                        st.metric("95% HR Uncertainty", f"{spread}")

# SANDBOX PLAYER MODE
elif page == "Sandbox Projections":
    st.title("Create-A-Player Sandbox Projections")
    st.markdown("""Use the model's learned parameters to project any hypothetical player. \
                Adjust their historical contact quality, age, team, and primary position to see exactly how the\
                Bayesian framework calculates their 2026 projected HRs and underlying true talent xHR%.\
                \\
                Note: To ensure instant interactivity, this engine uses the posterior means of the parameters rather than sampling from their full posterior distributions.
                """)

    with open('data/model_params.json', 'r') as f:
        params = json.load(f)
        M_mean = params["M"]
        beta_weights = params["beta"]
        alpha_baselines = params["alpha"]
        age_effects_dict = params["age_effects"]
    
    ballpark_factors = {
        'ARI': {2020: 112, 2021: 89, 2022: 86, 2023: 84, 2024: 86, 2025: 88},
        'ATH': {2020: 85, 2021: 85, 2022: 80, 2023: 84, 2024: 81, 2025: 100},
        'ATL': {2020: 94, 2021: 103, 2022: 104, 2023: 111, 2024: 105, 2025: 104},
        'BAL': {2020: 118, 2021: 124, 2022: 108, 2023: 104, 2024: 93, 2025: 105},
        'BOS': {2020: 92, 2021: 91, 2022: 99, 2023: 99, 2024: 98, 2025: 89},
        'CHC': {2020: 92, 2021: 104, 2022: 103, 2023: 106, 2024: 93, 2025: 99},
        'CHW': {2020: 107, 2021: 113, 2022: 113, 2023: 104, 2024: 98, 2025: 96},
        'CIN': {2020: 106, 2021: 126, 2022: 139, 2023: 131, 2024: 128, 2025: 123},
        'CLE': {2020: 101, 2021: 102, 2022: 98, 2023: 89, 2024: 86, 2025: 85},
        'COL': {2020: 118, 2021: 112, 2022: 108, 2023: 108, 2024: 109, 2025: 106},
        'DET': {2020: 93, 2021: 84, 2022: 76, 2023: 79, 2024: 86, 2025: 99},
        'HOU': {2020: 106, 2021: 106, 2022: 100, 2023: 101, 2024: 107, 2025: 105},
        'KCR': {2020: 80, 2021: 76, 2022: 80, 2023: 84, 2024: 85, 2025: 85},
        'LAA': {2020: 112, 2021: 107, 2022: 114, 2023: 112, 2024: 112, 2025: 113},
        'LAD': {2020: 119, 2021: 124, 2022: 126, 2023: 122, 2024: 122, 2025: 127},
        'MIA': {2020: 74, 2021: 81, 2022: 84, 2023: 86, 2024: 92, 2025: 90},
        'MIL': {2020: 110, 2021: 105, 2022: 110, 2023: 109, 2024: 114, 2025: 106},
        'MIN': {2020: 93, 2021: 99, 2022: 100, 2023: 104, 2024: 105, 2025: 102},
        'NYM': {2020: 105, 2021: 102, 2022: 95, 2023: 95, 2024: 99, 2025: 104},
        'NYY': {2020: 115, 2021: 111, 2022: 115, 2023: 116, 2024: 119, 2025: 119},
        'PHI': {2020: 116, 2021: 114, 2022: 113, 2023: 113, 2024: 115, 2025: 114},
        'PIT': {2020: 84, 2021: 81, 2022: 84, 2023: 80, 2024: 84, 2025: 76},
        'SDP': {2020: 99, 2021: 98, 2022: 100, 2023: 93, 2024: 104, 2025: 102},
        'SFG': {2020: 74, 2021: 79, 2022: 83, 2023: 84, 2024: 78, 2025: 82},
        'SEA': {2020: 100, 2021: 99, 2022: 100, 2023: 98, 2024: 96, 2025: 93},
        'STL': {2020: 86, 2021: 81, 2022: 84, 2023: 90, 2024: 91, 2025: 87},
        'TBR': {2020: 83, 2021: 84, 2022: 87, 2023: 94, 2024: 98, 2025: 100},
        'TEX': {2020: 100, 2021: 100, 2022: 101, 2023: 111, 2024: 115, 2025: 104},
        'TOR': {2020: 117, 2021: 117, 2022: 111, 2023: 104, 2024: 103, 2025: 104}, 
        'WSN': {2020: 111, 2021: 109, 2022: 108, 2023: 107, 2024: 101, 2025: 94}
    }

    available_teams = sorted(list(ballpark_factors.keys()))

    st.subheader("1. Player Context for 2026")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", min_value=19, max_value=45, value=29)
    with col2:
        pos = st.selectbox("Primary Position", list(alpha_baselines.keys()), index=4)
    with col3:
        selected_team = st.selectbox("Team", available_teams, index=available_teams.index('SEA'))
        park_factor = ballpark_factors[selected_team][2025]
        st.caption(f"Park Factor: {park_factor} (100 is Average)")
    with col4:
        proj_events = st.number_input("2026 Estimated Batted Ball Events", min_value=1, max_value=600, value=300)

    st.subheader("2. Historical Performance")
    
    hist1, hist2, hist3 = st.columns(3)
    with hist1:
        st.markdown("**2025 Outcomes**")
        xhr_lag1 = st.number_input("2025 xHR Count", min_value=0.0, value=25.0, step=0.1, format="%.1f")
        bbe_lag1 = st.number_input("2025 Batted Ball Event Count", min_value=0.0, value=350.0, step=1.0, format="%.0f")
    with hist2:
        st.markdown("**2024 Outcomes**")
        xhr_lag2 = st.number_input("2024 xHR Count", min_value=0.0, value=15.0, step=0.1, format="%.1f")
        bbe_lag2 = st.number_input("2024 Batted Ball Event Count", min_value=0.0, value=250.0, step=1.0, format="%.0f")
    with hist3:
        st.markdown("**2023 Outcomes**")
        xhr_lag3 = st.number_input("2023 xHR Count", min_value=0.0, value=5.0, step=0.1, format="%.1f")
        bbe_lag3 = st.number_input("2023 Batted Ball Event Count", min_value=0.0, value=100.0, step=1.0, format="%.0f")

    alpha_k = alpha_baselines[pos]    
    p_hat_1 = (xhr_lag1 + alpha_k * M_mean) / (bbe_lag1 + M_mean) if bbe_lag1 > 0 else alpha_k
    p_hat_2 = (xhr_lag2 + alpha_k * M_mean) / (bbe_lag2 + M_mean) if bbe_lag2 > 0 else alpha_k
    p_hat_3 = (xhr_lag3 + alpha_k * M_mean) / (bbe_lag3 + M_mean) if bbe_lag3 > 0 else alpha_k
    p_hist = (beta_weights[0] * p_hat_1) + (beta_weights[1] * p_hat_2) + (beta_weights[2] * p_hat_3)
    age_effect = age_effects_dict[pos][str(age)] 
    theta_logit = logit(p_hist) + age_effect
    theta_prob = expit(theta_logit)    
    park_adj_prob = theta_prob * ((park_factor / 100) + 1) / 2
    expected_hrs = proj_events * park_adj_prob

    st.subheader("Projected Outcome")
    
    res1, res2, res3 = st.columns(3)
    res1.metric(f"2026 Projected HRs in {selected_team}", f"{expected_hrs:.1f}")
    res2.metric("Estimated True Talent xHR%", f"{(theta_prob * 100):.1f}%")
    res3.metric(f"Global {pos} Baseline xHR%", f"{(alpha_k * 100):.1f}%")    

# PLAYER LEADERBOARD
elif page == "Player HR Leaderboard":
    st.title("Projected HR Leaders for 2026")
    st.markdown("""A leaderboard of the projected HR leaders for 2026. Includes every player that is projected to play in 2026, according to DepthCharts by FanGraphs\
                (also where the Proj. PAs is taken from).\\
                Note: the Projected HRs shown represents the posterior mean of each player's distribution. \
                The 2.5th percentile and 97.5th percentile of their posterior distribution is also included.""")
    
    display_df = df[['name', 'age', 'team', 'position', 'mean_hr', 'hdi_lower', 'hdi_upper', 'projected_pa']].copy()
    display_df = display_df.sort_values(by='mean_hr', ascending=False).reset_index(drop=True)
    display_df.index += 1
    
    st.dataframe(display_df, column_config={"name": "Player", 
                                            "age": st.column_config.NumberColumn("Age", format="%d"),
                                            "team": "Team",
                                            "position": "Pos",
                                            "mean_hr": st.column_config.ProgressColumn("Projected HRs", format="%.1f", min_value=0, max_value=60),
                                            "hdi_lower": "95% Floor",
                                            "hdi_upper": "95% Ceiling",
                                            "projected_pa": "Proj. PAs"
                                            }, use_container_width=True, height=600)

# TEAM LEADERBOARD
elif page == "Team HR Leaderboard":
    st.title("Projected Team HR Leaders for 2026")
    st.markdown("""A leaderboard of the projected HR leaders for 2026. Estimated by aggregating all 8,000 simulated samples for each individual player, \
                to estimate the 95% highest dentisy interval. Contains a boxplot distribution of the top 10 teams in projected HRs and also the posterior means, \
                and the 2.5th and 97th percentile for every team.""")
    
    st.subheader("Top 10 Team Leaders in Projected HRs for 2026")
    top_10_teams = team_df.head(10)
    
    fig = go.Figure()
    for _, row in top_10_teams.iterrows():
        team = row['team']
        sims = row['team_sims']
        fill_color = MLB_COLORS.get(team, '#1E90FF')
        
        fig.add_trace(go.Box(x=sims, name=team, marker_color=fill_color, boxpoints=False, line=dict(width=2)))

    fig.update_layout(xaxis_title="Projected # of HRs", 
                      yaxis_title="Team", 
                      yaxis=dict(autorange="reversed"), 
                      template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white", 
                      showlegend=False, 
                      height=500
                      )

    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Full League Leaderboard")
    display_team_df = team_df[['team', 'mean_hr', 'hdi_lower', 'hdi_upper']].copy()
    st.dataframe(display_team_df, column_config={"team": "MLB Team", 
                                                 "mean_hr": st.column_config.ProgressColumn("Projected HRs", format="%.1f", min_value=100, max_value=300),
                                                 "hdi_lower": "95% Projected Floor", 
                                                 "hdi_upper": "95% Projected Ceiling"
                                                 }, use_container_width=True, height=600
                                                 )

# METHODOLOGY
elif page == "Methodology":
    st.title("Methodology")
    st.markdown(r"""
    ### xHR Estimation
    Rather than modeling observed HR counts directly, the first step in our framework was to construct an xHR total for each player-season as the basis of our projection model and what we model the true talent of. Observed HR counts are subject to a variety of confounding factors that are outside of a hitter's control (e.g., ballpark dimensions, weather, batted ball luck), making them a noisy signal of true talent. By grounding our outcome variable in the physics of contact quality (i.e., exit velocity and launch angle), xHR reflects what a batted ball *should* have produced in expectation, independent of any context that the hitter cannot influence.[^1] 

    To estimate xHR, we model the probability of a batted ball resulting in a HR as a function of exit velocity and launch angle across roughly 900,000 batted ball events from 2018 to 2025. Given the smooth, nonlinear relationship between these features and HR probability, a Gaussian Process (GP) is a natural modeling choice, as the observed data can be modeled in a flexible way by generating any higher-order transformations (polynomials) of our features to capture potential nonlinearities (Andorra and Göbel, 2024). However, GPs are notably bottlenecked by its computational expense, $\mathcal{O}(N^3)$, where $N$ denotes the number of data points.[^2]

    We pivot to model xHR with a *Hilbert Space* Gaussian Process (HSGP) instead. HSGPs work quite well in our case because they reduce the computational cost of GPs cost substantially, $\mathcal{O}(mN)$[^3], by projecting the GP onto a set of basis functions (Orduz, 2024). This allows the model to scale linearly with the number of observations, making it feasible for the large number of data points we have. We model our HSGP with the Matérn 5/2 kernel as our covariance function[^4] (Engels and Andorra, 2024). We chose to combine the HSGP model with Stochastic Variational Inference (SVI) for posterior estimation rather than MCMC, thereby yielding *approximate* rather than *exact* posterior estimates (Orduz, 2025), a tradeoff we accept in favor of scalability.

    ### Bayesian Hierarchical Framework
    Our outcome of interest for a given player $i$ in a given season $j$ is their HR total $Y_{ij}$, which we model as a binomial process, however, with a few distinctions from Jensen's framework (Jensen et al., 2009). We model $Y_{ij}$ as:

    $$Y_{ij} \sim \text{Binomial}\biggl(N_{ij}, \theta_{ij}\times \biggl(\frac{BF_{ij} + 1.00}{2}\biggr)\biggr),$$

    where $\theta_{ij}$ is a player- and season-specific *expected* home run rate, $N_{ij}$ are the number of batted balls player $i$ had in season $j$, and $(BF_{ij} + 1.00) / 2$ is the adjusted ball park factor for player $i$'s team in season $j$. Since players only play roughly half their games in their home stadium, we average each team's ballpark factor with $1.00$—the league-neutral baseline—to yield the adjusted factor of BF that appropriately reduces the park effect for a full season. Because xHRs were modeled *solely* on batted ball events, we must condition on batted ball events in the binomial process as well. The iid assumption of the binomial model was justified in Jensen's paper (Jensen et al., 2009). Let $\theta_{ij}\times \biggl(\frac{BF_{ij} + 1.00}{2}\biggr)$ be denoted as $\tilde{\theta}_{ij}$ for notation purposes.

    We model $\theta_{ij}$ as a function of age $A_{ij}$, position $k = K_{ij}$, and the player's historical performance in the three seasons preceding season $j$ for a given player $i$:

    $$\theta_{ij} = \text{logit}^{-1}\biggl(\text{logit}\biggl(\sum_{\ell = 1}^3 \beta_{\ell}\hat{p}_{i, j-\ell}\biggr) + f_k(A_{ij})\biggr),$$

    where $\ell\in\{1, 2, 3\}$ indexes lagged seasons relative to season $j$. $\hat{p}_{i, j-\ell}$ represents the empirical partial pooling estimator that shrinks each player's observed xHR rate toward the positional mean $\alpha_k$ for season $j-\ell$, where the degree of shrinkage is governed by the parameter $M$. $\hat{p}_{i, j-\ell}$ is modeled explicitly as:

    $$\hat{p}_{i, j-\ell} = \frac{(xY_{i, j-\ell}\times N_{i, j-\ell)} + (\alpha_k\times M)}{N_{i, j-\ell} + M}.$$

    $M$ is our shrinkage parameter that effectively controls the rate at which a player's observed xHR rate is trusted over the positional prior (i.e., regression to the mean[^5]). Following the split-half reliability method of (Staude, 2013), we yielded $M = 75$ BIPs to be the point at which the correlation between first- and second-half xHR rates first exceeded $r=0.50$. We model $M$ as a stochastic parameter rather than a fixed one to acknowledge uncertainty in the stabilization point itself, as the split-half method yields an empirical estimate rather than a known truth. We use $M=75$ as an informed center, such that:

    $$M \sim \text{LogNormal}(\ln 75, 0.25).$$

    The LogNormal distribution constrains $M$ to be strictly positive while permitting *some* deviation from $M=75$. 

    $\alpha_k$ represents the baseline xHR rate for position $k$, and is itself modeled hierarchically as:

    $$\alpha_k = \text{logit}^{-1}(z_k),\quad z_k \sim \mathcal{N}(\mu, \sigma),\quad \mu \sim \mathcal{N}(-3, 0.75), \quad \sigma \sim \text{HalfNormal}(0.5).$$

    Rather than estimating a separate, independent baseline for each position, the hierarchical structure partially shares information across positions, meaning positions with limited data[^6] are shrunk towards the global mean, $\mu$, while positions with more data are allowed to deviate more freely. The prior on $\mu$ is centered at $-3$ on the log-odds scale, which is roughly consistent with the empirical league-wide xHR rate observed in our dataset. 

    Referring back to the $\theta_{ij}$ equation, the weights $\beta_\ell = (\beta_1, \beta_2, \beta_3)$ govern the relative contribution of each lagged season, and are modeled as:

    $$\beta \sim \text{Dirichlet}(1, 3, 6).$$

    The Dirichlet prior places the most weight on the most recent season with the concentration parameters of (1, 3, 6), implying that season $j-1$ *should* carry approximately $60\%$ of the total weight on average.[^7] 

    And we continue with Jensen's implementation by using a cubic B-spline to model $f_k(A_{ij})$ (Jensen et al., 2009). This allows $A_{ij}$ to be modeled flexibly for each position $k$. We make ours a little more flexible by using six B-spline basis functions per position, totaling $54$ parameters (versus Jensen's $36$), where each B-spline coefficient, $\gamma_k$, is modeled as:

    $$\gamma_{kc} \sim \mathcal{N}(0, 0.375), \quad \forall k = 1, \dots, 9, c = 1\dots, 6,$$

    where $c=1,\dots,6$ indexes the B-spline basis functions for the age trajectory of position $k$.

    We specify informative priors throughout the framework to encode a degree of inductive bias into the model, centering hyperparameters on intuitive baselines for player-level HR production. We did so to ensure that the resulting projections remain grounded in the physical realities of the sport. The joint posterior distribution of our unknown parameters, incorporating the prior specifications from the equations above, is defined as:

    $$
    \begin{aligned}
    p(\boldsymbol{\alpha}, \boldsymbol{\beta}, \boldsymbol{\gamma}, M, \mu, \sigma \mid \mathbf{X}) \propto \prod_{i, j} 
    &p(Y_{i, j} \mid N_{i, j}, \tilde{\theta}_{i, j})\cdot p(\theta_{ij} \mid K_{ij}, A_{ij}, \boldsymbol{\alpha}, \boldsymbol{\beta}, \boldsymbol{\gamma}, M) \\
    &\cdot p(\boldsymbol{\alpha} \mid \mu, \sigma)\cdot p(\boldsymbol{\beta})\cdot p(\boldsymbol{\gamma})\cdot p(M)\cdot p(\mu)\cdot p(\sigma),
    \end{aligned}
    $$
    
    where $\mathbf{X}$ contains the observed data $\mathbf{Y}$ and covariates ($\mathbf{A}, \mathbf{BF}, \mathbf{K}, \mathbf{N}$) (Jensen et al., 2009). We estimate the posterior distribution with MCMC, specifically NumPyro's No U-Turn Sampler (NUTS) implementation (NumPyro, 2019). 

    We ran NUTS with four chains of $3000$ samples each, discarding $1000$ warmup samples, yielding $8000$ total posterior draws. Convergence was assessed via the Gelman-Rubin statistic $\hat{R}$, with all parameters achieving $\hat{R} < 1.01$, and effective sample sizes (ESS) were sufficient across all parameters (PyMC, 2014).
    
    If you're interested in the code, feel free to check it out [here](https://github.com/d2osborn/probabilistic_projection_project)!

    [^1]: E.g., a ball struck at 105 mph with a 20° launch angle to straightaway centerfield carries a much lower HR probability than identically-struck ball pulled toward the foul line. This distinction, however, which was driven entirely by context and not the quality of the contact.
    [^2]: The estimation and prediction process for GPs requires the inversion of the kernel matrix (Andorra and Göbel, 2024).
    [^3]: Where $m$ is the number of basis functions used in the approximation.
    [^4]: We evaluated both the squared exponential and Matérn 5/2 kernel functions, selecting the latter on the basis of a higher out-of-sample Expected Log Pointwise Predictive Density (ELPD) score.
    [^5]: $\hat{p}_{i, j-\ell}$ is a rearrangement of the general equation for estimating the group level mean in a partially pooled model, where $M$ represents the ratio of variances in the equation $\frac{\sigma_y^2}{\sigma_\alpha^2}$ (Kumar, 2024).
    [^6]: i.e., DHs.
    [^7]: This reflects the intuition that recent performance is the strongest predictor of future outcomes, with older seasons contributing progressively less.
    
    ---
    * Alexandre Andorra and Maximilian Göbel. Unveiling true talent: The soccer factor model for skill evaluation. arXiv (Cornell University), 2024. doi: https://doi.org/10.48550/arXiv.2412.05911.            
    * Bill Engels and Alexandre Andorra. Park factors leaderboard, 2024. URL https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Basic.html.
    * Shane T. Jensen, Blakeley B. McShane, and Abraham J. Wyner. Hierarchical bayesian modeling of hitting performance in baseball. Bayesian Analysis, 4(4), 2009. doi: https://doi.org/10.1214/09-BA424.
    * Sayam Kumar. A gentle introduction to partial pooled models, 2024. URL https://www.codingpaths.com/bayesian/partial_pooling_demonstration/#partial-pooled-model.
    * NumPyro. Nuts, 2019. URL https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.NUTS.
    * Juan Orduz. A conceptual and practical introduction to hilbert space gps approximation methods, 2024. URL https://juanitorduz.github.io/hsgp_intro/.
    * Juan Orduz. Pydata berlin 2025: Introduction to stochastic variational inference with numpyro, 2025. URL https://juanitorduz.github.io/intro_svi/.
    * PyMC. Model checking and diagnostics, 2014. URL https://pymcmc.readthedocs.io/en/latest/modelchecking.html#formal-methods.
    * Steve Staude. Randomness, stabilization, and regression, 2013. URL https://blogs.fangraphs.com/randomness-stabilization-regression/.
    """)

# ABOUT ME
elif page == "About":
    st.title("About Me")
    st.write("Hi there! My name is Diego Osborn, and I'm a fourth-year Data Science student at UC San Diego.")
    st.write("A bit about me: I've been working with my school's baseball team since 2023, and I've become especially interested in \
             probabilistic thinking and modeling through the years. \
             I'm highly interested in long-term, research-focused work in baseball, with a focus on leveraging Bayesian statistics \
             to enhance player projection systems across all levels of competition. \
             This project is my first step toward using advanced statistical methods to better understand and capture variability in player performance!")
    st.write("Feel free to explore my socials and portfolio. I'd love to hear any thoughts or questions you may have!")
    st.markdown("""
    * [My Portfolio](https://d2osborn.github.io/portfolio/)
    * [My GitHub](https://github.com/d2osborn)
    * [My LinkedIn](https://www.linkedin.com/in/diego-osborn/)
    * [My Twitter](https://x.com/diegosborn)
    * [My BlueSky](https://bsky.app/profile/diegosborn.bsky.social)
    """)