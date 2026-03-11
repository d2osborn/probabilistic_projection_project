import streamlit as st
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. PAGE CONFIG & DATA LOADING
# ==========================================
st.set_page_config(page_title="BayesBall", layout="wide")

@st.cache_data
def load_player_data():
    # Placeholder for data_pred.csv equivalent
    hitters = pd.DataFrame({"Name": ["Mike Trout", "Aaron Judge", "Mookie Betts"]})
    pitchers = pd.DataFrame({"Name": ["Gerrit Cole", "Max Fried", "Zack Wheeler"]})
    return hitters, pitchers

@st.cache_resource
def load_trace():
    # Placeholder: az.from_netcdf("model_artifacts/trace.nc")
    return None 

hitters_df, pitchers_df = load_player_data()
trace = load_trace()

# ==========================================
# 2. SIDEBAR UI (Replicating fluidPage & conditionalPanel)
# ==========================================
with st.sidebar:
    st.title("BayesBall Settings")
    
    pos = st.radio("Hitters/Pitchers:", ["Hitters", "Pitchers"])
    prorated = st.radio("Prorated (600 PA):", ["No", "Yes"])
    comp = st.radio("Select Baseline:", ["None", "Display League Average", "Compare Two Players"])
    
    # Conditional logic replaces conditionalPanel()
    if pos == "Hitters":
        player1 = st.selectbox("Player 1:", hitters_df["Name"])
        stat = st.selectbox("Statistic:", ['PA', 'AB', 'H', 'HR', 'wOBA', 'BatWAR', 'Sharpe'])
        
        if comp == "Compare Two Players":
            player2 = st.selectbox("Player 2:", hitters_df["Name"], index=1)
            
    else:
        player1 = st.selectbox("Player 1:", pitchers_df["Name"])
        stat = st.selectbox("Statistic:", ['W', 'ERA', 'IP', 'K', 'BB', 'FIP', 'WAR', 'Sharpe'])
        
        if comp == "Compare Two Players":
            player2 = st.selectbox("Player 2:", pitchers_df["Name"], index=1)

# ==========================================
# 3. HELPER FUNCTIONS (Replicating the Server Math)
# ==========================================
def generate_posterior_predictive(player_name, stat_name, prorate_flag):
    # This is where you replace the heavy R matrix math:
    # loglam = mod_csim[,1] + mod_csim[,9:18] %*% x
    # lam = exp(loglam) ... etc.
    
    # In PyMC, you'd extract the samples for this specific player:
    # player_idx = get_player_idx(player_name)
    # samples = trace.posterior[stat_name].sel(player=player_idx).values.flatten()
    
    # Mocking the output for the sake of the template
    mean_val = np.random.uniform(0.300, 0.400) if stat_name == "wOBA" else np.random.uniform(10, 40)
    samples = np.random.normal(loc=mean_val, scale=mean_val*0.1, size=3000)
    return samples

# ==========================================
# 4. MAIN PANEL TABS (Replicating tabsetPanel)
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["Projections", "Leaderboard", "Methodology", "About"])

with tab1:
    st.markdown("<h2 style='text-align: center;'>Posterior Predictive Distribution</h2>", unsafe_allow_html=True)
    
    # Generate data
    p1_samples = generate_posterior_predictive(player1, stat, prorated)
    
    # Plotting logic (Replicating ggplot2 overlapping densities)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.kdeplot(p1_samples, fill=True, alpha=0.4, label=player1, color='blue', ax=ax)
    ax.axvline(np.mean(p1_samples), color='blue', linestyle='--', linewidth=2)
    
    if comp == "Compare Two Players":
        p2_samples = generate_posterior_predictive(player2, stat, prorated)
        sns.kdeplot(p2_samples, fill=True, alpha=0.4, label=player2, color='red', ax=ax)
        ax.axvline(np.mean(p2_samples), color='red', linestyle='--', linewidth=2)
        
        # Calculate probability P1 > P2
        prob_greater = np.mean(p1_samples > p2_samples)
        st.write(f"**Probability {player1} > {player2}:** {prob_greater:.2%}")
        
    elif comp == "Display League Average":
        league_samples = generate_posterior_predictive("League", stat, prorated)
        sns.kdeplot(league_samples, fill=True, alpha=0.4, label="League Average", color='grey', ax=ax)
        ax.axvline(np.mean(league_samples), color='grey', linestyle='--', linewidth=2)

    ax.set_title(f"Kernel Density Plot of Projected {stat}")
    ax.set_xlabel(stat)
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)
    
    # Data Table (Replicating DT::dataTableOutput)
    st.markdown("<h3 style='text-align: center;'>Summary Statistics</h3>", unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        "Metric": ["Mean", "Std Dev", "5th Percentile", "95th Percentile"],
        player1: [np.mean(p1_samples), np.std(p1_samples), np.percentile(p1_samples, 5), np.percentile(p1_samples, 95)]
    })
    st.dataframe(summary_df, use_container_width=True)

with tab2:
    st.markdown("<h2 style='text-align: center;'>Leaderboard</h2>", unsafe_allow_html=True)
    sort_by = st.radio("Sort By:", ["Upside (95th %ile)", "Bust Potential (5th %ile)", "High Variance", "Expected WAR"], horizontal=True)
    
    # Placeholder for leaderboard dataframe sorting
    st.write(f"Displaying leaderboard sorted by: **{sort_by}**")
    st.dataframe(pd.DataFrame({"Player": ["Player A", "Player B"], "WAR": [4.5, 3.2]})) # Mock data

with tab3:
    st.header("Methodology")
    st.markdown("""
    ### Bayesian Hierarchical Projection in Major League Baseball
    
    **1. Introduction**
    Public sabermetric research has largely focused on making precise predictions of future performance but has ignored the question of uncertainty...
    
    *(You can simply drop standard Markdown text here, avoiding the need for Shiny's tedious `tags$p()`, `tags$h2()`, etc.)*
    """)

with tab4:
    st.header("About Me")
    st.markdown("""
    * **Name:** Your Name
    * **Focus:** Empirical modeling of baseball performance using Bayesian statistics.
    """)