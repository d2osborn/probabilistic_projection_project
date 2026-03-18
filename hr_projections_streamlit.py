import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# --- 1. Page Config ---
st.set_page_config(page_title="Bayesian HR Projections", layout="wide", page_icon="⚾")
blue_color = '#1E90FF'
red_color = '#EF3E42'

# --- 2. Load Data ---
@st.cache_data
def load_data():
    df = pd.read_parquet('data/2026_hr_posterior_samples.parquet')
    
    # Pre-calculate summary stats for the leaderboard so it's snappy
    df['mean_hr'] = df['simulated_hrs'].apply(np.mean).round(1)
    df['hdi_lower'] = df['simulated_hrs'].apply(lambda x: np.percentile(x, 2.5)).astype(int)
    df['hdi_upper'] = df['simulated_hrs'].apply(lambda x: np.percentile(x, 97.5)).astype(int)
    return df

df = load_data()

# --- 3. Sidebar Navigation & Filters ---
st.sidebar.title("⚾ BayesBall Projections")
page = st.sidebar.radio("Navigation", ["Distributions", "Leaderboard", "Methodology", "About Me"])

st.sidebar.markdown("---")

# --- PAGE: DISTRIBUTIONS ---
if page == "Distributions":
    st.title("Player Posterior Distributions")
    st.markdown("Explore the full range of projected outcomes for 2026 based on our Bayesian hierarchical model.")

    # Front-Office Level Filtering
    col1, col2 = st.columns(2)
    with col1:
        team_filter = st.selectbox("Filter by Team", ["All Teams"] + sorted(df['team'].dropna().unique().tolist()))
    with col2:
        pos_filter = st.selectbox("Filter by Position", ["All Positions"] + sorted(df['position'].dropna().unique().tolist()))

    # Apply Filters
    filtered_df = df.copy()
    if team_filter != "All Teams":
        filtered_df = filtered_df[filtered_df['team'] == team_filter]
    if pos_filter != "All Positions":
        filtered_df = filtered_df[filtered_df['position'] == pos_filter]

    selected_players = st.multiselect(
        "Select Players to Compare:",
        options=filtered_df['name'].tolist(),
        default=[filtered_df['name'].iloc[0]] if not filtered_df.empty else []
    )

    if selected_players:
        # Build the Plotly Chart
        fig = go.Figure()
        
        for player in selected_players:
            player_data = df[df['name'] == player].iloc[0]
            sims = player_data['simulated_hrs']
            
            # Calculate KDE for smooth Plotly curves
            kde = gaussian_kde(sims, bw_method=0.3)
            x_vals = np.linspace(0, max(sims) + 5, 200)
            y_vals = kde(x_vals)

            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, 
                fill='tozeroy', 
                mode='lines',
                name=player,
                hovertemplate="<b>HRs:</b> %{x:.0f}<br><b>Density:</b> %{y:.3f}<extra></extra>"
            ))

        fig.update_layout(
            title="Projected Home Run Density",
            xaxis_title="Simulated Home Runs",
            yaxis_title="Probability Density",
            hovermode="x unified",
            template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Dynamic Player Context
        st.markdown("### Player Insights")
        cols = st.columns(len(selected_players))
        
        for i, player in enumerate(selected_players):
            with cols[i]:
                p_data = df[df['name'] == player].iloc[0]
                # Updated info box to include Age!
                st.info(f"**{p_data['name']}** (Age: {p_data['age']} | {p_data['position']} - {p_data['team']})")
                
                # Dynamic text generation based on the shape of the distribution
                spread = p_data['hdi_upper'] - p_data['hdi_lower']
                confidence = "narrow" if spread < 10 else "wide"
                
                st.metric("Expected HRs (Mean)", f"{p_data['mean_hr']}")
                st.write(f"**95% Interval:** {p_data['hdi_lower']} to {p_data['hdi_upper']} HRs")
                st.write(f"**Projected PAs:** {p_data['projected_pa']}")
                st.caption(f"The model has a {confidence} range of outcomes for {player}, spanning a {spread}-HR gap between his 2.5th and 97.5th percentiles.")

# --- PAGE: LEADERBOARD ---
elif page == "Leaderboard":
    st.title("2026 Projected HR Leaderboard")
    
    # Formatting the dataframe for a clean UI - Now including Age!
    display_df = df[['name', 'age', 'team', 'position', 'mean_hr', 'hdi_lower', 'hdi_upper', 'projected_pa']].copy()
    display_df = display_df.sort_values(by='mean_hr', ascending=False).reset_index(drop=True)
    display_df.index += 1 # Make index 1-based
    
    st.dataframe(
        display_df,
        column_config={
            "name": "Player",
            "age": st.column_config.NumberColumn("Age", format="%d"), # Ensures no weird commas like "2,024"
            "team": "Team",
            "position": "Pos",
            "mean_hr": st.column_config.ProgressColumn("Projected HRs", format="%.1f", min_value=0, max_value=60),
            "hdi_lower": "95% Floor",
            "hdi_upper": "95% Ceiling",
            "projected_pa": "Proj. PAs"
        },
        use_container_width=True,
        height=600
    )

# --- PAGE: METHODOLOGY ---
elif page == "Methodology":
    st.title("Model Methodology")
    st.markdown("""
    ### The Engine: Bayesian Hierarchical Modeling
    This projection system doesn't just look at a player's past; it looks at how similar players age and regress. 
    
    * **Shrinkage (Partial Pooling):** Rookies and small-sample players are regressed aggressively toward a global positional mean. 
    * **Aging Curves:** We use a cubic B-spline to model how power peaks and declines.
    * **Time-Series Weights:** A Dirichlet prior weights the last three seasons (Lag 1, Lag 2, Lag 3) to capture recent performance trends.
    
    By simulating 8,000 parallel realities using a NUTS (No-U-Turn Sampler), we don't just get a single HR guess—we map out the entire landscape of what is statistically possible for a player next year.
    """)

# --- PAGE: ABOUT ME ---
elif page == "About Me":
    st.title("About the Author")
    st.write("I am a Data Science student building applied probabilistic models for professional sports...")
    # Add your links
    st.markdown("[GitHub](https://github.com/yourprofile) | [LinkedIn](https://linkedin.com/in/yourprofile)")