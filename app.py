# In app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Import the classes I built in my other files.
from simulation import AdCreative, AdEnvironment
from bandit import ThompsonSamplingBandit

# --- App Configuration ---
# Wide layout for my dashboards.
st.set_page_config(layout="wide", page_title="Ad Creative Optimization")

# --- Main App Title ---
st.title("🚀 Real-Time Ad Optimization: Multi-Armed Bandits vs. A/B Testing")
st.markdown("""
This dashboard simulates a real-time ad campaign to compare two optimization strategies.
* **A/B Test:** A traditional approach that splits traffic evenly.
* **Thompson Sampling Bandit:** A reinforcement learning algorithm that learns and adapts in real-time.
Which strategy will generate more clicks? Let's find out!
""")

# --- Sidebar for Controls ---
# Places controls in the sidebar to keep the main view clean.
st.sidebar.header("Simulation Settings")
num_impressions = st.sidebar.slider(
    "Number of Impressions (Users)",
    min_value=100,
    max_value=20000,
    value=5000,
    step=100
)

# This is the "ground truth" of the simulation.
# The algorithm will not know these true CTRs.
creatives_config = [
    {'id': 1, 'name': 'Ad A (Low CTR)', 'true_ctr': 0.015},
    {'id': 2, 'name': 'Ad B (High CTR)', 'true_ctr': 0.021},
    {'id': 3, 'name': 'Ad C (Medium CTR)', 'true_ctr': 0.018}
]
ad_creatives = [AdCreative(c['id'], c['true_ctr']) for c in creatives_config]
ad_names = {c['id']: c['name'] for c in creatives_config}


# --- Main Simulation Logic ---
# Function to run the simulation.
def run_simulation(creatives, total_impressions):
    # Initialize environment, bandit, and tracking variables
    environment = AdEnvironment(creatives)
    bandit = ThompsonSamplingBandit(creatives)

    # Variables for Bandit results
    bandit_impressions = {ad.creative_id: 0 for ad in creatives}
    bandit_clicks = {ad.creative_id: 0 for ad in creatives}
    bandit_history = []

    # Variables for A/B Test results
    ab_impressions = {ad.creative_id: 0 for ad in creatives}
    ab_clicks = {ad.creative_id: 0 for ad in creatives}
    ab_history = []

    num_creatives = len(creatives)

    # The main loop where each iteration is one user impression
    for i in range(total_impressions):
        # --- Bandit's Turn ---
        chosen_ad_bandit = bandit.select_ad()
        reward_bandit = chosen_ad_bandit.show_ad()
        bandit.update(chosen_ad_bandit.creative_id, reward_bandit)

        # Track bandit results
        bandit_impressions[chosen_ad_bandit.creative_id] += 1
        bandit_clicks[chosen_ad_bandit.creative_id] += 1
        bandit_history.append(sum(bandit_clicks.values()))

        # --- A/B Test's Turn ---
        # Cycle through ads evenly
        ad_to_show_ab = creatives[i % num_creatives]
        reward_ab = ad_to_show_ab.show_ad()

        # Track A/B test results
        ab_impressions[ad_to_show_ab.creative_id] += 1
        ab_clicks[ad_to_show_ab.creative_id] += 1
        ab_history.append(sum(ab_clicks.values()))

    return bandit_impressions, bandit_clicks, bandit_history, ab_impressions, ab_clicks, ab_history


# --- App Layout ---
# Only want to run the simulation when the user clicks the button.
if st.sidebar.button("Run Simulation"):

    # Call main simulation function
    b_impressions, b_clicks, b_history, ab_impressions, ab_clicks, ab_history = run_simulation(ad_creatives,
                                                                                               num_impressions)

    st.header("📈 Performance Results")
    st.markdown("---")

    # --- Key Metrics ---
    # Use columns to display the final KPIs side-by-side.
    total_bandit_clicks = sum(b_clicks.values())
    total_ab_clicks = sum(ab_clicks.values())
    bandit_ctr = total_bandit_clicks / num_impressions
    ab_ctr = total_ab_clicks / num_impressions

    col1, col2, col3 = st.columns(3)
    col1.metric("Bandit Total Clicks", f"{total_bandit_clicks}",
                f"{((total_bandit_clicks - total_ab_clicks) / total_ab_clicks):.2%} vs A/B")
    col2.metric("A/B Test Total Clicks", f"{total_ab_clicks}")
    col3.metric("Bandit Final CTR", f"{bandit_ctr:.4f}")

    # --- The Main Performance Chart ---
    st.subheader("Cumulative Clicks Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(b_history, label="Thompson Sampling Bandit", color='orange')
    ax.plot(ab_history, label="Traditional A/B Test", color='blue', linestyle='--')
    ax.set_title("Bandit vs. A/B Test: Cumulative Clicks")
    ax.set_xlabel("Impressions Shown")
    ax.set_ylabel("Total Clicks")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.markdown(
        "*The growing gap between the lines represents the **extra clicks gained** by using the smarter bandit algorithm.*")

    st.markdown("---")
    st.header("⚙️ How the Bandit Learned")

    # --- Behind the Scenes Charts ---
    col_dist, col_impressions = st.columns(2)

    with col_dist:
        st.subheader("Final Ad CTR Beliefs (Beta Distributions)")
        fig_dist, ax_dist = plt.subplots()
        x = np.linspace(0, 0.05, 200)
        for ad in ad_creatives:
            dist = beta(b_clicks[ad.creative_id] + 1, b_impressions[ad.creative_id] - b_clicks[ad.creative_id] + 1)
            ax_dist.plot(x, dist.pdf(x), label=ad_names[ad.creative_id])
        ax_dist.set_title("Bandit's Beliefs About Each Ad's CTR")
        ax_dist.set_xlabel("Click-Through Rate")
        ax_dist.set_ylabel("Probability Density")
        ax_dist.legend()
        st.pyplot(fig_dist)

    with col_impressions:
        st.subheader("Impressions per Ad Creative")
        impressions_df = pd.DataFrame({
            "Ad Creative": [ad_names[ad_id] for ad_id in b_impressions.keys()],
            "Impressions": list(b_impressions.values())
        })
        st.bar_chart(impressions_df.set_index("Ad Creative"))
        st.markdown(
            "*Notice how the bandit correctly allocated the most impressions to **Ad B**, the best-performing ad.*")