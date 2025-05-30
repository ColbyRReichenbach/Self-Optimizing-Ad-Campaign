# In app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# I need to import the classes I built in my other files.
from simulation import AdCreative, AdEnvironment
from bandit import ThompsonSamplingBandit

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="Ad Creative Optimization")

# --- Main App Title ---
st.title("Real-Time Ad Optimization: Multi-Armed Bandits vs. A/B Testing")
st.markdown("""
This dashboard simulates real-time ad campaigns to compare optimization strategies.
It runs multiple simulations to provide a robust view of average performance and variability.
* **A/B Test:** A traditional approach that splits traffic evenly.
* **Thompson Sampling Bandit:** A reinforcement learning algorithm that learns and adapts.
""")

# --- Sidebar for Controls ---
st.sidebar.header("Simulation Settings")
num_impressions = st.sidebar.slider(
    "Number of Impressions (per Simulation)",
    min_value=100,
    max_value=10000,  # Reduced max for faster multiple runs
    value=2000,  # Reduced default for faster initial runs
    step=100
)
num_simulations_to_run = st.sidebar.number_input(
    "Number of Simulations to Run",
    min_value=1,
    max_value=200,  # Cap at 200 to keep it manageable for Streamlit
    value=50,  # Default to 50 runs
    step=10
)

# Define my ad creatives here.
creatives_config = [
    {'id': 1, 'name': 'Ad A (Low CTR)', 'true_ctr': 0.015},
    {'id': 2, 'name': 'Ad B (High CTR)', 'true_ctr': 0.021},
    {'id': 3, 'name': 'Ad C (Medium CTR)', 'true_ctr': 0.018}
]
ad_creatives_list = [AdCreative(c['id'], c['true_ctr']) for c in creatives_config]  # Renamed for clarity
ad_names = {c['id']: c['name'] for c in creatives_config}


# --- Main Simulation Logic (Unchanged from before) ---
def run_simulation(creatives, total_impressions):
    environment = AdEnvironment(creatives)
    bandit = ThompsonSamplingBandit(creatives)

    bandit_impressions = {ad.creative_id: 0 for ad in creatives}
    bandit_clicks = {ad.creative_id: 0 for ad in creatives}
    bandit_history = []

    ab_impressions = {ad.creative_id: 0 for ad in creatives}
    ab_clicks = {ad.creative_id: 0 for ad in creatives}
    ab_history = []

    num_creatives_env = len(creatives)

    for i in range(total_impressions):
        chosen_ad_bandit = bandit.select_ad()
        reward_bandit = chosen_ad_bandit.show_ad()
        bandit.update(chosen_ad_bandit.creative_id, reward_bandit)

        bandit_impressions[chosen_ad_bandit.creative_id] += 1
        bandit_clicks[chosen_ad_bandit.creative_id] += reward_bandit
        bandit_history.append(sum(bandit_clicks.values()))

        ad_to_show_ab = creatives[i % num_creatives_env]
        reward_ab = ad_to_show_ab.show_ad()

        ab_impressions[ad_to_show_ab.creative_id] += 1
        ab_clicks[ad_to_show_ab.creative_id] += reward_ab
        ab_history.append(sum(ab_clicks.values()))

    # Return impressions and clicks for each ad for this single run
    return bandit_impressions, bandit_clicks, bandit_history, ab_impressions, ab_clicks, ab_history


# --- App Layout ---
if st.sidebar.button("Run Multiple Simulations"):

    # Lists to store results from ALL simulations
    all_bandit_histories = []
    all_ab_histories = []

    # Store final clicks and impressions for each ad from ALL simulations
    # This will be a list of dictionaries
    all_bandit_final_impressions = []
    all_bandit_final_clicks = []
    all_ab_final_impressions = []
    all_ab_final_clicks = []

    # Placeholder for the progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # --- Loop for Multiple Simulations ---
    for i in range(num_simulations_to_run):
        status_text.text(f"Running simulation {i + 1} of {num_simulations_to_run}...")
        # Create a fresh list of AdCreative objects for each simulation run
        # This is important if your AdCreative objects have internal state that changes
        current_sim_ad_creatives = [AdCreative(c['id'], c['true_ctr']) for c in creatives_config]

        b_impressions_run, b_clicks_run, b_history_run, \
            ab_impressions_run, ab_clicks_run, ab_history_run = run_simulation(current_sim_ad_creatives,
                                                                               num_impressions)

        all_bandit_histories.append(b_history_run)
        all_ab_histories.append(ab_history_run)

        all_bandit_final_impressions.append(b_impressions_run)
        all_bandit_final_clicks.append(b_clicks_run)
        all_ab_final_impressions.append(ab_impressions_run)
        all_ab_final_clicks.append(ab_clicks_run)

        progress_bar.progress((i + 1) / num_simulations_to_run)

    status_text.text("Simulations complete! Calculating results...")

    # --- Convert histories to NumPy arrays for easy calculation ---
    np_bandit_histories = np.array(all_bandit_histories)
    np_ab_histories = np.array(all_ab_histories)

    # Calculate mean and standard deviation across all simulations
    mean_bandit_clicks_hist = np.mean(np_bandit_histories, axis=0)
    std_bandit_clicks_hist = np.std(np_bandit_histories, axis=0)

    mean_ab_clicks_hist = np.mean(np_ab_histories, axis=0)
    std_ab_clicks_hist = np.std(np_ab_histories, axis=0)

    st.header("📈 Aggregated Performance Results")
    st.markdown(f"*Based on **{num_simulations_to_run}** simulation runs of **{num_impressions}** impressions each.*")
    st.markdown("---")

    # --- NEW: Key Metrics (Averages) ---
    avg_total_bandit_clicks = mean_bandit_clicks_hist[-1]  # Last value is total for one run
    avg_total_ab_clicks = mean_ab_clicks_hist[-1]

    # Calculate average final CTRs
    avg_bandit_ctr = avg_total_bandit_clicks / num_impressions
    avg_ab_ctr = avg_total_ab_clicks / num_impressions

    uplift_percentage = ((
                                     avg_total_bandit_clicks - avg_total_ab_clicks) / avg_total_ab_clicks) * 100 if avg_total_ab_clicks > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg. Bandit Total Clicks", f"{avg_total_bandit_clicks:.2f}", f"{uplift_percentage:.2f}% vs A/B")
    col2.metric("Avg. A/B Test Total Clicks", f"{avg_total_ab_clicks:.2f}")
    col3.metric("Avg. Bandit Final CTR", f"{avg_bandit_ctr:.4f}")

    # --- NEW: The Main Performance Chart with Confidence Intervals ---
    st.subheader("Average Cumulative Clicks Over Time (with +/- 1 Std Dev)")
    fig, ax = plt.subplots(figsize=(12, 6))

    x_impressions = np.arange(num_impressions)  # X-axis for impressions

    # Bandit Plot
    ax.plot(x_impressions, mean_bandit_clicks_hist, label="Avg. Thompson Bandit Clicks", color='orange', linewidth=2)
    ax.fill_between(x_impressions,
                    mean_bandit_clicks_hist - std_bandit_clicks_hist,
                    mean_bandit_clicks_hist + std_bandit_clicks_hist,
                    color='orange', alpha=0.2, label='Bandit +/- 1 Std Dev')

    # A/B Test Plot
    ax.plot(x_impressions, mean_ab_clicks_hist, label="Avg. A/B Test Clicks", color='blue', linestyle='--', linewidth=2)
    ax.fill_between(x_impressions,
                    mean_ab_clicks_hist - std_ab_clicks_hist,
                    mean_ab_clicks_hist + std_ab_clicks_hist,
                    color='blue', alpha=0.1, label='A/B +/- 1 Std Dev')

    ax.set_title("Bandit vs. A/B Test: Average Cumulative Clicks")
    ax.set_xlabel("Impressions Shown")
    ax.set_ylabel("Average Total Clicks")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.markdown(
        "*The shaded areas represent one standard deviation from the mean across all simulation runs, indicating the variability of the results. The wider the band, the more variable the outcome was for that strategy at that point.*")

    st.markdown("---")
    st.header(
        "Detailed Breakdown (Averages from Last Run - Needs Update for Averages")  # This section needs more thought for aggregated display

    # --- Behind the Scenes Charts (Updated for Averages) ---
    # For simplicity, we'll calculate average impressions and clicks per ad
    # This part requires careful aggregation from the list of dictionaries

    # Calculate average impressions per ad for Bandit
    avg_b_impressions_per_ad = {ad_id: 0 for ad_id in ad_names.keys()}
    for run_impressions in all_bandit_final_impressions:
        for ad_id, count in run_impressions.items():
            avg_b_impressions_per_ad[ad_id] += count
    for ad_id in avg_b_impressions_per_ad:
        avg_b_impressions_per_ad[ad_id] /= num_simulations_to_run

    # Calculate average clicks per ad for Bandit (to derive alpha, beta for an "average" run)
    avg_b_clicks_per_ad = {ad_id: 0 for ad_id in ad_names.keys()}
    for run_clicks in all_bandit_final_clicks:
        for ad_id, count in run_clicks.items():
            avg_b_clicks_per_ad[ad_id] += count
    for ad_id in avg_b_clicks_per_ad:
        avg_b_clicks_per_ad[ad_id] /= num_simulations_to_run

    col_dist, col_impressions_chart = st.columns(2)  # Renamed for clarity

    with col_dist:
        st.subheader("Average Final Ad CTR Beliefs (Beta Distributions)")
        fig_dist_avg, ax_dist_avg = plt.subplots()
        x = np.linspace(0, 0.05, 200)  # Range for plotting CTR
        for ad in ad_creatives_list:
            # Calculate average alpha and beta from the average clicks and impressions
            avg_alpha = avg_b_clicks_per_ad[ad.creative_id] + 1  # Add 1 for prior
            avg_beta = avg_b_impressions_per_ad[ad.creative_id] - avg_b_clicks_per_ad[
                ad.creative_id] + 1  # Add 1 for prior

            dist = beta(avg_alpha, avg_beta)
            ax_dist_avg.plot(x, dist.pdf(x), label=f"{ad_names[ad.creative_id]} (Avg. Belief)")
        ax_dist_avg.set_title("Bandit's Average Beliefs About Each Ad's CTR")
        ax_dist_avg.set_xlabel("Click-Through Rate")
        ax_dist_avg.set_ylabel("Probability Density")
        ax_dist_avg.legend()
        st.pyplot(fig_dist_avg)

    with col_impressions_chart:
        st.subheader("Average Impressions per Ad Creative (Bandit)")
        avg_impressions_df = pd.DataFrame({
            "Ad Creative": [ad_names[ad_id] for ad_id in avg_b_impressions_per_ad.keys()],
            "Average Impressions": list(avg_b_impressions_per_ad.values())
        })
        st.bar_chart(avg_impressions_df.set_index("Ad Creative"))
        st.markdown("*Shows how the bandit, on average, allocated its impressions.*")

else:
    st.info("Adjust the simulation settings in the sidebar and click 'Run Multiple Simulations' to begin.")
