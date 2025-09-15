# Project: Advanced Ad Campaign Optimization with Multi-Armed Bandits

**Live App:** [https://self-optimizing-ad-campaign-cvyuyilsfhqbprtznidsjb.streamlit.app/]

*An interactive web application that simulates and visualizes how a Reinforcement Learning agent (a Thompson Sampling Bandit) consistently outperforms traditional A/B testing in real-time to maximize advertising ROI, with results aggregated over multiple simulation runs for statistical robustness.*

---

## 1. Project Evolution & Motivation

This project builds upon insights gained from a previous analysis of traditional A/B testing for ad performance. In my earlier work, **[Ad Performance Analysis with A/B Testing](https://github.com/ColbyRReichenbach/Ad-Performance-Analysis-AB-Testing)**, I utilized statistical methods to analyze ad campaign performance.

While that approach effectively identified top campaigns retrospectively, it highlighted a core inefficiency: traditional A/B tests inherently require a fixed "exploration" period where significant budget is spent on underperforming ad variants. This realization sparked the central question for the current project: **Could a machine learning system learn and adapt in real-time to minimize this wasted ad spend and discover the optimal creative more efficiently?**

## 2. Problem Statement

The primary challenge addressed here is the inefficiency of traditional A/B testing in dynamic advertising environments. Key limitations of standard A/B tests include:
* **Opportunity Cost:** Significant ad spend is allocated to less effective ad creatives during the fixed testing duration.
* **Slow Adaptation:** A/B tests are not designed to react to changes in ad performance or market dynamics once the test is underway.
* **Delayed Optimization:** Decisions to shift budget to the winning ad can only be made after the testing period concludes and statistical significance is reached.

This project seeks to demonstrate a more agile and profitable approach using a Multi-Armed Bandit algorithm.

## 3. Solution Overview

This project implements and evaluates a **Multi-Armed Bandit** algorithm (specifically, Thompson Sampling) against a traditional A/B test within a simulated advertising environment.

Key components of the solution:
* A **Python-based simulation** defines multiple ad creatives, each with a distinct (but initially unknown to the algorithm) true click-through rate (CTR).
* The **Thompson Sampling Bandit** dynamically updates its beliefs about each ad's performance after every impression.
* It intelligently allocates more traffic to ads that demonstrate higher performance, balancing exploration of new options with exploitation of known winners.
* To ensure conclusions, the entire simulation campaign is **run multiple times (user-configurable)**.
* The aggregated results are presented in an interactive **Streamlit** dashboard, showcasing average performance and variability.

## 4. Key Features

* Interactive Streamlit Dashboard for running and visualizing simulations.
* Side-by-side comparison of a Thompson Sampling Bandit and a traditional A/B Test.
* Configurable number of simulation runs for statistical analysis.
* Calculation and display of average performance metrics (Total Clicks, Click-Through Rate, Percentage Lift) across all runs.
* Visualization of performance variability using standard deviation bands on the main comparative chart.
* Breakdown of average impression allocation and final algorithmic beliefs (Beta distributions based on average outcomes).

## 5. Methodology

**Thompson Sampling:** This Bayesian algorithm treats each ad creative as an "arm" of a multi-armed bandit. For each arm, it maintains a Beta distribution representing the current belief about its true CTR (parameterized by observed successes/clicks and failures/non-clicks). To choose an ad, it samples from each arm's Beta distribution and selects the arm with the highest sampled value. This naturally balances exploration (giving chances to arms with higher uncertainty) and exploitation (favoring arms with a high proven success rate).

**A/B Testing Simulation:** The A/B test serves as a baseline, allocating an equal number of impressions to each ad creative throughout the simulation.

**Statistical Robustness:** To account for the stochastic nature of individual simulation runs, the core experiment is executed multiple times. Performance metrics are then averaged, and standard deviations are calculated and visualized. This provides a much more reliable understanding of each strategy's true effectiveness and consistency.

## 6. Key Results & Insights

* On average, the Thompson Sampling bandit achieved an **increase in total clicks** compared to the traditional A/B test.
* The bandit's average final click-through rate was, significantly higher than the A/B test's, and much closer to the true CTR of the best-performing ad (2.1% in our defined scenario).

The analysis of the average Beta distributions at the end of the simulations shows that the Thompson Sampling algorithm correctly identifies and converges on the best-performing ad with high confidence and less wasted exploration compared to the fixed A/B test.

## 7. Tech Stack
* **Language:** Python
* **Core Libraries:** Streamlit, Pandas, NumPy, SciPy (for `scipy.stats.beta`), Matplotlib
* **Core Concepts:** Reinforcement Learning (Multi-Armed Bandits), Bayesian Statistics (Beta Distributions), A/B Testing, Monte Carlo Simulation, Statistical Significance, Marketing Analytics.

## 8. Project Setup & Usage

This project is designed to be run locally using the provided environment configuration.

**To run this app locally:**
1.  Clone the repository:
    ```bash
    git clone [https://github.com/ColbyRReichenbach/ad-optimization-bandit.git](https://github.com/ColbyRReichenbach/ad-optimization-bandit.git) 
    cd ad-optimization-bandit
    ```
2.  Set up the Conda environment:
    ```bash
    # Ensure you have Anaconda or Miniconda installed
    conda create --name bandit-env python=3.9 # Or your preferred Python version
    conda activate bandit-env
    pip install -r requirements.txt
    ```
3.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser, allowing you to configure and run the simulations.

---
