# Project: Self-Optimizing Ad Campaign with a Multi-Armed Bandit

**Live App:** [Need to upload to community]

> An interactive web application that simulates and visualizes how a Reinforcement Learning agent (a Thompson Sampling Bandit) can outperform traditional A/B testing in real-time to maximize advertising ROI.

**(Optional but HIGHLY recommended: Insert a GIF of your Streamlit app in action here. It's incredibly effective.)**

---

## 1. Problem Statement

In a previous project, Ad Performance Analysis with A/B Testing, I analyzed historical campaign data to identify a winning ad using traditional statistical methods.

While effective, that analysis revealed a key business problem: standard A/B testing is inherently inefficient. It requires 

a fixed testing period where 50% of the budget is spent on the eventual losing ad. This project is the solution. It explores 

if a reinforcement learning approach can find the winner faster and more profitably by dynamically allocating budget to the 

best-performing creative in real-time.
## 2. Solution Overview

I built a simulation environment in Python to test a **Multi-Armed Bandit** algorithm, a concept from reinforcement learning, against a traditional A/B test.

* The **Thompson Sampling Bandit** updates its beliefs about each ad's performance after every impression.
* It dynamically allocates more traffic to the ads that are performing better, balancing the need to explore new options with exploiting known winners.
* The entire simulation is wrapped in an interactive **Streamlit** dashboard for analysis and visualization.

## 3. Key Results & Insights

The simulation results clearly demonstrate the superiority of the bandit algorithm.
* Over 1,000,000 impressions, the Thompson Sampling bandit achieved a **9.1% lift in total clicks** compared to the A/B test.
* The bandit's final click-through rate was **1.87%**, significantly higher than the A/B test's **1.71%**, and much closer to the true CTR of the best-performing ad (2.1%).
* The "Cumulative Clicks" chart below visualizes this outperformance, where the growing gap between the lines represents tangible value gained by using the smarter algorithm.

## 4. Tech Stack

* **Language:** Python
* **Libraries:** Streamlit, Pandas, NumPy, SciPy, Matplotlib
* **Core Concepts:** Reinforcement Learning (Multi-Armed Bandits), Bayesian Statistics (Beta Distributions), A/B Testing, Simulation

## 5. Project Setup & Usage

This project is containerized in a reproducible environment.

**To run this app locally:**
1.  Clone the repository:
    ```bash
    git clone [https://github.com/ColbyRReichenbach/Self-Optimizing-Ad-Campaign](https://github.com/ColbyRReichenbach/Self-Optimizing-Ad-Campaign)
    cd ad-optimization-bandit
    ```
2.  Set up the Conda environment:
    ```bash
    # Create the environment from the requirements file
    conda create --name bandit-env python=3.9
    conda activate bandit-env
    pip install -r requirements.txt
    ```
3.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser.