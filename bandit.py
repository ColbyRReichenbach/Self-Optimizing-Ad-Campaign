# In bandit.py
import numpy as np
from typing import List, Dict
from simulation import AdCreative

class ThompsonSamplingBandit:
    """
    Implements the Thompson Sampling algorithm.
    """
    def __init__(self, creatives: List[AdCreative]):
        """
        Initializes the Thompson Sampling bandit.

        Args:
            creatives (List[AdCreative]): A list of AdCreative objects to be tested.
        """
        self.creatives = creatives
        # self.ad_stats stores the "successes" (alpha)
        # and "failures" (beta) for each ad's Beta distribution.
        # We initialize with alpha=1 and beta=1 to represent a uniform prior belief,
        # avoiding issues with zeros and encouraging initial exploration.
        self.ad_stats: Dict[int, Dict[str, int]] = {
            ad.creative_id: {'alpha': 1, 'beta': 1} for ad in self.creatives
        }

    def select_ad(self) -> AdCreative:
        """
        Selects an ad to show next based on the Thompson Sampling strategy.

        Returns:
            AdCreative: The ad creative chosen to be displayed for the current impression.
        """
        best_ad = None
        max_sample = -1

        # Loop through each ad creative available
        for ad in self.creatives:
            # For each ad, draw a random sample from its current Beta distribution.
            # This sample represents a "possible CTR" for this ad.
            alpha = self.ad_stats[ad.creative_id]['alpha']
            beta = self.ad_stats[ad.creative_id]['beta']
            sampled_ctr = np.random.beta(alpha, beta)

            # Keep track of the ad that produced the highest sample in this round.
            if sampled_ctr > max_sample:
                max_sample = sampled_ctr
                best_ad = ad

        return best_ad

    def update(self, ad_id: int, was_clicked: int):
        """
        Updates the algorithm's beliefs based on the outcome of an ad impression.

        Args:
            ad_id (int): The ID of the ad that was shown.
            was_clicked (int): The result of the impression (1 for a click, 0 for no click).
        """
        # If the ad was clicked, we count it as a "success" and increment its alpha value.
        if was_clicked == 1:
            self.ad_stats[ad_id]['alpha'] += 1
        # If the ad was not clicked, we count it as a "failure" and increment its beta value.
        else:
            self.ad_stats[ad_id]['beta'] += 1