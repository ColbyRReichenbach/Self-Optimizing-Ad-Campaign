# In simulation.py
import numpy as np
from typing import List

class AdCreative:
    """
    Represents a single ad creative with a hidden, true click-through rate (CTR).
    This class simulates the 'real world' performance of an individual ad.

    """
    def __init__(self, creative_id: int, true_ctr: float):
        """
        Initializes an AdCreative instance.

        Args:
            creative_id (int): A unique identifier for the ad (e.g., 1, 2, 3).
            true_ctr (float): The actual, hidden probability (e.g., 0.021 for 2.1%)
                              that this ad will be clicked when shown.
        """
        self.creative_id = creative_id
        self.true_ctr = true_ctr

    def show_ad(self) -> int:
        """
        Simulates showing the ad to a single user and observing the outcome.

        Returns:
            int: 1 if the ad was clicked (a success), 0 if it was not (a failure).
        """
        # We use NumPy's binomial function to simulate a single coin flip
        # with a weighted probability (our true_ctr). This is a Bernoulli trial.
        return np.random.binomial(1, self.true_ctr)


class AdEnvironment:
    """
    Manages the set of ad creatives for our simulation.
    This acts as a simple container for all the ads in our campaign.
    """
    def __init__(self, creatives: List[AdCreative]):
        """
        Initializes the ad environment.

        Args:
            creatives (List[AdCreative]): A list of AdCreative objects.
        """
        self.creatives = creatives

    def get_creatives(self) -> List[AdCreative]:
        """Returns the list of ad creatives currently in the environment."""
        return self.creatives