from .generator import *
from .planner import Planner

# Instantiate the Planner class
planner = Planner()

# Loop through each dataset and generate data
for dataset in [dataset01, dataset02, dataset03, dataset04, dataset05]:
    """
    Generates data for each dataset using the Planner class.

    Parameters:
    dataset (list): List of datasets to generate data for.
    size (int): Size of the data to generate for each dataset.

    Returns:
    None
    """
    planner.generate(dataset, size=10)