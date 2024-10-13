import os

class Planner:
    """
    A class used to generate datasets for training and testing.

    Attributes:
    train_dir (str): Directory for training data.
    test_dir (str): Directory for testing data.
    split_ratio (float): Ratio to split the data into training and testing sets.
    positive_ratio (float): Ratio of positive samples in the dataset.
    """

    def __init__(self, train_dir: str = "train", test_dir: str = "test", split_ratio: float = 0.2, positive_ratio: float = 0.3):
        """
        Initializes the Planner with directories and ratios.

        Parameters:
        train_dir (str): Directory for training data. Default is "train".
        test_dir (str): Directory for testing data. Default is "test".
        split_ratio (float): Ratio to split the data into training and testing sets. Default is 0.2.
        positive_ratio (float): Ratio of positive samples in the dataset. Default is 0.3.
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.split_ratio = split_ratio
        self.positive_ratio = positive_ratio

    def generate(self, func, size: int = 100):
        """
        Generates datasets by calling the provided function.

        Parameters:
        func (function): Function to generate the data. It should accept three parameters: directory, size, and positive_ratio.
        size (int): Total size of the dataset to generate. Default is 100.

        Returns:
        None
        """
        test_size = int(size * self.split_ratio)
        train_size = size - test_size

        train_dir = os.path.join(self.train_dir, func.__name__)
        test_dir = os.path.join(self.test_dir, func.__name__)

        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

        func(train_dir, train_size, self.positive_ratio)
        func(test_dir, test_size, self.positive_ratio)