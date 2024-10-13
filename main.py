from data.generator import *

import concurrent.futures
import os

def ensure_path_exists(path):
    """
    Ensures that the specified directory path exists.
    If the directory doesn't exist, it creates the necessary directories.
    """
    if not os.path.exists(path):
        os.makedirs(path)
def main():
    # Common configuration for training datasets
    train_config = {
        "size": 10,
        "positive_ratio": 0.3,
        "video_frames": 180,
        "framerate": 60
    }
    
    # Common configuration for test datasets
    test_config = {
        "size": 5,  # Smaller test size
        "positive_ratio": 0.3,
        "video_frames": 180,
        "framerate": 60
    }

    # List of dataset functions and their respective arguments for training
    train_dataset_functions = [
        # (dataset01_video, {**train_config, "path": "datasets/dataset01/train"}),
        (dataset02_video, {**train_config, "path": "datasets/dataset02/train"}),
        (dataset03_video, {**train_config, "path": "datasets/dataset03/train"}),
        (dataset04_video, {**train_config, "path": "datasets/dataset04/train"}),
        (dataset05_video, {**train_config, "path": "datasets/dataset05/train"}),
        # (dataset06_video, {**train_config, "path": "datasets/dataset06/train"}),
        # (dataset07_video, {**train_config, "path": "datasets/dataset07/train"}),
    ]
    
    # List of dataset functions and their respective arguments for testing
    test_dataset_functions = [
        # (dataset01_video, {**test_config, "path": "datasets/dataset01/test"}),
        (dataset02_video, {**test_config, "path": "datasets/dataset02/test"}),
        (dataset03_video, {**test_config, "path": "datasets/dataset03/test"}),
        (dataset04_video, {**test_config, "path": "datasets/dataset04/test"}),
        (dataset05_video, {**test_config, "path": "datasets/dataset05/test"}),
        # (dataset06_video, {**test_config, "path": "datasets/dataset06/test"}),
        # (dataset07_video, {**test_config, "path": "datasets/dataset07/test"}),
    ]


    dataset_functions = train_dataset_functions + test_dataset_functions
    
    
    
    for func, kwargs in dataset_functions:
        ensure_path_exists(kwargs["path"])  # Ensure the directory exists
        func(**kwargs)
    
    # # Use ProcessPoolExecutor for CPU-bound tasks
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     # Submit all dataset functions to the executor
    #     futures = [executor.submit(func, **kwargs) for func, kwargs in dataset_functions]

    #     # Optionally, wait for all futures to complete and handle exceptions
    #     for future in concurrent.futures.as_completed(futures):
    #         try:
    #             future.result()  # Get the result (None in this case)
    #         except Exception as e:
    #             print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
