import pandas as pd
import sys


class ClientBuffer:
    """
    ClientBuffer is a class for managing a buffer of client-specific data. It is designed for use in continual learning
    scenarios where data from past tasks needs to be stored and accessed for future training.
    """

    def __init__(self, reference_dataset):
        """
        Initialize a new ClientBuffer with the specified buffer size.

        Parameters:
        - reference_dataset (dataset.Dataset2D): A reference Dataset2D instance.
        """
        # Create an empty dataset instance
        self.dataset = reference_dataset.new_instance(pd.DataFrame())

        # Set to very large value (almost "infinite" memory)
        self.buffer_size = sys.maxsize

    def __len__(self):
        return len(self.dataset)

    def add_task_data(self, new_dataframe):
        """
        Append task data to the buffer. If adding this data would exceed the maximum buffer size, the oldest data
        is removed to maintain the buffer's size.

        Parameters:
        - new_dataframe (pd.DataFrame): A DataFrame containing data from the completed task.
        """
        current_dataframe = self.dataset.dataframe
        combined_datafarme = pd.concat([current_dataframe, new_dataframe], ignore_index=True)

        if len(combined_datafarme) > self.buffer_size:
            # Keep last "self.buffer_size" rows
            combined_datafarme = combined_datafarme[-self.buffer_size :]

        self.dataset = self.dataset.new_instance(combined_datafarme)
