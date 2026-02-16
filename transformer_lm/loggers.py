import csv
import time
import os
from typing import Optional, Union


class ExperimentLogger:
    def __init__(self, log_dir: str, filename: str = "metrics.csv"):
        """
        Initialize the experiment logger.

        Args:
            log_dir: Directory to save the log file.
            filename: Name of the CSV file.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)

        # Check if file exists to determine if we need to write header
        file_exists = os.path.isfile(self.filepath)

        self.file = open(self.filepath, "a", newline="")
        self.writer = csv.writer(self.file)

        if not file_exists:
            # Header
            self.writer.writerow(["step", "wall_time", "name", "value"])
            self.file.flush()

        self.start_time = time.time()

    def log_scalar(self, name: str, value: Union[float, int], step: int):
        """
        Log a scalar value.

        Args:
            name: Name of the metric (e.g., "train_loss").
            value: Value of the metric.
            step: Current training step/iteration.
        """
        wall_time = time.time() - self.start_time
        self.writer.writerow([step, wall_time, name, value])
        self.file.flush()

    def close(self):
        """Close the log file."""
        if self.file:
            self.file.close()
