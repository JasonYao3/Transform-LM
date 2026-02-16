import numpy as np
import os

# Create 10MB of random uint16 data
data = np.random.randint(0, 50257, size=(1024 * 100,), dtype=np.uint16)
data.tofile("dummy_data.bin")
print("Created dummy_data.bin")
