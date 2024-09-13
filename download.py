import sys
import os

# Add the absolute path of the 'src' directory to the system path
sys.path.append(os.path.abspath('C:/Users/VICTUS/Product-Identification/src'))

# Now import your function from utils.py
from src.utils import download_images

import pandas as pd

# Load the train dataset
train_df = pd.read_csv('train.csv')

# Download train images using the correct argument 'download_folder'
download_images(train_df['image_link'], download_folder='train_images/')
