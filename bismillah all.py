import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_image_comparison import image_comparison

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.transforms.functional import to_pil_image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import seaborn as 

#option menu di sidebar
with st.sidebar:
  selected = option_menu(
    menu_title = "Tugas Akhir",
    options = ["Preprocessing", "Segmentasi", "Klasifikasi"],
    menu_icon = None,
    icons = ["house", "hr", "diagram-3, "person"],
    default_index = 0,
    style = {
      "icon" : {"color" : "blue"},
      "nav-link" : {
        "text-align:" : "left",
        "margin" : "0px",
        "--hover-color" : "#eee"
      }
    }
  )
