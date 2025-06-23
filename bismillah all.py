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
import seaborn as sns

# Sidebar dengan option_menu
with st.sidebar:
    selected = option_menu(
        menu_title="Tugas Akhir",
        options=["Preprocessing", "Segmentasi", "Klasifikasi"],
        menu_icon=None,
        icons=["house", "hr", "diagram-3", "person"],
        default_index=0,
        styles={
            "icon": {"color": "blue"},
            "nav-link": {
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
        }
    )

# Tampilan halaman berdasarkan pilihan
if selected == "Preprocessing":
    st.title("Image Preprocessing")
    st.write("Silakan unggah citra untuk dianalisa")

    #Tiga kolom input citra
    col1, col2, col3 = st.columns(3)
    with col1:
        inputcitra1 = st.file_uploader("FLAIR", type=["nii", "jpg", "jpeg", "png"])
    with col2:
        inputcitra2 = st.file_uploader("T1ce", type=["nii", "jpg", "jpeg", "png"])
    with col1:
        inputcitra1 = st.file_uploader("T2", type=["nii", "jpg", "jpeg", "png"])

    if inputcitra1:
        st.write(f"FLAIR: {inputcitra1.name}")
    if inputcitra2:
        st.write(f"T1ce: {inputcitra2.name}")
    if inputcitra3:
        st.write(f"T2: {inputcitra3.name}")

elif selected == "Segmentasi":
    st.title("Halaman Segmentasi")
    st.write("Implementasikan segmentasi di sini...")

elif selected == "Klasifikasi":
    st.title("Halaman Klasifikasi")
    st.write("Implementasikan klasifikasi di sini...")
