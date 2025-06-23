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

#====================HALAMAN PREPROCESSING====================#
if selected == "Preprocessing":
    st.title("Image Preprocessing")
    
    st.write("Silakan unggah citra untuk dianalisa")

    #Tiga kolom input citra
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            flair = st.file_uploader("FLAIR", type=["nii", "jpg", "jpeg", "png"])
        with col2:
            t1ce = st.file_uploader("T1ce", type=["nii", "jpg", "jpeg", "png"])
        with col1:
            t2 = st.file_uploader("T2", type=["nii", "jpg", "jpeg", "png"])

    if flair:
        st.write(f"FLAIR: {flair.name}")
    if t1ce:
        st.write(f"T1ce: {t1ce.name}")
    if t2: 
        st.write(f"T2: {t2.name}") 
    
    st.markdown("""
        <style>
        div[data-testid="stFileUploader"] > label {
            font-size: 14px;
        }
        div[data-testid="stFileUploader"] {
            padding-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    def display_file(file, name):
        if file is not None:
            file_type = file.name.split(".")[-1].lower()
            if file_type in ["nii", "nii.gz"]:
                # Baca NIfTI
                img = nib.load(file)
                data = img.get_fdata()
                st.write(f"**{name}**: Volume shape{data.shape}")
                # Kalau single channel, expand jadi channel axis=1
                if data.ndim =3 :
                    data = np.expand_dims(data, axis=-1)

                slice_idx = st.slider(
                    f"Choose slice index {name}",
                    min_value = 0,
                    max_value = data.shape[2]-1,
                    value = data.shape[2]//2,
                    key = f"{name}_slider"
                )

                # Kalau ada beberapa channel, tampilkan semua
                fig, axes = plt.subplots(1, data.shape[3], figsize=[5*data.shape[3], 5))
                if data.shape[3] == 1:
                    axes = [axes]    # single channel, untuk list
                for i in range (data.shape[3]):
                    axes[i].imshow(data[:, :, slice_idx, i], cmap="gray")
                    axes[i].set_title(f"{name} - Channel {i+1}")
                    axes[i].axis("off")
                st.pyplot(fig)
            elif file_type in ["jpg", "jpeg", "png"]:
                # Baca dan tampilkan citra biasa
                img = Image.open(file)
                st.image(img, caption=f"{name} Image", use_column_width=True)
            else:
                st.error(f"Format{file_type} tidak didukung untuk {name}.")
    display_file(flair_file, "FLAIR")
    display_file(t1ce_file, "T1CE")
    display_file(t2_file, "T2")
elif selected == "Segmentasi":
    st.title("Halaman Segmentasi")
    st.write("Implementasikan segmentasi di sini...")

elif selected == "Klasifikasi":
    st.title("Halaman Klasifikasi")
    st.write("Implementasikan klasifikasi di sini...")
