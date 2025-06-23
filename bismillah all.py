import streamlit as st
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
