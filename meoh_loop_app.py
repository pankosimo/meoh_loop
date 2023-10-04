import streamlit as st
import pandas as pd
import numpy as np
import scipy
from scipy.optimize import fsolve
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xlwings as xw

def load_model(F_mass, Sn, co2_co_ratio, X_CO2, X_CO):
    wb=xw.Book('Power_to_MeOH_Mass_Balance_Model_solution.xlsx')
    sheet = wb.sheets['Tabelle1']
    sheet['D17'].value = F_mass
    sheet['B31'].value = Sn
    sheet['B33'].value = co2_co_ratio
    sheet['H18'].value = X_CO2/100
    sheet['H19'].value = X_CO/100
    year_capacity = sheet['L26'].value
    recycle_feed_ratio = sheet['F25'].value
    sn_reac_in = sheet['H25'].value
    inert_reac_in = sheet['G6'].value
    h2_conc = sheet['C5'].value
    co_conc = sheet['C4'].value
    co2_conc = sheet['C3'].value
    inert_conc = sheet['C6'].value
    return year_capacity, recycle_feed_ratio, sn_reac_in, inert_reac_in, h2_conc, co_conc, co2_conc, inert_conc
st.set_page_config(layout="wide", page_title="MeOH Synthesis Loop") 
st.title('Hands-On Exercise: MeOH Synthesis Loop')
with st.sidebar:
    
    st.header("Properties")
    F_mass = st.number_input("Feed total mass flow / kg/h",2500,10000,step=150,value=5000, key="F_mass")
    Sn = st.number_input("Feed Stoichiometric Number / -",2.05,10.0,step=0.05, value=3.0,  key="Sn",format="%.2f")
    co2_co_ratio = st.number_input("Feed Ratio CO2 / CO / -",0.000,1000.000,step=0.5, value=1.0, key="co2_co_ratio",format="%.1f")
    X_CO2 = st.number_input("CO2 Conversion Reator / %",0.000,100.000,step=0.5, value=20.0, key="X_CO2",format="%.1f")
    X_CO = st.number_input("CO Conversion Reator / %",0.000,100.000,step=0.5, value=70.0, key="X_CO",format="%.1f")

year_capacity, recycle_feed_ratio, sn_reac_in, inert_reac_in, h2_conc, co_conc, co2_conc, inert_conc = load_model(F_mass, Sn, co2_co_ratio, X_CO2, X_CO)
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
   st.header("Feed Composition")
   col11, col22 = st.columns([2, 1])
   with col11:
        st.write("H2 / mol-%")
        st.write("CO / mol-%")
        st.write("CO2 / mol-%")
        st.write("Inerts / mol-%")
   with col22:
        st.write(f'{h2_conc:.2%}')
        st.write(f'{co_conc:.2%}')
        st.write(f'{co2_conc:.2%}')
        st.write(f'{inert_conc:.2%}')

with col2:
    st.image("meoh_loop_scheme.jpg")
   
with col3:
   st.header("Process Results")
   col11, col22 = st.columns([2, 1])
   with col11:
        st.write("MeOH Capacity / kt/a")
        st.write("Reactor Inlet Stoichiometric Number / -")
        st.write("Reactor Inlet Inert Concentration / mol-%")
        st.write("Ratio Recylce Flow Rate / Feed Flowrate / -")
   with col22:
        st.write(f'{year_capacity:.1f}')
        st.write(f'{sn_reac_in:.1f}')
        st.write(f'{inert_reac_in:.2%}')
        st.write(f'{recycle_feed_ratio:.1f}')

    