import streamlit as st
import pandas as pd
import numpy as np
import scipy
from scipy.optimize import fsolve
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class binary_column:
    def __init__(self,alpha,F,xf_1,xs_1,xd_1,P,v=2.5, qf=1):
        self.F = F
        self.xf_1 = xf_1
        self.xf_2 = 1 - self.xf_1
        self.xs_1 = xs_1
        self.xs_2 = 1 - self.xs_1
        self.xd_1 = xd_1
        self.xd_2 = 1 - self.xd_1
        self.P = P
        self.v = v
        self.qf = qf
        if self.qf == 1:
            self.qf = 1.000001               
        self.d_hlv = 35
        self.mole_balance()
        self.update_on_v(self.v)
        self.enthalpy_balance()
        self.thermo_model(alpha)
        print("Colum init done")
    def mole_balance(self):
        self.D = self.F*(self.xf_1-self.xs_1)/(self.xd_1-self.xs_1)
        self.S = self.F - self.D
        self.L = self.v*self.D
        self.G = self.D + self.L
        self.L_star = self.L + self.qf*self.F
        self.G_star = self.L_star - self.S
        
    def update_on_v(self,v):
        self.v = v
        self.r = (self.v*self.D+self.qf*self.F-self.S)/self.S
        self.mole_balance()
        self.enthalpy_balance()
    def update_on_r(self,r):
        self.r = r
        self.v = (self.r*self.S-self.qf*self.F+self.S)/self.D        
        self.mole_balance()
        self.enthalpy_balance()
    def update_on_qf(self,qf):
        self.qf = qf
        self.r = (self.v*self.D+self.qf*self.F-self.S)/self.S        
        self.mole_balance()
        self.enthalpy_balance()
    def enthalpy_balance(self):
        self.Qv = self.d_hlv*self.G_star/3.6
        self.Qk = self.d_hlv*self.G/3.6
    def thermo_model(self,alpha):
        self.alpha = alpha
        self.x_ggw = np.linspace(0,1,200)
        self.y_ggw = lambda x: self.alpha*x/(x*(self.alpha-1)+1)

    def calc_stages(self):
        self.einspeisegerade = lambda x: (-self.qf)*x/(1-self.qf)+self.xf_1/(1-self.qf)
        self.verstärkergerade = lambda x: self.v/(self.v+1)*x+self.xd_1/(self.v+1)
        self.abtriebsgerade = lambda x: (self.r+1)/self.r*x - self.xs_1/self.r
    
        
    
        #schnittpunkt arbeitsgeraden - einspeisegerade
        x_s_ae = fsolve(lambda x: self.einspeisegerade(x) - self.verstärkergerade(x),self.xf_1)[0]
        y_s_ae = self.einspeisegerade(x_s_ae)
    
        #schnittpunkt x-achse - einspeisegerade
        x_s_ax = fsolve(lambda x: self.einspeisegerade(x) - 0,self.xf_1)[0]
    
        #x-abschnitte geraden
    
        self.x_einspeise = np.linspace(min(x_s_ae,x_s_ax),max(x_s_ae,x_s_ax),50)
        self.x_verstärker = np.linspace(x_s_ae, self.xd_1,50)
        self.x_abtrieb = np.linspace(self.xs_1, x_s_ae, 50)
    
        max_steps = 100
    
        x_ag = self.xs_1
        y_ag = self.xs_1
        self.n_steps = 0
        self.x_pair = []
        self.y_pair = []
        while y_ag < self.xd_1:
            if self.n_steps > max_steps:
                self.n_steps = np.inf
                break
            self.n_steps+=1
    
            xggw = x_ag
            yggw = self.y_ggw(xggw)
            self.x_pair.append([x_ag,xggw])
            self.y_pair.append([y_ag,yggw])
            #ax1.plot([x_ag,xggw],[y_ag,yggw],lw=1,color='red')
            y_ag = yggw
            x_ag = max(fsolve(lambda x: self.abtriebsgerade(x)-y_ag,x_ag)[0],fsolve(lambda x: self.verstärkergerade(x)-y_ag,x_ag)[0])
            #ax1.plot([x_ag,xggw],[y_ag,yggw],lw=1,color='red')
            self.x_pair.append([x_ag,xggw])
            self.y_pair.append([y_ag,yggw])
    def iter_n(self,v):
        print(self.v)
        print(self.n_steps)
        self.update_on_v(v)
        self.calc_stages()
        return self.n_steps
    def plot_stages(self):
        fig, ax1 = plt.subplots(figsize=(20,20))
    
    
        ax1.plot(self.x_ggw,self.y_ggw(self.x_ggw), color = 'black', label=r'$P\ = \ '+str(P)+' \ bar$')
        
        ax_twin = ax1.twinx()
        ax_twiny = ax1.twiny()
        
        major_tick = 0.1
        minor_tick = 0.01
        
        
        ax1.set_xlim(0,1)
        ax1.set_ylim(0,1)
        
        ax1.plot([0,1],[0,1], color='black', lw=2,ls='--')
        
        ax1.plot(self.x_einspeise, self.einspeisegerade(self.x_einspeise))
        ax1.plot(self.x_verstärker, self.verstärkergerade(self.x_verstärker))
        ax1.plot(self.x_abtrieb, self.abtriebsgerade(self.x_abtrieb))
        
        # ax1.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True,
        #                      bottom=True, top=True, left=True, right=True)
        # ax1.yaxis.set_major_locator(ticker.MultipleLocator(major_tick))
        # ax1.yaxis.set_minor_locator(ticker.MultipleLocator(minor_tick))
        # ax_twin.yaxis.set_major_locator(ticker.MultipleLocator(major_tick))
        # ax_twin.yaxis.set_minor_locator(ticker.MultipleLocator(minor_tick))
        # ax1.grid(visible=True,axis='y', which='major', color='black', linestyle='-', linewidth=1)
        # ax1.grid(visible=True,axis='y', which='minor', color='grey', linestyle='--', linewidth=0.5)
        # ax_twin.grid(visible=True,axis='y', which='major', color='black', linestyle='-', linewidth=1)
        # ax_twin.grid(visible=True,axis='y', which='minor', color='grey', linestyle='--', linewidth=0.5)
        
        # ax1.xaxis.set_major_locator(ticker.MultipleLocator(major_tick))
        # ax1.xaxis.set_minor_locator(ticker.MultipleLocator(minor_tick))
        # ax_twiny.xaxis.set_major_locator(ticker.MultipleLocator(major_tick))
        # ax_twiny.xaxis.set_minor_locator(ticker.MultipleLocator(minor_tick))
        # ax1.grid(visible=True,axis='x', which='major', color='black', linestyle='-', linewidth=1)
        # ax1.grid(visible=True,axis='x', which='minor', color='grey', linestyle='--', linewidth=0.5)
        
        for i in np.arange(0,1,0.05):
            ax1.axhline(i, color = 'black', lw=1, zorder=-10)
            ax1.axvline(i, color = 'black', lw=1, zorder=-10)
        
        ax_twin.invert_yaxis()
        ax_twiny.invert_xaxis()
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        ax_twin.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        ax_twiny.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        #ax1.set_xlabel(r'$x_{{{:s}}}$'.format(comp_01))
        #ax1.set_ylabel(r'$y_{{{:s}}}$'.format(comp_01))
        #ax_twiny.set_xlabel(r'$x_{{{:s}}}$'.format(comp_02))
        #ax_twin.set_ylabel(r'$y_{{{:s}}}$'.format(comp_02))

        for i in range(len(self.x_pair)):
            ax1.plot(self.x_pair[i],self.y_pair[i],lw=1,color='red')

def r_slider():
    app_column.update_on_r(st.session_state.r_slider)
    st.session_state.v_slider = app_column.v
    app_column.calc_stages()
def v_slider():
    app_column.update_on_v(st.session_state.v_slider)
    st.session_state.r_slider = app_column.r
    app_column.calc_stages()
def qf_slider():
    app_column.update_on_qf(st.session_state.qf_slider)
    st.session_state.r_slider = app_column.r
    app_column.calc_stages()

P = 1
F = 150
xf_1 = 0.65
xd_1  = 0.97
xs_1  = 0.05
alpha = 1.5
qf=1
# start_v=4
# app_column = binary_column(alpha=start_alpha,F=start_F,xf_1=start_xf_1,xs_1=start_xs_1,xd_1=start_xd_1,P=start_P,
#                            qf=start_qf,v=start_v)
#v_slider()


st.set_page_config(layout="wide", page_title="Binäre Destillation") 
st.title('Hands-On Übung zum Thema Rektifikationskolonnen')
with st.sidebar:
    
    st.header("Einstellungen")
    qf = st.number_input("Thermischer Zustand Feed",0.000,1.000,step=0.05,value=1.0, key="qf_slider",format="%.2f", on_change=qf_slider)
    v = st.number_input("Rücklaufverhältnis",1.000,500.000,step=0.05,  key="v_slider",format="%.2f", on_change=v_slider)
    r = st.number_input("Verdampfungsverhältnis",1.000,500.000,step=0.05,  key="r_slider",format="%.2f", on_change=r_slider)
    
# P = 1
# F = 150
# xf_1 = 0.65
# xd_1  = 0.97
# xs_1  = 0.05
# alpha = 1.5
app_column = binary_column(alpha=alpha,F=F,xf_1=xf_1,xs_1=xs_1,xd_1=xd_1,P=P,qf=qf,v=v)
app_column.calc_stages()

col1, col2, col3 = st.columns([1, 2, 2])
with col1:
   #st.header("A cat")
    st.image("column_scheme.jpg")

with col2:
    st.header("")
    st.divider()
    st.subheader("Kolonnenabmessungen")
    st.divider()
    col11, col22 = st.columns([1, 2])
    with col11:
        st.write("Anzahl an notwendigen Trennstufen:")
    with col22:
        st.write(app_column.n_steps)
    
    st.divider()
    st.subheader("Innere Ströme")
    st.divider()
    col11, col22 = st.columns([1, 2])
    with col11:
        st.write("Rücklaufverhältnis v / -:")
        st.write("Verdampfungsverhältnis r / kmol/h:")
        st.write("Flüssigkeit Verstärkerteil L / kmol/h:")
        st.write("Gas Verstärkerteil G / kmol/h:")
        st.write("Flüssigkeit Abtriebsteil L* / kmol/h:")
        st.write("Gas Abtriebsteil G* / kmol/h:")
    with col22:
        st.write(round(app_column.v,2))
        st.write(round(app_column.r,2))
        st.write(round(app_column.L,1))
        st.write(round(app_column.G,1))
        st.write(round(app_column.L_star,1))
        st.write(round(app_column.G_star,1))
with col3:
    st.header("")
    st.divider()
    st.subheader("Massenbilanz")
    st.divider()
    col11, col22 = st.columns([1, 2])
    with col11:
        st.write("Feed F/ kmol/h:")
        st.write("Kopfstrom D / kmol/h:")
        st.write("Sumpfstrom S / kmol/h:")
    with col22:
        st.write(round(app_column.F,1))
        st.write(round(app_column.D,1))
        st.write(round(app_column.S,1))
    st.divider()
    st.subheader("Energiebedarf")
    st.divider()
    col11, col22 = st.columns([1, 2])
    with col11:
        st.write("Verdampferleistung / kW:")
        st.write("Kondensatorleistung / kW:")

    with col22:
        st.write(round(app_column.Qv,0))
        st.write(round(app_column.Qk,0))

    