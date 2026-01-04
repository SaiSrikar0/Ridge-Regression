import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#page configuration
st.set_page_config(page_title = "Ridge Regression App", layout = "centered")

#load CSS
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)
load_css("style.css")

#title
st.markdown("""
            <div class = "card">
            <h1>Ridge Regression App</h1>
            <p> Predict<b> Tip Amount </b> from <b> Total Bill </b> using Ridge Regression</p>
            </div>
""", unsafe_allow_html = True)

#load data
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df =  load_data()

#dataset preview
st.markdown('<div class = "card"><h2>Dataset Preview</h2></div>', unsafe_allow_html = True)
st.dataframe(df.head())

#prepare data
x, y = df[["total_bill"]].values, df["tip"].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#hyperparameter tuning
st.markdown('<div class = "card"><h2>Ridge Regularization Parameter</h2></div>', unsafe_allow_html = True)
alpha = st.slider("Alpha (Regularization Strength)", 0.001, 1.0, 0.1, 0.001)

#train model
model = Ridge(alpha = alpha, random_state = 42)
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)

#model evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 2)

#visualization
st.markdown('<div class = "card"><h2>Total bill vs tip</h2></div>', unsafe_allow_html = True)
fig, ax =  plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha = 0.6)
ax.plot(df["total_bill"], model.predict(scaler.transform(df[["total_bill"]].values)), color = "red")
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
st.pyplot(fig)

#performance
st.markdown('<div class = "card"><h2>Model Performance</h2>', unsafe_allow_html = True)
c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3, c4 = st.columns(2)
c3.metric("R2", f"{r2:.2f}")
c4.metric("Adjusted R2", f"{adj_r2:.2f}")
st.markdown('</div>', unsafe_allow_html = True)

#m and c
st.markdown(f"""
            <div class = "card">
            <h3>Model Coefficient and Intercept</h3>
            <p> <b> Coefficient : </b> {model.coef_[0]:.3f} <br>
            <b> Intercept : </b> {model.intercept_:.3f} </p>
            </div>
""", unsafe_allow_html = True)

#prediction
bill = st.slider("total bill $", float(df["total_bill"].min()), float(df["total_bill"].max()), 30.0)
tip = model.predict(scaler.transform([[bill]]))[0]
st.markdown(f'''
            <div class = "card">
                <h2>Predict Tip Amount  $ {tip:.2f}</h2>
            </div>
            ''', unsafe_allow_html = True)
