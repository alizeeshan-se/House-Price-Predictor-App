import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load trained model
model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Predictor")



# ..........................st.Expander(),,,,,,,,,,,................

with st.expander("ℹ️ About the Inputs (click to expand)"):
    st.markdown("""
- **Bedrooms**: Number of bedrooms in the house.
- **Bathrooms**: Number of bathrooms (full bathrooms only).
- **Living Area (sq ft)**: Total usable indoor living space (excluding basement).
- **Lot Area (sq ft)**: Entire land size including house, yard, and driveway.
- **Floors**: Total number of floors in the house.
- **Condition (1–5)**: Condition of the house from 1 (poor) to 5 (excellent).
- **Basement Area (sq ft)**: Area of the basement, if any.
- **Schools Nearby**: Number of schools near the house.
- **Distance to Airport (km)**: Approximate distance to the nearest airport.
    """)





# ..........................sidebar.................

# 🎯 Streamlit UI
st.set_page_config(layout="wide")

st.sidebar.title("House Price Predictor App")

image = Image.open("img1.jpg")  #left-sidebar image
st.sidebar.image(image, use_container_width=True)


st.sidebar.markdown("""
    <h2 style='font-family:Arial; color:#3366cc;'>Zeeshan Ali</h2>
    <p style='margin-bottom:5px;'>University of Malakand</p>
    <p style='margin-bottom:10px;'>Machine Learning Project</p>

    <p style='margin: 5px 0;'>
        <a href='https://www.linkedin.com/in/alizeeshanse' target='_blank' style='text-decoration: none;'>
            <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' width='20' style='vertical-align:middle; margin-right:5px;'>
            LinkedIn
        </a>
    </p>

    <p style='margin: 5px 0;'>
        <a href='https://github.com/alizeeshan-se' target='_blank' style='text-decoration: none;'>
            <img src='https://cdn-icons-png.flaticon.com/512/25/25231.png' width='20' style='vertical-align:middle; margin-right:5px;'>
            GitHub
        </a>
    </p>

    <p style='margin: 5px 0;'>
        <img src='https://cdn-icons-png.flaticon.com/512/281/281769.png' width='20' style='vertical-align:middle; margin-right:5px;'>
        <a href='mailto:alizeeshanse@gmail.com' style='text-decoration: none; color: white;'>alizeeshanse@gmail.com</a>
    </p>

    <p style='margin: 5px 0;'>
        <img src='https://cdn-icons-png.flaticon.com/512/733/733585.png' width='20' style='vertical-align:middle; margin-right:5px;'>
        <a href='https://wa.me/923499373126' target='_blank' style='text-decoration: none; color: white;'>+92 349 9373126</a>
    </p>

    <hr style='margin:10px 0; border: 0; border-top: 1px solid #ccc;'>

    <p style='color:white; font-weight:bold;'>✅ Available for Work</p>
""", unsafe_allow_html=True)







# ----------- User Inputs (9 fields only) -----------

st.write("Enter the basic house details below:")


bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2, step=1, format="%d")
living_area = st.number_input("Living area (sq ft)", 100, 15000, 2000,help="The total area inside the house (excluding garden,yard,garage/basement).")
lot_area = st.number_input("Lot area (sq ft)", 100, 100000, 4000,    help="The total land area the house sits on, including yard/garden/driveway.")
floors = st.number_input("Floors", 1, 4, 1, step=1, format="%d")
condition = st.slider("Condition (1‑5)", 1, 5, 3)
basement = st.number_input("Basement area (sq ft)", 0, 5000, 0)
schools = st.number_input("Schools nearby", 0, 10, 2)
airport_dist = st.number_input("Distance to airport (km)", 0, 200, 50)

# ----------- Prepare final feature list -----------
# Fill other features with average/default values
# The order of features must match training model

input_row = np.array([[  # ordered same as training features
    bedrooms,
    bathrooms,
    living_area,
    lot_area,
    floors,
    0,                 # waterfront (default)
    0,                 # views (default)
    condition,
    8,                 # grade (average default)
    living_area - basement,
    basement,
    2000,              # built year (default)
    0,                 # renovation year (default)
    122001,            # postal code (default)
    52.90,             # latitude (default)
    -114.47,           # longitude (default)
    living_area,
    lot_area,
    schools,
    airport_dist
]])

# ----------- Predict and Show Result -----------
if st.button("Predict price"):
    price = model.predict(input_row)[0]
    st.success(f"Estimated Price: ₹{price:,.0f}")
