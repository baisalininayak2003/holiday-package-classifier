import streamlit as st
import numpy as np
import joblib
import pandas as pd

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Travel Package Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #F4F7FB;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #123B63;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #526579;
        text-align: center;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        background-color: white;
        border: 1px solid #DCE5EF;
    }

    div.stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 10px;
        background-color: #1261A0;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #0B4778;
        color: white;
    }

    [data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #DCE5EF;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD TRAINED MODEL AND PREPROCESSOR
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    model_data = joblib.load("model.joblib")
    preprocessor = joblib.load("preprocessor.joblib")

    # Your notebook saves the trained XGBoost model
    # inside a dictionary with the key "Xgboost".
    if isinstance(model_data, dict):
        model = model_data["Xgboost"]
    else:
        model = model_data

    return model, preprocessor


try:
    model, preprocessor = load_artifacts()

except Exception as e:
    st.error(
        "Unable to load the model or preprocessor. "
        "Check the file paths and Python library versions."
    )
    st.exception(e)
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">✈️ Travel Package Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict customer interest in purchasing a travel package '
    'using Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("🌍 Travel Insights")

    st.write(
        "This application predicts whether a customer "
        "is likely to purchase a travel package."
    )

    st.divider()

    st.subheader("How to use")

    st.write("1. Enter customer details.")
    st.write("2. Provide travel preferences.")
    st.write("3. Click the Predict button.")
    st.write("4. View the prediction result.")

    st.divider()

    st.caption("ML Model: XGBoost Classifier")
    st.caption("Frontend: Streamlit")


# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

with st.form("travel_prediction_form"):

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        marital_status = st.selectbox(
            "Marital Status",
            ["Married", "Unmarried", "Divorced"]
        )

    with col2:
        occupation = st.selectbox(
            "Occupation",
            ["Salaried", "Small Business", "Large Business"]
        )

        designation = st.selectbox(
            "Designation",
            ["Executive", "Manager", "Senior Manager",
             "AVP", "VP"]
        )

        monthly_income = st.number_input(
            "Monthly Income (₹)",
            min_value=0,
            max_value=1000000,
            value=30000,
            step=1000
        )

    with col3:
        city_tier = st.selectbox(
            "City Tier",
            [1, 2, 3]
        )

        passport = st.selectbox(
            "Has Passport?",
            ["Yes", "No"]
        )

        own_car = st.selectbox(
            "Owns a Car?",
            ["Yes", "No"]
        )

    st.divider()

    st.subheader("🧳 Travel Preferences")

    col4, col5, col6 = st.columns(3)

    with col4:
        type_of_contact = st.selectbox(
            "Type of Contact",
            ["Self Enquiry", "Company Invited"]
        )

        duration_of_pitch = st.number_input(
            "Duration of Pitch (minutes)",
            min_value=0,
            max_value=300,
            value=15
        )

        number_of_followups = st.number_input(
            "Number of Follow-ups",
            min_value=0,
            max_value=50,
            value=3
        )

    with col5:
        product_pitched = st.selectbox(
            "Product Pitched",
            ["Basic", "Deluxe", "Standard",
             "Super Deluxe", "King"]
        )

        preferred_property_star = st.selectbox(
            "Preferred Property Rating",
            [3, 4, 5]
        )

        number_of_trips = st.number_input(
            "Trips per Year",
            min_value=0,
            max_value=100,
            value=3
        )

    with col6:
        pitch_satisfaction_score = st.select_slider(
            "Pitch Satisfaction Score",
            options=[1, 2, 3, 4, 5],
            value=3
        )

        number_of_person_visiting = st.number_input(
            "Number of Adults Visiting",
            min_value=1,
            max_value=20,
            value=2
        )

        number_of_children_visiting = st.number_input(
            "Number of Children Visiting",
            min_value=0,
            max_value=20,
            value=0
        )

    st.divider()

    predict_button = st.form_submit_button(
        "🔮 Predict Travel Package Purchase"
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_button:

    # Apply the same data cleaning rules used in training.
    cleaned_marital_status = (
        "Unmarried"
        if marital_status == "Unmarried"
        else marital_status
    )

    # Create the TotalVisiting feature used in your notebook.
    total_visiting = (
        number_of_person_visiting
        + number_of_children_visiting
    )

    # Build input using the training feature names.
    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": cleaned_marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
        "TotalVisiting": total_visiting
    }])

    try:
        # Transform input with the saved preprocessor.
        transformed_data = preprocessor.transform(input_data)

        # Predict using the trained model.
        prediction = model.predict(transformed_data)[0]

        st.divider()
        st.subheader("📊 Prediction Result")

        if int(prediction) == 1:

            st.markdown("""
            <div class="result-box">
                <h2 style="color:#16803D;">
                    ✅ Likely to Purchase
                </h2>
                <p>
                    The model predicts that this customer
                    may purchase the travel package.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-box">
                <h2 style="color:#C0392B;">
                    ❌ Not Likely to Purchase
                </h2>
                <p>
                    The model predicts that this customer
                    may not purchase the travel package.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Display probability when supported by the model.
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(transformed_data)[0]

            classes = list(model.classes_)
            positive_index = classes.index(1)

            purchase_probability = probabilities[positive_index] * 100

            col_a, col_b = st.columns(2)

            with col_a:
                st.metric(
                    "Purchase Probability",
                    f"{purchase_probability:.2f}%"
                )

            with col_b:
                st.metric(
                    "Predicted Class",
                    str(int(prediction))
                )

            st.progress(
                int(round(purchase_probability))
            )

            st.caption(
                "Probability is the model's estimate, "
                "not a guarantee of customer behavior."
            )

        with st.expander("View Customer Input Data"):
            st.dataframe(
                input_data,
                use_container_width=True
            )

    except Exception as e:
        st.error(
            "Prediction failed. Check whether the input categories "
            "and preprocessing match the model used during training."
        )
        st.exception(e)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    "<center>Built with Python, XGBoost and Streamlit</center>",
    unsafe_allow_html=True
)
