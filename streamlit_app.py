import pandas as pd
import matplotlib.pyplot as plt

def get_clean_data():
    data = pd.read_csv("PCOS_data_infertility.csv")

    # ✅ Clean column names
    data.columns = data.columns.str.strip()

    # ✅ Drop unwanted columns only if they exist
    cols_to_drop = ['Unnamed: 44', 'Sl. No', 'Patient File No.']
    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])

    return data

    # Convert PCOS column to numeric
    if data['PCOS (Y/N)'].dtype == object:
        data['PCOS (Y/N)'] = data['PCOS (Y/N)'].map({'Y': 1, 'N': 0})

    return data

def plot_data(df):
    plot = df['PCOS (Y/N)'].value_counts().plot(
        kind='bar', title="Class distributions \n(0: No | 1: Yes)")
    plot.set_xlabel("Diagnosis")
    plot.set_ylabel("Frequency")
    plt.show()
        
def get_model():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    df = get_clean_data()

    # scale predictors and split data
    X = df.drop(['PCOS (Y/N)'], axis=1)
    y = df['PCOS (Y/N)']
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test the model
    y_pred = model.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    return model, scaler
  
def create_radar_chart(input_data):
    import plotly.graph_objects as go

    input_data = get_scaled_values_dict(input_data)

    # Define radar categories
    hormonal_markers = [
        ("FSH", "FSH_mIU_mL"),
        ("LH", "LH_mIU_mL"),
        ("FSH/LH", "FSH_LH"),
        ("TSH", "TSH_mIU_L"),
        ("AMH", "AMH_ng_mL"),
        ("PRL", "PRL_ng_mL"),
        ("Vit D3", "VitD3_ng_mL"),
        ("PRG", "PRG_ng_mL"),
        ("I β-HCG", "I_beta_HCG_mIU_mL"),
        ("II β-HCG", "II_beta_HCG_mIU_mL")
    ]

    anthropometric_markers = [
        ("BMI", "BMI"),
        ("Hip", "Hip_inch"),
        ("Waist", "Waist_inch"),
        ("Waist:Hip", "Waist_Hip_Ratio"),
        ("Weight", "Weight_Kg"),
        ("Height", "Height_Cm")
    ]

    symptom_markers = [
        ("Weight gain", "Weight_gain_YN"),
        ("Hair growth", "Hair_growth_YN"),
        ("Skin darkening", "Skin_darkening_YN"),
        ("Hair loss", "Hair_loss_YN"),
        ("Pimples", "Pimples_YN")
    ]

    fig = go.Figure()

    def add_trace(group, name):
        r_values = []
        labels = []
        for label, key in group:
            val = input_data.get(key)
            if isinstance(val, str):  # Convert 'Yes'/'No' to numeric if needed
                val = 1.0 if val.lower() == 'yes' else 0.0
            r_values.append(val)
            labels.append(label)
        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=labels,
            fill='toself',
            name=name
        ))

    add_trace(hormonal_markers, "Hormonal Profile")
    add_trace(anthropometric_markers, "Anthropometric Measures")
    add_trace(symptom_markers, "Symptoms")

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        autosize=True
    )

    return fig

def create_input_form(data):
    import streamlit as st

    st.sidebar.header("Patient PCOS Dataset Details")

    slider_labels = [
        (" Age (yrs)", "Age_yrs"),
        ("Weight (Kg)", "Weight_Kg"),
        ("Height (Cm)", "Height_Cm"),
        ("BMI", "BMI"),
        ("Blood Group", "Blood_Group"),
        ("Pulse rate (bpm)", "Pulse_rate_bpm"),
        ("RR (breaths/min)", "RR_breaths_min"),
        ("Hb (g/dl)", "Hb_g_dl"),
        ("Cycle (R/I)", "Cycle_RI"),
        ("Cycle length (days)", "Cycle_length_days"),
        ("Marriage Status (Yrs)", "Marriage_Status_Yrs"),
        ("Pregnant (Y/N)", "Pregnant_YN"),
        ("No. of abortions", "No_of_abortions"),
        ("I beta-HCG (mIU/mL)", "I_beta_HCG_mIU_mL"),
        ("II beta-HCG (mIU/mL)", "II_beta_HCG_mIU_mL"),
        ("FSH (mIU/mL)", "FSH_mIU_mL"),
        ("LH (mIU/mL)", "LH_mIU_mL"),
        ("FSH/LH", "FSH_LH"),
        ("Hip (inch)", "Hip_inch"),
        ("Waist (inch)", "Waist_inch"),
        ("Waist:Hip Ratio", "Waist_Hip_Ratio"),
        ("TSH (mIU/L)", "TSH_mIU_L"),
        ("AMH (ng/mL)", "AMH_ng_mL"),
        ("PRL (ng/mL)", "PRL_ng_mL"),
        ("Vit D3 (ng/mL)", "VitD3_ng_mL"),
        ("PRG (ng/mL)", "PRG_ng_mL"),
        ("RBS (mg/dl)", "RBS_mg_dl"),
        ("Weight gain (Y/N)", "Weight_gain_YN"),
        ("Hair growth (Y/N)", "Hair_growth_YN"),
        ("Skin darkening (Y/N)", "Skin_darkening_YN"),
        ("Hair loss (Y/N)", "Hair_loss_YN"),
        ("Pimples (Y/N)", "Pimples_YN"),
        ("Fast food (Y/N)", "Fast_food_YN"),
        ("Reg. Exercise (Y/N)", "Reg_Exercise_YN"),
        ("BP Systolic (mmHg)", "BP_Systolic_mmHg"),
        ("BP Diastolic (mmHg)", "BP_Diastolic_mmHg"),
        ("Follicle No. (L)", "Follicle_No_L"),
        ("Follicle No. (R)", "Follicle_No_R"),
        ("Avg. F size (L) (mm)", "Avg_F_size_L_mm"),
        ("Avg. F size (R) (mm)", "Avg_F_size_R_mm"),
        ("Endometrium (mm)", "Endometrium_mm"),
    ]

    input_data = {}

    for label, col in slider_labels:
        try:
            input_data[col] = st.sidebar.slider(
                label,
                float(data[col].min()),
                float(data[col].max()),
                float(data[col].mean())
            )
        except:
            # For non-numeric fields like Blood Group or Y/N fields, use selectbox
            unique_vals = data[col].dropna().unique()
            input_data[col] = st.sidebar.selectbox(label, sorted(map(str, unique_vals)))

    return input_data

def get_scaled_values_dict(values_dict):
    # Define a Function to Scale the Values based on the Min and Max of the Predictor in the Training Data
    data = get_clean_data()
    X = data.drop(['PCOS (Y/N)'], axis=1)

    scaled_dict = {}

    for key, value in values_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

def display_predictions(input_data, model, scaler):
    import streamlit as st

    import numpy as np
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_data_scaled = scaler.transform(input_array)

    st.subheader('PCOS Prediction')
    st.write("PCOS %: ")

    prediction = model.predict(input_data_scaled)
    if prediction[0] == 0:
        st.write("<span class='diagnosis bright-green'>No</span>",
                 unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis bright-red'>Yes</span>",
                 unsafe_allow_html=True)

    st.write("Probability of no PCOS: ",
             model.predict_proba(input_data_scaled)[0][0])
    st.write("Probability of PCOS: ",
             model.predict_proba(input_data_scaled)[0][1])

    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def create_app():
    import streamlit as st

    st.set_page_config(page_title="Detect PCOS",
                    page_icon="👩‍⚕️", 
                    layout="wide", 
                    initial_sidebar_state="expanded")

    # load css
    with open("./assets/style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()),
                    unsafe_allow_html=True)

    with st.container():
        st.title("PCOS Diagnosis")
        st.write("Please input current medical history and symptoms. This app predicts PCOS diagnosis using Machine Learning. ")

    data = get_clean_data()
    input_data = create_input_form(data)

    model, scaler = get_model()
    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = create_radar_chart(input_data)
        st.plotly_chart(radar_chart, use_container_width=True)

    with col2:
        # load the model
        display_predictions(input_data, model, scaler)

def main():
    # EDA
    # df = get_clean_data()
    # plot_data(df)

    # MODEL
    # model = get_model()
    # print("Model: ", model)

    # APP
    create_app()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
