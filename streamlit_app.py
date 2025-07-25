import pandas as pd
import matplotlib.pyplot as plt

def get_clean_data():
    data = pd.read_csv("PCOS_data_infertility.csv")

    # Clean column names
    data.columns = data.columns.str.strip()

    # Drop unwanted columns only if they exist
    cols_to_drop = ['Unnamed: 44', 'Sl. No', 'Patient File No.']
    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])

    # Convert PCOS column to numeric if needed
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
    from sklearn.preprocessing import StandardScaler

    df = get_clean_data()

    X = df.drop(['PCOS (Y/N)'], axis=1)
    y = df['PCOS (Y/N)']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    return model, scaler

def create_radar_chart(input_data):
    import plotly.graph_objects as go

    input_data = get_scaled_values_dict(input_data)

    hormonal_markers = [
        ("FSH", "FSH_mIU_mL"),
        ("LH", "LH_mIU_mL"),
        ("FSH/LH", "FSH_LH"),
        ("TSH", "TSH_mIU_L"),
        ("AMH", "AMH_ng_mL"),
        ("PRL", "PRL_ng_mL"),
        ("Vit D3", "VitD3_ng_mL"),
        ("PRG", "PRG_ng_mL"),
        ("I \u03b2-HCG", "I_beta_HCG_mIU_mL"),
        ("II \u03b2-HCG", "II_beta_HCG_mIU_mL")
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
            if isinstance(val, str):
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
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        autosize=True
    )

    return fig

def create_input_form(data):
    import streamlit as st

    st.sidebar.header("Patient PCOS Dataset Details")

    slider_labels = [
        # Same as before, unchanged for brevity...
    ]

    input_data = {}

    for label, col in slider_labels:
        col = col.strip()
        if col in data.columns:
            try:
                input_data[col] = st.sidebar.slider(
                    label,
                    float(data[col].min()),
                    float(data[col].max()),
                    float(data[col].mean())
                )
            except:
                unique_vals = data[col].dropna().unique()
                input_data[col] = st.sidebar.selectbox(label, sorted(map(str, unique_vals)))
        else:
            st.sidebar.warning(f"Column '{col}' not found in dataset.")

    return input_data

def get_scaled_values_dict(values_dict):
    data = get_clean_data()
    X = data.drop(['PCOS (Y/N)'], axis=1)

    scaled_dict = {}

    for key, value in values_dict.items():
        if key in X.columns:
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
        st.write("<span class='diagnosis bright-green'>No</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis bright-red'>Yes</span>", unsafe_allow_html=True)

    st.write("Probability of no PCOS: ", model.predict_proba(input_data_scaled)[0][0])
    st.write("Probability of PCOS: ", model.predict_proba(input_data_scaled)[0][1])
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def create_app():
    import streamlit as st

    st.set_page_config(page_title="Detect PCOS", page_icon="👩‍⚕️", layout="wide", initial_sidebar_state="expanded")

    with open("./assets/style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

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
        display_predictions(input_data, model, scaler)

def main():
    create_app()

if __name__ == '__main__':
    main()
