import streamlit as st
import pickle
import pandas as pd


def get_clean_data():
  data = pd.read_csv("Data/PCOS_data_infertility.csv")
  
  data = data.drop(['Unnamed: 44', 'Si. No', 'Patient File No.'], axis=1)
  
  data['PCOS (Y/N)'] = data['PCOS (Y/N)'].map({ 'Y': 1, 'N': 0 })
  
  return data


# Add a title
st.set_page_config(page_title="Detect PCOS",
                    page_icon="👩‍⚕️", 
                    layout="wide", 
                    initial_sidebar_state="expanded")

# Set up the structure
with st.container():
    st.title("PCOS Disgnosis")
    st.write("Please input current medical history and symptoms. This app predicts PCOS diagnosis using Machine Learning ")
    col1, col2 = st.columns([4,1])
    with col1:
        st.write("Column 1")
    with col2:
        st.write("Column 2")
        

def add_sidebar():
  st.sidebar.header("Cell Nuclei Measurements")
  
  data = get_clean_data()
  
 st.sidebar.header("Patient PCOS Dataset Details")

def create_input_form(data):
    import streamlit as st

    st.sidebar.header("Patient PCOS Dataset Details")

    slider_labels = [
        ("PCOS (Y/N)", "PCOS_YN"),
        ("Age (yrs)", "Age_yrs"),
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
