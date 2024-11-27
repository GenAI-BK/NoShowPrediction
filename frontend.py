import streamlit as st
from database import booking_data
import datetime
from Preprocessing import AppointmentNoShowPredictor
import pickle
import pandas as pd

with open('predictor2 (1).pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

db_file_path="database.db"
# Function to predict with the model

def display_slot_management(predicted_df):
    st.subheader("Slot Management Data")
    st.dataframe(predicted_df)

# Function to display Resource Management (for now, showing message)
def display_resource_management():
    st.subheader("Dashboard")

    # Fetch pre-calculated metrics from session state
    total_appointments = st.session_state.total_appointments
    predicted_show_up = st.session_state.predicted_results

    # Display total numbers
    st.write(f"The total number of appointments are: **{total_appointments}**")
    st.write(f"Total number of patients that will show up for the appointment are: **{predicted_show_up}**")
    st.write("Out of the patients that will show up, the groupings are as follows:")

    # Fetch pre-calculated grouped data from session state
    data = {
        'Metric': [
            'Age Group (<= 15)', 'Age Group (16-35)', 'Age Group (36-50)', 'Age Group (> 50)',
            'Patients with Diabetes', 'Patients with Hypertension', 'Patients with Handicap',
            'SMS Received Yes', 'SMS Received No'
        ],
        'Count': [
            st.session_state.age_group_a, st.session_state.age_group_b, st.session_state.age_group_c, st.session_state.age_group_d,
            st.session_state.diabetes_count, st.session_state.hypertension_count, st.session_state.handicap_count,
            st.session_state.sms_yes_count, st.session_state.sms_no_count
        ]
    }

    # Create and display a summary table
    summary_df = pd.DataFrame(data)
    st.table(summary_df)

# Streamlit UI components
st.title('No-Show Predictor')

# Date range input


# Correct usage with full module name
start_date = st.date_input('Start Date', min_value=datetime.datetime(2016, 1, 1), max_value=datetime.datetime(2016, 12, 31), value=datetime.datetime(2016, 1, 1))
end_date = st.date_input('End Date', min_value=datetime.datetime(2016, 1, 1), max_value=datetime.datetime(2016, 12, 31), value=datetime.datetime(2016, 12, 31))


if st.button('Predict'):
    if start_date > end_date:
        st.error('End Date cannot be before Start Date.')
    else:
        # Fetch data based on selected date range
        data = booking_data(start_date, end_date, db_file_path)

        # Perform prediction
        result_df = loaded_model.predict_csv(data)  # Assuming the model returns a numpy.ndarray
        result_df = pd.DataFrame(result_df, columns=['Predictions'])  # Ensure this is a DataFrame

        # Concatenate data and predictions
        final_df = pd.concat([data, result_df], axis=1, ignore_index=False)
        print(final_df.columns)
        # Store the predicted results in session state
        st.session_state.final_df = final_df  # Storing the final DataFrame with predictions

        # Calculate metrics for Slot Management and Resource Management
        st.session_state.total_appointments = len(final_df)
        st.session_state.predicted_results = len(final_df[final_df['Predictions'] == 0])

        st.session_state.age_group_a = len(final_df[(final_df['Age'] <= 15) & (final_df['Predictions'] == 0)])
        st.session_state.age_group_b = len(final_df[(final_df['Age'] > 15) & (final_df['Age'] <= 35) & (final_df['Predictions'] == 0)])
        st.session_state.age_group_c = len(final_df[(final_df['Age'] > 35) & (final_df['Age'] <= 50) & (final_df['Predictions'] == 0)])
        st.session_state.age_group_d = len(final_df[(final_df['Age'] > 50) & (final_df['Predictions'] == 0)])

        # Resource Management Metrics
        st.session_state.diabetes_count = len(final_df[(final_df['Diabetes'] == 1) & (final_df['Predictions'] == 0)])
        st.session_state.hypertension_count = len(final_df[(final_df['Hipertension'] == 1) & (final_df['Predictions'] == 0)])
        st.session_state.handicap_count = len(final_df[(final_df['Handcap'] == 1) & (final_df['Predictions'] == 0)])
        st.session_state.sms_yes_count = len(final_df[(final_df['SMS_received'] == 1) & (final_df['Predictions'] == 0)])
        st.session_state.sms_no_count = len(final_df[(final_df['SMS_received'] == 0) & (final_df['Predictions'] == 0)])

        st.success('Prediction completed successfully!')

# Create columns to display buttons horizontally
col1, col2 = st.columns([1, 1])

# Slot Management Button in the first column
with col1:
    if st.button('Slot Management'):
        st.session_state.slot_management_clicked = True
        st.session_state.resource_management_clicked = False
        # display_slot_management(st.session_state.final_df)

# Resource Management Button in the second column
with col2:
    if st.button('Dashboard'):
        
        st.session_state.resource_management_clicked = True
        st.session_state.slot_management_clicked = False
        display_resource_management()

# If Slot Management button was clicked, display the metrics in a table
if st.session_state.get('slot_management_clicked', False):
    if 'final_df' in st.session_state:
    # Step 1: Store the session state DataFrame into a new variable
        df_copy = st.session_state.final_df.copy()

    # Step 2: Keep only the required columns
        columns_to_keep = ['PatientId', 'AppointmentID', 'Gender', 'ScheduledDay', 'AppointmentDay', 'Age', 'Predictions']
        df_copy = df_copy[columns_to_keep]

        # Step 3: Convert AppointmentDay to datetime if it's a string
        df_copy['AppointmentDay'] = pd.to_datetime(df_copy['AppointmentDay'], errors='coerce')

        # Step 4: Define the time-to-slot mapping dictionary
        time_slot_mapping = {
            '10:00': 'Slot 1', '10:30': 'Slot 2', '11:00': 'Slot 3', '11:30': 'Slot 4',
            '12:00': 'Slot 5', '12:30': 'Slot 6', '13:00': 'Slot 7', '13:30': 'Slot 8',
            '14:00': 'Slot 9', '14:30': 'Slot 10', '15:00': 'Slot 11', '15:30': 'Slot 12',
            '16:00': 'Slot 13', '16:30': 'Slot 14', '17:00': 'Slot 15'
        }

        # Step 5: Function to map the time in AppointmentDay to a slot number using the time_slot_mapping dictionary
        def assign_slot(row):
            if pd.isna(row['AppointmentDay']):
                return None
            # Extract the time part of the AppointmentDay (HH:MM format)
            time_str = row['AppointmentDay'].strftime('%H:%M')
            # Return the corresponding slot or None if the time is not in the mapping
            return time_slot_mapping.get(time_str, None)

        # Step 6: Apply the function to the DataFrame
        df_copy['slots'] = df_copy.apply(assign_slot, axis=1)

        # Step 7: Add a new column for date to group by
        df_copy['Date'] = df_copy['AppointmentDay'].dt.date

        # Step 8: Function to assign slot numbers sequentially for each date
        def assign_sequential_slots(group):
            slot_number = 1
            for idx, row in group.iterrows():
                if pd.notna(row['slots']):
                    # Assign slot number sequentially, starting from Slot 1 for each date
                    group.at[idx, 'slots'] = f"Slot {slot_number}"
                    slot_number += 1
            return group

        # Step 9: Apply sequential slot assignment for each day
        df_copy = df_copy.groupby('Date').apply(assign_sequential_slots)

        # Step 10: Reorder columns to place 'slots' between 'Age' and 'Predictions'
        column_order = ['PatientId', 'AppointmentID', 'Gender','AppointmentDay', 'Age', 'slots', 'Predictions']
        df_copy = df_copy[column_order]
        df_copy['Gender'] = df_copy['Gender'].replace({0: 'F', 1: 'M'})

    # Step 4: Convert Predictions values
        df_copy['Predictions'] = df_copy['Predictions'].replace({
            0: 'Patient will show up',
            1: 'Patient might not show up'
        })

        # Step 5: Drop unnecessary columns (if present)
        # Check for extra columns like 'Date' or unnamed columns and remove them
        df_copy = df_copy.loc[:, ~df_copy.columns.str.contains('^Unnamed')]
        if 'Date' in df_copy.columns:
            df_copy = df_copy.drop(columns=['Date'])
        if 'Unnamed: 0' in df_copy.columns:
            df_copy = df_copy.drop(columns=['Unnamed: 0'])

        # Step 6: Reset the index to remove any extra indices from groupby or apply
        df_copy = df_copy.reset_index(drop=True)

        # Step 7: Display the updated DataFrame
        
        print(df_copy.columns)
        # Step 11: Display the updated DataFrame
        display_slot_management(df_copy)

    else:
        st.write("No data available.")