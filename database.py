import pandas as pd
import sqlite3

db_file_path="database.db"

def booking_data(start_date, end_date, db_file_path):
    # Connect to the database
    conn = sqlite3.connect(db_file_path)

    # Ensure start_date and end_date are in the correct format for SQL query
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Execute query to select data between start_date and end_date (inclusive) based on AppointmentDay
    query = f"""
    SELECT * 
    FROM Demo_data
    WHERE DATE(AppointmentDay) >= '{start_date_str}' 
    AND DATE(AppointmentDay) <= '{end_date_str}'
    """
    df = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()
    
    return df



def insert_prediction(prediction_data):
    """
    Inserts a record into the 'predictions' table in the specified SQLite database.

    Parameters:
    - db_file_path (str): Path to the SQLite database file.
    - prediction_data (dict): Dictionary with keys 'PatientId', 'AppointmentID', and 'Prediction'.
    """
    # Extract values from the dictionary
    patient_id = prediction_data.get("PatientId")
    appointment_id = prediction_data.get("AppointmentID")
    prediction = prediction_data.get("Prediction")

    # Connect to the database
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    # Insert data into the 'predictions' table
    insert_query = """
    INSERT INTO predictions (PatientId, AppointmentID, Prediction)
    VALUES (?, ?, ?)
    """
    cursor.execute(insert_query, (patient_id, appointment_id, prediction))

    # Commit the transaction and close the connection
    conn.commit()
    cursor.close()
    conn.close()
    
    




