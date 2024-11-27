import sqlite3

def delete_all_data_from_prediction():
    # Connect to the database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Execute the DELETE statement to remove all rows from the Prediction table
    cursor.execute("DROP TABLE IF EXISTS Booking_data;")

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    print("All data from the Prediction table has been deleted.")
delete_all_data_from_prediction()
import sqlite3
import pandas as pd
def create_table_and_insert_xlsx(db_file_path, table_name, xlsx_file_path):
    # Step 1: Read the data from the XLSX file
    df = pd.read_excel(xlsx_file_path)  # Explicitly specify engine if needed
   
    # Step 2: Connect to the SQLite database
    conn = sqlite3.connect(db_file_path)

    # Step 3: Insert the data from the DataFrame into the database
    df.to_sql(table_name, conn, if_exists='replace', index=False)

    # Step 4: Close the database connection
    conn.close()

    print(f"Data from {xlsx_file_path} has been inserted into the table {table_name}.")

# Example usage
db_file_path = "database.db"
table_name = "Demo_data"
xlsx_file_path = "C:/Users/mihir.sinha/Downloads/one_week_data (1).xlsx"

# create_table_and_insert_xlsx(db_file_path, table_name, xlsx_file_path)
