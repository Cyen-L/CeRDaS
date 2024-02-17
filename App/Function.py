# Import library
import psycopg2
import pandas as pd
import joblib
from io import BytesIO

def SQL_Query(query, val_tuple):

    # Format
    # "INSERT INTO your_table_name (model_name, model_version, created_at, model_data) VALUES (%s, %s, %s, %s)"
    # (model_name, model_version, created_at, model_buffer.getvalue())

    # Initialize variable
    output = None

    # Connect to your database
    conn = psycopg2.connect(
        dbname='iris-db', 
        user='cy', 
        password='cy', 
        host='iris-postgress'
    )

    # Commit the transaction
    cur = conn.cursor()
    cur.execute(query, val_tuple)

    if 'SELECT' in query:
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description] # Get the column names
        output = pd.DataFrame(rows, columns=colnames) # Convert to DataFrame
    else:
        # Commit the transaction
        conn.commit()

    # Close the cursor and connection
    cur.close()
    conn.close()

    if output is not None:
        return output

# Convert the model into DB compatible variable
def DB_Compatible_Conversion(input_model):
    
    # Initialize BytesIO
    model_buffer = BytesIO()

    # Dump the model to a BytesIO object
    joblib.dump(input_model, model_buffer) 

    # Rewind the buffer to the beginning
    model_buffer.seek(0) 

    # Create DB compatible variable
    model_file = model_buffer.getvalue()

    return model_file
