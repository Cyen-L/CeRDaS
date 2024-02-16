CREATE TABLE Model (
    Id SERIAL PRIMARY KEY,
    Model_Algo VARCHAR(255) NOT NULL,
    Saved_Name VARCHAR(255) NOT NULL,
    Model_Version INTEGER NOT NULL,
    Balance_Accuracy DOUBLE PRECISION NOT NULL, 
    Generated_Time TIMESTAMP NOT NULL, 
    Model_File BYTEA NOT NULL
);

CREATE TABLE Prediction (
    Id SERIAL PRIMARY KEY,
    Model_Id INTEGER NOT NULL,
    Generated_Time TIMESTAMP NOT NULL, 
    Feature_Data JSONB NOT NULL,
    Output_Data JSONB NOT NULL,
    FOREIGN KEY (Model_Id) REFERENCES model(Id)
);