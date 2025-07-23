from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, Mapped
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Path
from sqlalchemy import Column, Integer, String, Float, Index, DateTime
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import joblib
import io
import csv
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import os
from minio import Minio
from minio.error import S3Error
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pickle

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Database configuration
# To run on server use the following:
my_database_connection = "postgresql://dedalus:dedalus@dedalus_postgres:5432"
# To run locally use the following:
#my_database_connection = "postgresql://dedalus:dedalus@dedalus.epu.ntua.gr:5432"

engine = create_engine(my_database_connection, pool_pre_ping = True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=True, bind=engine)
Base = declarative_base()

series_list = []

# MinIO configuration
# To run locally use the following:
#MINIO_URL = os.getenv("MINIO_URL", "localhost:9000")
# To run on server use the following:
MINIO_URL = os.getenv("MINIO_URL", "s3:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "accessminio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "secretminio")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "mlflow-bucket")

minio_client = Minio(
    MINIO_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# Ensure the bucket exists
async def create_user_bucket(user_id: str, country: str):
    user_bucket = f"{user_id}-{country}".lower().replace("_", "-")
    user_bucket = ''.join(c for c in user_bucket if c.isalnum() or c == '-')
    try:
        if not minio_client.bucket_exists(user_bucket):
            minio_client.make_bucket(user_bucket)
            logging.info(f"Bucket {user_bucket} created successfully.")
        else:
            logging.info(f"Bucket {user_bucket} already exists.")
    except S3Error as err:
        logging.error(f"Failed to create or verify bucket {user_bucket}: {err}")
    return user_bucket


# -----------------------------
# Define Models
# -----------------------------
class User(Base):
    __tablename__ = "users"
    user_id = Column(String, primary_key=True, index=True)
    password = Column(String)
    country = Column(String) 
    #authenticated = Column(Boolean, default=False)

class CsvData(Base):
    __tablename__ = 'csv_data'
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime)
    consumption = Column(Float)
    production = Column(Float, nullable=True)  # Make production optional
    #production = Column(Float)
    user_id = Column(String)
    country = Column(String) 
    method = Column(String) # Aggreegated or Disaggregated
    #model_name = Column(String, unique=True, index=True)
    model_name = Column(String, index=True)

class CsvDataResponse(BaseModel):
    timestamp: datetime  # Keep as datetime, will handle serialization
    consumption: float
    production: Optional[float] = None  # Production can be None
    
    class Config:
        orm_mode = True  # Use ORM mode for automatic conversion from SQLAlchemy models


class Model(Base):
    __tablename__ = 'models'
    model_name = Column(String, primary_key=True, index=True)
    model_desc = Column(String)
    country = Column(String)
    method = Column(String) # Aggregated or Disaggregated
    user_id = Column(String)

Base.metadata.create_all(bind=engine)

# -----------------------------
# Define Schemas
# -----------------------------
class UserBase(BaseModel):
    user_id: str
    password: str
    country: str
    
    class Config:
        orm_mode = True

class ModelBase(BaseModel):
    model_name: str
    model_desc: str
    country: str
    method : str
    user_id: str

    class Config:
        orm_mode = True

# Input schema for disaggregation models
class DisaggregationModelInput(BaseModel):
    user_id: str
    country: str
    method: str
    model_names: List[str]  # List of 5 model names
    descriptions: List[str]  # List of 5 descriptions

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "country": "Greece",
                "method": "Disaggregated",
                "model_names": ["Forecasting_Model", "TV_Model", "Dishwasher_Model", "Washing_Machine_Model", "Washer_Dryer_Model"],
                "descriptions": [
                    "Forecasting energy consumption",
                    "Disaggregation for TV usage",
                    "Disaggregation for Dishwasher usage",
                    "Disaggregation for Washing Machine usage",
                    "Disaggregation for Washer-Dryer usage"
                ]
            }
        }

class CsvDataSchema(BaseModel):
    id: int
    timestamp: datetime
    consumption: float
    #production: float
    production: float = None  # Make production optional
    user_id: str
    country: str
    method : str
    model_name: str

    class Config:
        orm_mode = True

# Schema for returning CSV data
class CsvDataResponse(BaseModel):
    timestamp: datetime
    consumption: float
    production: Optional[float] = None  # Make production optional

    class Config:
        orm_mode = True

# Schema for returning monthly statistics
class MonthlyStats(BaseModel):
    month: str
    mean_consumption: float
    min_consumption: float
    max_consumption: float
    mean_production: Optional[float] = None
    min_production: Optional[float] = None
    max_production: Optional[float] = None

    class Config:
        orm_mode = True        

# Schema for returning weekly statistics
class WeeklyStats(BaseModel):
    month: str
    day_of_week: str
    mean_consumption: float
    min_consumption: float
    max_consumption: float
    mean_production: Optional[float] = None
    min_production: Optional[float] = None
    max_production: Optional[float] = None

    class Config:
        orm_mode = True

class HourlyStats(BaseModel):
    hour: int
    baseline_consumption: Optional[float] = None
    flexibility_up_consumption: Optional[float] = None
    flexibility_down_consumption: Optional[float] = None
    real_value_consumption: Optional[float] = None
    efi_up_consumption: Optional[float] = None
    efi_down_consumption: Optional[float] = None
    baseline_production: Optional[float] = None
    flexibility_up_production: Optional[float] = None
    flexibility_down_production: Optional[float] = None
    real_value_production: Optional[float] = None
    efi_up_production: Optional[float] = None
    efi_down_production: Optional[float] = None    

class DailyHourlyStats(BaseModel):
    date: str
    day_of_week: str
    hourly_stats: List[HourlyStats]

    class Config:
        orm_mode = True

# -----------------------------
# CRUD Functions
# -----------------------------        
async def get_user(db: Session, user_id: str):
    return db.query(User).filter(User.user_id == user_id).first()

async def get_model(db: Session, model_name: str, country: str, user_id: str, method: str):
    return db.query(Model).filter(Model.model_name == model_name, Model.country == country, Model.method == method, Model.user_id == user_id).first()

async def create_model(db: Session, model: ModelBase):
    db_model = Model(
        model_name=model.model_name,
        model_desc=model.model_desc,
        country=model.country,
        method=model.method,
        user_id=model.user_id
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

async def create_csv_data(db: Session, csv_data: CsvDataSchema):
    db_csv_data = CsvData(
        #id=csv_data.id,
        timestamp=csv_data.timestamp,
        consumption=csv_data.consumption,
        production=csv_data.production,
        user_id=csv_data.user_id,
        country=csv_data.country,
        method=csv_data.method,
        model_name=csv_data.model_name
    )
    db.add(db_csv_data)
    db.commit()
    db.refresh(db_csv_data)
    return db_csv_data

async def delete_csv_data(db: Session, user_id: str):
    db.query(CsvData).filter(CsvData.user_id == user_id).delete()
    db.commit()

# -----------------------------
# FastAPI application setup
# -----------------------------  

app = FastAPI(
    title="FLEXiT API",
    # docs_url=None,  # disable default docs
    # redoc_url=None,
    swagger_ui_parameters={"tryItOutEnabled": False},  # disables "Try it out"
    description="Collection of REST APIs for Serving Execution of FLEXiT",
    version="0.0.1",
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "General Model Functions",
            "description": "Endpoints for managing forecasting and disaggregation models listing and deletion."
        },
        {
            "name": "Aggregated Flexibility Models",
            "description": "Endpoints for managing aggregated forecasting, including creation, listing and retrieval."
        },
        {
            "name": "Disaggregated Flexibility Models",
            "description": "Endpoints for managing disaggregated forecasting, including creation, listing and retrieval."
        },
        {
            "name": "CSV Data Handling",
            "description": "Endpoints for uploading and deleting CSV data files, including energy consumption & production data for the aggregated and disaggregated flexibility estimation."
        },
        {
            "name": "Data Analysis",
            "description": "Endpoints for retrieving, cleaning, and analyzing CSV data, including total consumption, production, and statistical insights."
        },
        {
            "name": "Aggregated Flexibility Model Training",
            "description": "Endpoints for training forecasting models for aggregated flexibility predictions."
        },
        {
            "name": "Disaggregated Flexibility Model Training",
            "description": "Endpoints for training forecasting models for disaggregated flexibility predictions."
        },
        {
            "name": "Aggegated Flexibility Forecasting",
            "description": "Endpoints for generating aggregated energy forecasts, including total consumption, production, flexibility metrics for upcoming days."
        },
        {
            "name": "Disaggegated Flexibility Forecasting",
            "description": "Endpoints for generating disaggregated energy forecasts, flexibility metrics and appliance-level disaggregation for upcoming days."
        }
    ]
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# app.mount("/static", StaticFiles(directory="static"), name="static")
 
# # Custom Swagger UI endpoint with custom JS
# @app.get("/docs", include_in_schema=False)
# async def custom_swagger_ui():
#     html = get_swagger_ui_html(
#         openapi_url=app.openapi_url,
#         title="FLEXIBILITY TOOL API - Docs"
#     ).body.decode()

#     # Inject custom JavaScript to disable "Try it out"
#     custom_script = '<script src="/static/swagger_override.js"></script></body>'
#     html = html.replace("</body>", custom_script)

#     return HTMLResponse(content=html, status_code=200)

# -----------------------------
# Model Routes
# ----------------------------- 
# Get model names from dB

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

@app.get("/models/get/", tags=['Aggregated Flexibility Models'], response_model=List[ModelBase])
async def get_models(user_id: str, country: str, db: Session = Depends(get_db)):
    """
    Retrieve a list of model names from the database based on the provided user and country.
    
    **Description:**
    Fetches model names from the PostgreSQL database that match the specified user and country,
    and uses the hardcoded method "Aggregated".
    
    **Query Parameters:**
    - **user_id** (str): The unique identifier of the user.
    - **country** (str): The country associated with the models.
    
    **Returns:**
    - **List[ModelBase]**: A list of models matching the criteria.
    
    **Raises:**
    - **404 Not Found**: If no models are found for the given parameters.
    """
    models = db.query(Model).filter(
        Model.user_id == user_id,
        Model.country == country,
        Model.method == "Aggregated"
    ).all()

    if not models:
        raise HTTPException(status_code=404, detail="Models not found")

    return models

# Get models from MinIO
@app.get("/models/list/", tags=['General Model Functions'])
async def list_models(user_id: str, country: str, method: str):
    """
    Retrieve a list of model files stored in MinIO for the specified user, country, and method.

    **Description:**
    Lists available model files stored in MinIO under the user's designated bucket.

    **Query Parameters:**
    - **user_id** (str): The unique identifier of the user.
    - **country** (str): The country associated with the models.
    - **method** (str): The method used (e.g., 'Aggregated' or 'Disaggregated').

    **Returns:**
    - **JSON Object**: 
      ```json
      {
          "models": ["model1.h5", "model2.h5"]
      }
      ```

    **Raises:**
    - **404 Not Found**: If the user's bucket does not exist in MinIO.
    - **500 Internal Server Error**: If there is an error retrieving the model list.
    """
    user_bucket = f"{user_id}-{country}".lower().replace("_", "-")
    user_bucket = ''.join(c for c in user_bucket if c.isalnum() or c == '-')
    
    try:
        if not minio_client.bucket_exists(user_bucket):
            raise HTTPException(status_code=404, detail="User bucket does not exist")
        
        # List objects in the bucket
        objects = minio_client.list_objects(user_bucket)
        models = [obj.object_name for obj in objects]
        
        return {"models": models}
    
    except S3Error as err:
        logging.error(f"Failed to list objects in bucket {user_bucket}: {err}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {err}")    

# Create new model
@app.post("/models/create", tags=['Aggregated Flexibility Models'], response_model=ModelBase)
async def create_model_endpoint(model: ModelBase, db: Session = Depends(get_db)):
    """
    Fetch all forecasting models used for disaggregation from the database for a specific user.

    **Description:**
    Retrieves a list of forecasting models that are specifically used for disaggregation.

    **Query Parameters:**
    - **user_id** (str): The unique identifier of the user.
    - **country** (str): The country where the models are used.
    - **method** (str, default="Disaggregated"): The aggregation method (default is 'Disaggregated').

    **Returns:**
    - **List[ModelBase]**: A list of disaggregation models matching the criteria.

    **Raises:**
    - **404 Not Found**: If no models are found for the user.
    - **500 Internal Server Error**: If an error occurs during retrieval.
    """
    existing_model = await get_model(db, model_name=model.model_name, country=model.country, user_id=model.user_id, method="Aggregated")
    if existing_model:
        raise HTTPException(status_code=400, detail="Model already exists")
    
    # Check the number of existing models for the user and method
    model_count = db.query(Model).filter(Model.user_id == model.user_id, Model.method == "Aggregated").count()
    # Change to 20
    if model_count >= 10:
        raise HTTPException(status_code=400, detail="Model name limit reached. You cannot create more than 10 models per user per method.")
    
    # Create user bucket if it doesn't exist
    #create_user_bucket(model.user_id, model.country)

    # Create user bucket if it doesn't exist
    user_bucket = await create_user_bucket(model.user_id, model.country)

    # Check the number of existing models in MinIO for the user and method
    try:
        objects = minio_client.list_objects(bucket_name=user_bucket, prefix="Aggregated/", recursive=True)
        minio_model_count = sum(1 for _ in objects)
        if minio_model_count >= 10:
            raise HTTPException(status_code=400, detail="Model limit reached. You cannot create more than 10 models per user per method.")
    except S3Error as err:
        logging.error(f"Failed to list objects in bucket {user_bucket}: {err}")
        raise HTTPException(status_code=500, detail=f"MinIO error: {err}")
    
    # Create the model in PostgreSQL
    return await create_model(db, model)

@app.get("/models/disaggregation", tags=['Disaggregated Flexibility Models'], response_model=List[ModelBase])
async def get_disaggregation_models(user_id: str, country: str, db: Session = Depends(get_db)):
    """
    Fetch all the forecasting models used for disaggregation from db for a user.

    Parameters:
    - **user_id**: The unique identifier of the user.
    - **country**: The country where the models are used.
    - **method**: The aggregation method (default is 'Disaggregated').

    Returns:
    - A list of disaggregation models for the specified user.
    """
    try:
        # Hardcoded method
        method = "Disaggregated"
        # Query for disaggregation models based on the user_id, country, and method
        models = db.query(Model).filter(
            Model.user_id == user_id,
            Model.country == country,
            Model.method == method
        ).all()

        if not models:
            raise HTTPException(status_code=404, detail="No disaggregation models found for the user.")

        return models
    except Exception as e:
        logging.error(f"Error fetching disaggregation models for user_id={user_id}, country={country}, method={method}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching disaggregation models: {str(e)}")

# Create Model for Forecasting (Disaggregation)
@app.post("/models/create-forecasting-disaggregation", tags=['Disaggregated Flexibility Models'], response_model=ModelBase)
async def create_forecasting_model(model: ModelBase, db: Session = Depends(get_db)):
    """
    Create a forecasting model for disaggregation.

    **Description:**
    This endpoint allows users to create a forecasting model for disaggregation. 
    It ensures that users do not exceed the model limit and that models are stored correctly in MinIO and PostgreSQL.

    **Parameters:**
    - **model_name** (str): The name of the forecasting model.
    - **model_desc** (str): A brief description of the forecasting model.
    - **user_id** (str): The unique identifier of the user creating the model.
    - **country** (str): The country where the model will be applied.
    - **method** (str): The aggregation method (e.g., 'Aggregated' or 'Disaggregated').

    **Returns:**
    - **ModelBase**: The created forecasting model object.

    **Raises:**
    - **400 Bad Request**: If the model already exists or if the user exceeds the model creation limit.
    - **500 Internal Server Error**: If there is an error with MinIO storage.

    **Process:**
    1. **Check if the model already exists** in the database.
    2. **Ensure the user has not exceeded the model limit** (max: 10 models per method).
    3. **Create a MinIO storage bucket for the user** if it does not exist.
    4. **Verify the number of models stored in MinIO**, ensuring the limit is not exceeded.
    5. **Save the model in PostgreSQL** and return the newly created model.
    """
    method = "Disaggregated"  # Hardcoded

    # Check if the model already exists
    existing_model = db.query(Model).filter(
        Model.model_name == model.model_name,
        Model.country == model.country,
        Model.user_id == model.user_id,
        Model.method == method
    ).first()
    
    if existing_model:
        raise HTTPException(status_code=400, detail="Model already exists")

    # Check the number of existing models for the user and method
    model_count = db.query(Model).filter(Model.user_id == model.user_id, Model.method == method).count()
    if model_count >= 10:  # Adjust limit if needed
        raise HTTPException(status_code=400, detail="Model name limit reached. You cannot create more than 10 models per user per method.")

    # Ensure user bucket exists
    user_bucket = await create_user_bucket(model.user_id, model.country)

    # Check model count in MinIO
    try:
        objects = minio_client.list_objects(bucket_name=user_bucket, prefix=f"{method}/", recursive=True)
        minio_model_count = sum(1 for _ in objects)
        if minio_model_count >= 10:
            raise HTTPException(status_code=400, detail="Model limit reached in MinIO. You cannot create more than 10 models per user per method.")
    except S3Error as err:
        logging.error(f"Failed to list objects in bucket {user_bucket}: {err}")
        raise HTTPException(status_code=500, detail=f"MinIO error: {err}")

    # Create the model in PostgreSQL
    new_model = Model(
        model_name=model.model_name,
        model_desc=model.model_desc,
        country=model.country,
        method=method,
        user_id=model.user_id
    )

    db.add(new_model)
    db.commit()
    db.refresh(new_model)  # Ensure model is saved in DB

    # Convert SQLAlchemy model to Pydantic model before returning
    return ModelBase(
        model_name=new_model.model_name,
        model_desc=new_model.model_desc,
        country=new_model.country,
        method=new_model.method,
        user_id=new_model.user_id
    )

# Delete model from db and MinIO
@app.delete("/models/delete/{user_id}/{country}/{model_name}/{method}", tags=['General Model Functions'])
async def delete_model(
    user_id: str = Path(..., description="The ID of the user."),
    country: str = Path(..., description="The country."),
    model_name: str = Path(..., description="The name of the model."),
    method: str = Path(..., description="The method (Aggregated or Disaggregated)."),
    db: Session = Depends(get_db)
):
    """
    Delete a model from the database and MinIO storage.

    **Description:**
    This endpoint allows users to delete a specific model from both PostgreSQL and MinIO storage.

    **Path Parameters:**
    - **user_id** (str): The unique identifier of the user.
    - **country** (str): The country associated with the model.
    - **model_name** (str): The name of the model to delete.
    - **method** (str): The method used (e.g., 'Aggregated' or 'Disaggregated').

    **Returns:**
    - **JSON Object**: 
      ```json
      {
          "message": "Model <model_name> deleted successfully"
      }
      ```

    **Raises:**
    - **404 Not Found**: If the model is not found in the database.
    - **500 Internal Server Error**: If there is an error deleting the model from MinIO.

    **Process:**
    1. **Check if the model exists** in the PostgreSQL database.
    2. **Delete the model entry** from the database.
    3. **Remove the model file from MinIO storage**.
    4. **Return a success message** upon successful deletion.
    """
    # Delete the model from PostgreSQL
    model = db.query(Model).filter(Model.user_id == user_id, Model.country == country, Model.model_name == model_name, Model.method == method).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    db.delete(model)
    db.commit()

    # Delete the model from MinIO
    user_bucket = f"{user_id}-{country.lower()}"
    try:
        model_file_name = f"{model_name}.h5"
        minio_client.remove_object(user_bucket, model_file_name)
        logging.info(f"Model {model_file_name} deleted from bucket {user_bucket} successfully.")
    except S3Error as err:
        logging.error(f"Failed to delete model {model_file_name} from bucket {user_bucket}: {err}")
        raise HTTPException(status_code=500, detail=f"MinIO error: {err}")

    return {"message": f"Model {model_name} deleted successfully"}  

# -----------------------------
# Upload CSV Routes
# ----------------------------- 
# Upload dataset
@app.post("/upload-csv/", tags=['CSV Data Handling'])
async def upload_csv(user_id: str, country: str, model_name: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload a CSV dataset for a specified user, country, model, and "Aggregated" method.

    **Description:**
    This endpoint allows users to upload CSV files containing time-series data for consumption and optionally production.

    **Query Parameters:**
    - **user_id** (str): The unique identifier of the user.
    - **country** (str): The country associated with the dataset.
    - **model_name** (str): The name of the model.
    - **method** (str): The aggregation method (in this case: 'Aggregated').

    **File Parameters:**
    - **file** (UploadFile): The CSV file to be uploaded.

    **CSV Requirements:**
    - **Required Columns**: `Timestamp`, `Consumption`
    - **Optional Column**: `Production`
    - **Timestamp Format**: `%d/%m/%y %H:%M`
    - **Delimiter**: `;` (semicolon)

    **Returns:**
    - **JSON Object**:
      ```json
      {
          "message": "Upload successful"
      }
      ```

    **Raises:**
    - **400 Bad Request**: If the file is empty, contains incorrect columns, or has invalid data formats.
    - **500 Internal Server Error**: If an unexpected error occurs.

    **Process:**
    1. **Read and decode the CSV file.**
    2. **Validate the required columns.**
    3. **Check data formatting (timestamp & numerical values).**
    4. **Delete previous data for the user (if applicable).**
    5. **Store the new dataset in the database.**
    """
    try:
        method = "Aggregated"  # Hardcoded

        contents = await file.read()
        decoded_contents = contents.decode('utf-8-sig')  # Handle BOM

        csv_reader = csv.DictReader(io.StringIO(decoded_contents), delimiter=';')
        rows = list(csv_reader)

        if not rows:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        #required_columns = ['Timestamp', 'Consumption', 'Production']
        required_columns = ['Timestamp', 'Consumption']
        if 'Production' in csv_reader.fieldnames:
            required_columns.append('Production')

        if set(required_columns) - set(csv_reader.fieldnames):
            raise HTTPException(status_code=400, detail=f"Incorrect columns. Expected: {', '.join(required_columns)}")    


        #if csv_reader.fieldnames != required_columns:
        #    raise HTTPException(status_code=400, detail=f"Incorrect columns. Expected: {', '.join(required_columns)}")

        for row in rows:
            try:
                datetime.strptime(row['Timestamp'], '%d/%m/%y %H:%M')
                float(row['Consumption'])
                #float(row['Production'])
                if 'Production' in row and row['Production']:
                    float(row['Production'])
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Incorrect data format in row: {row}")

        # Delete existing data for the user and country if model_name doesn't already exist
        #delete_csv_data(db=db, user_id=user_id, country=country, model_name=model_name, method=method)

        # Delete existing data for the user
        delete_csv_data(db, user_id=user_id)

        for index, row in enumerate(rows):
            data = CsvData(
                user_id=user_id,
                country=country,
                model_name=model_name,
                method=method,
                timestamp=datetime.strptime(row['Timestamp'], '%d/%m/%y %H:%M'),
                consumption=float(row['Consumption']),
                #production=float(row['Production'])
                production=float(row['Production']) if 'Production' in row and row['Production'] else None

            )
            db.add(data)
        db.commit()

        # data to be used seperately for each user -- ! to be deleted after new data upload ! -- 
        # CHECK data in db, delete all

        return {"message": "Upload successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))   
    
@app.post("/upload-disaggregation-csv/", tags=['CSV Data Handling'])
async def upload_disaggregation_csv(
    user_id: str,
    country: str,
    model_name: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a CSV dataset for disaggregation modeling.

    **Description:**
    This endpoint allows users to upload CSV files containing time-series data for disaggregation.
    The file must include timestamps and consumption data, with an optional production column.

    **Query Parameters:**
    - **user_id** (str): The unique identifier of the user.
    - **country** (str): The country associated with the dataset.
    - **model_name** (str): The name of the model.
    - **method** (str): The aggregation method (in this case: 'Disaggregated').

    **File Parameters:**
    - **file** (UploadFile): The CSV file to be uploaded.

    **CSV Requirements:**
    - **Required Columns**: `Timestamp`, `Consumption`
    - **Optional Column**: `Production`
    - **Timestamp Format**: `%d/%m/%y %H:%M`
    - **Delimiter**: `;` (semicolon)
    - **Missing values**: Rows with missing or invalid data will be removed.

    **Returns:**
    - **JSON Object**:
      ```json
      {
          "message": "Disaggregation CSV upload successful!"
      }
      ```

    **Raises:**
    - **400 Bad Request**: If required columns are missing or data contains invalid formats.
    - **500 Internal Server Error**: If an unexpected error occurs during processing.

    **Process:**
    1. **Read and decode the CSV file.**
    2. **Validate the required columns.**
    3. **Convert timestamp and numerical values.**
    4. **Remove rows with missing or invalid values.**
    5. **Delete previous data for the user (if applicable).**
    6. **Store the cleaned dataset in the database.**
    """
    try:
        
        method = "Disaggregated"  # Hardcoded
        contents = await file.read()
        decoded_contents = contents.decode('utf-8-sig')  # Handle UTF-8 BOM

        # Read CSV with explicit comma delimiter
        df = pd.read_csv(io.StringIO(decoded_contents), delimiter=",", skipinitialspace=True)
        df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces from column names

        # Ensure required columns exist
        required_columns = ['Timestamp', 'Consumption']
        if 'Production' in df.columns:
            required_columns.append('Production')

        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Incorrect columns. Missing: {', '.join(missing_columns)}")

        # Convert 'Timestamp' column to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%y %H:%M', errors='coerce')

        # Drop rows with invalid timestamps
        df.dropna(subset=['Timestamp'], inplace=True)

        # Convert numeric columns
        df['Consumption'] = pd.to_numeric(df['Consumption'], errors='coerce')
        if 'Production' in df.columns:
            df['Production'] = pd.to_numeric(df['Production'], errors='coerce')

        # Remove invalid values
        df.dropna(inplace=True)

        # Delete existing data for the user before uploading new data
        delete_csv_data(db, user_id=user_id)

        # Insert new cleaned data into the database
        for _, row in df.iterrows():
            data_entry = CsvData(
                user_id=user_id,
                country=country,
                model_name=model_name,
                method=method,
                timestamp=row['Timestamp'],
                consumption=row['Consumption'],
                production=row.get('Production', None)  # Handles missing 'Production' column
            )
            db.add(data_entry)

        db.commit()

        return {"message": "Disaggregation CSV upload successful!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# CSV - Data Analysis - Routes
# ----------------------------- 
# Delete CSV data for a user
def delete_csv_data(db: Session, user_id: str):
    db.query(CsvData).filter(CsvData.user_id == user_id).delete()
    db.commit()
    
# Endpoint to delete all CSV data for a specific user
@app.delete("/delete-csv-data/{user_id}", tags=['CSV Data Handling'])
async def delete_csv_data_endpoint(user_id: str, db: Session = Depends(get_db)):
    """
    Delete all CSV data associated with a specific user.

    **Description:**
    This endpoint deletes all stored CSV data entries for a given user from the database.

    **Path Parameters:**
    - **user_id** (str): The unique identifier of the user whose data will be deleted.

    **Returns:**
    - **JSON Object**:
      ```json
      {
          "message": "Successfully deleted all CSV data for user <user_id>"
      }
      ```

    **Raises:**
    - **404 Not Found**: If no CSV data is found for the specified user.
    - **500 Internal Server Error**: If an unexpected error occurs during deletion.

    **Process:**
    1. **Query the database to find all CSV data linked to the user.**
    2. **Delete all matching records.**
    3. **Commit the deletion to the database.**
    4. **Return a success message if records were deleted.**
    """
    try:
        # Delete all data associated with the given user_id
        delete_count = db.query(CsvData).filter(CsvData.user_id == user_id).delete()
        
        # Commit the deletion to the database
        db.commit()

        if delete_count == 0:
            raise HTTPException(status_code=404, detail="No CSV data found for the given user")

        return {"message": f"Successfully deleted all CSV data for user {user_id}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to get all CSV data for a user
@app.get("/csv-data/{user_id}/", tags=['Data Analysis'], response_model=List[CsvDataResponse])
async def get_csv_data(user_id: str, db: Session = Depends(get_db)):
    """
    Retrieve all stored CSV data entries for a given user.

    **Description:**
    This endpoint fetches all CSV records associated with a specific user from the database.

    **Path Parameters:**
    - **user_id** (str): The unique identifier of the user.

    **Returns:**
    - **List[CsvDataResponse]**: A list of all stored CSV records for the user.

    **Raises:**
    - **404 Not Found**: If no CSV data is found for the specified user.
    - **500 Internal Server Error**: If an unexpected error occurs during retrieval.

    **Process:**
    1. **Query the database for all CSV records associated with the user.**
    2. **Return the list of records if found.**
    """
    try:
        csv_data = db.query(CsvData).filter(CsvData.user_id == user_id).all()
        
        if not csv_data:
            raise HTTPException(status_code=404, detail="No CSV data found for the given user")

        return csv_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to clean all CSV data for a user
@app.get("/csv-data/{user_id}/cleaned", tags=['Data Analysis'], response_model=List[CsvDataResponse])
async def get_cleaned_csv_data(user_id: str, db: Session = Depends(get_db)):
    """
    Retrieve cleaned CSV data for a given user.

    **Description:**
    This endpoint fetches and cleans the stored CSV records by:
    - Removing negative consumption and production values.
    - Eliminating "stuck" values (unchanging data points).
    - Filling missing values using linear interpolation.
    - Setting production values to zero between 20:00 and 06:00.

    **Path Parameters:**
    - **user_id** (str): The unique identifier of the user.

    **Returns:**
    - **List[CsvDataResponse]**: A list of cleaned CSV records.

    **Raises:**
    - **404 Not Found**: If no CSV data is found for the user.
    - **500 Internal Server Error**: If an unexpected error occurs.

    **Process:**
    1. **Fetch CSV data from the database.**
    2. **Remove negative values.**
    3. **Eliminate stuck values (unchanging for more than 8 timestamps).**
    4. **Interpolate missing values.**
    5. **Set nighttime production to zero.**
    6. **Return the cleaned data.**
    """
    try:
        # Fetch CSV data from the database for the given user
        csv_data = db.query(CsvData).filter(CsvData.user_id == user_id).all()

        if not csv_data:
            raise HTTPException(status_code=404, detail="No CSV data found for the given user")

        # Convert query results to DataFrame for easier cleaning
        data = [{
            'timestamp': record.timestamp,  # Match Pydantic model field names
            'consumption': record.consumption,  # Match Pydantic model field names
            'production': record.production
        } for record in csv_data]

        df = pd.DataFrame(data)

        # Data Cleaning Step 1: Remove rows where consumption is negative
        df = df[df['consumption'] >= 0]

        # If 'production' column exists, apply cleaning for 'production'
        if 'production' in df.columns:
            # Remove rows where production is negative
            df = df[(df['production'] >= 0) | df['production'].isnull()]

        # Data Cleaning Step 2: Remove rows where values are stuck for more than 8 timestamps
        def remove_stuck_values(df, column_name, threshold=24):
            df['Stuck'] = (df[column_name] != df[column_name].shift(1)).cumsum()
            group_counts = df.groupby('Stuck')[column_name].transform('size')
            return df[group_counts <= threshold].drop(columns='Stuck')

        df = remove_stuck_values(df, 'consumption', threshold=8)

        if 'production' in df.columns:
            df = remove_stuck_values(df, 'production', threshold=8)

        # Data Cleaning Step 3: Fill missing values with the mean of neighboring values
        df['consumption'] = df['consumption'].interpolate(method='linear', limit_direction='both')

        if 'production' in df.columns:
            df['production'] = df['production'].interpolate(method='linear', limit_direction='both')

        # Data Cleaning Step 4: Set production to zero for hours between 20:00 and 06:00
        df['hour'] = df['timestamp'].dt.hour  # Extract the hour from the timestamp
        df.loc[(df['hour'] >= 20) | (df['hour'] < 6), 'production'] = 0

        # Drop the 'hour' column since it's no longer needed
        df.drop(columns='hour', inplace=True)

        # Rename DataFrame columns to match the Pydantic model field names
        df.rename(columns={'timestamp': 'timestamp', 'consumption': 'consumption', 'production': 'production'}, inplace=True)

        # Convert the DataFrame back to a list of dictionaries for response
        cleaned_data = df.to_dict(orient='records')

        return cleaned_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  

def get_cleaned_data_for_user(user_id: str, db: Session):
    """
    This function fetches and cleans the CSV data for the specified user.
    """
    # Fetch CSV data from the database for the given user
    csv_data = db.query(CsvData).filter(CsvData.user_id == user_id).all()

    if not csv_data:
        raise HTTPException(status_code=404, detail="No CSV data found for the given user")

    # Convert query results to DataFrame for easier cleaning
    data = [{
        'timestamp': record.timestamp,  
        'consumption': record.consumption,  
        'production': record.production
    } for record in csv_data]

    df = pd.DataFrame(data)

    # Data Cleaning Step 1: Remove rows where consumption is negative
    df = df[df['consumption'] >= 0]

    # If 'production' column exists, apply cleaning for 'production'
    if 'production' in df.columns:
        df = df[(df['production'] >= 0) | df['production'].isnull()]

    # Data Cleaning Step 2: Remove stuck values
    def remove_stuck_values(df, column_name, threshold=24):
        df['Stuck'] = (df[column_name] != df[column_name].shift(1)).cumsum()
        group_counts = df.groupby('Stuck')[column_name].transform('size')
        return df[group_counts <= threshold].drop(columns='Stuck')

    df = remove_stuck_values(df, 'consumption', threshold=48)

    if 'production' in df.columns:
        df = remove_stuck_values(df, 'production', threshold=48)

    # Data Cleaning Step 3: Fill missing values with the mean of neighboring values
    df['consumption'] = df['consumption'].interpolate(method='linear', limit_direction='both')

    if 'production' in df.columns:
        df['production'] = df['production'].interpolate(method='linear', limit_direction='both')


    # Data Cleaning Step 4: Set production to zero for hours between 20:00 and 06:00
    df['hour'] = df['timestamp'].dt.hour
    df.loc[(df['hour'] >= 20) | (df['hour'] < 6), 'production'] = 0

    df.drop(columns='hour', inplace=True)

    return df  # Return the cleaned DataFrame

# # Clean data for disaggregation
@app.get("/csv-data/{user_id}/cleaned-disaggregated", tags=['Data Analysis'], response_model=List[CsvDataResponse])
async def get_cleaned_disaggregated_csv_data(user_id: str, db: Session = Depends(get_db)):
    """
    Clean CSV data for disaggregated energy modeling:
    - Adds datetime features
    - Removes invalid data
    - Adds rolling/lags
    - Normalizes
    """

    # Step 1: Fetch Data
    csv_data = db.query(CsvData).filter(CsvData.user_id == user_id).all()
    if not csv_data:
        raise HTTPException(status_code=404, detail="No CSV data found for the given user")

    # Step 2: Create DataFrame
    df = pd.DataFrame([{
        'timestamp': row.timestamp,
        'consumption': row.consumption,
        'production': row.production
    } for row in csv_data])

    print(f"Loaded {len(df)} rows")
    print("Sample data:")
    print(df.head())

    # Step 3: Type Conversion
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['consumption'] = pd.to_numeric(df['consumption'], errors='coerce')
    df['production'] = pd.to_numeric(df['production'], errors='coerce')

    df = df.dropna(subset=['timestamp', 'consumption'])
    df = df[df['consumption'] >= 0]
    print(f"After basic cleaning: {len(df)} rows")

    if df.empty:
        raise HTTPException(status_code=500, detail="No valid consumption data after cleaning.")

    # Step 4: Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Step 5: Rolling Stats & Lag Features
    def create_features(df, col, lags=[1, 5, 10, 30], win_short=10, win_long=60):
        df[f'rolling_mean_{win_short}_{col}'] = df[col].rolling(window=win_short, min_periods=1).mean()
        df[f'rolling_std_{win_short}_{col}'] = df[col].rolling(window=win_short, min_periods=1).std()
        df[f'rolling_max_{win_short}_{col}'] = df[col].rolling(window=win_short, min_periods=1).max()
        df[f'rolling_min_{win_short}_{col}'] = df[col].rolling(window=win_short, min_periods=1).min()

        df[f'rolling_mean_{win_long}_{col}'] = df[col].rolling(window=win_long, min_periods=1).mean()
        df[f'rolling_std_{win_long}_{col}'] = df[col].rolling(window=win_long, min_periods=1).std()

        for lag in lags:
            df[f'lag_{lag}_{col}'] = df[col].shift(lag)

    # Apply to consumption
    create_features(df, 'consumption')

    # Apply to production only if it contains usable values
    if 'production' in df.columns and df['production'].notna().any() and df['production'].ge(0).any():
        df = df[df['production'].ge(0)]
        create_features(df, 'production')
    else:
        df.drop(columns=['production'], inplace=True, errors='ignore')
        print("Production column dropped (no valid data)")

    print(f"Before final dropna: {len(df)} rows")
    df = df.dropna()
    print(f"Final cleaned rows: {len(df)}")

    if df.empty:
        raise HTTPException(status_code=500, detail="Data became empty after rolling/lags.")

    # # Step 6: Normalize
    # try:
    #     df['consumption'] = MinMaxScaler().fit_transform(df[['consumption']])
    #     if 'production' in df.columns:
    #         df['production'] = MinMaxScaler().fit_transform(df[['production']])
    # except ValueError as e:
    #     raise HTTPException(status_code=500, detail=f"Scaling error: {e}")

    # print("Data cleaning and feature engineering complete.")

    return df.to_dict(orient='records')

    
@app.get("/csv-data/{user_id}/monthly-stats", tags=['Data Analysis'], response_model=List[MonthlyStats])
async def get_monthly_stats(user_id: str, db: Session = Depends(get_db)):
    """
    Retrieve monthly consumption and production statistics.

    **Description:**
    This endpoint calculates and returns monthly statistics including:
    - Mean, min, and max consumption.
    - Mean, min, and max production (if available).

    **Path Parameters:**
    - **user_id** (str): The unique identifier of the user.

    **Returns:**
    - **List[MonthlyStats]**: Monthly statistics for the user.

    **Raises:**
    - **404 Not Found**: If no data is available.
    - **500 Internal Server Error**: If an unexpected error occurs.
    """
    try:
        csv_data = db.query(CsvData).filter(CsvData.user_id == user_id).all()
        
        if not csv_data:
            raise HTTPException(status_code=404, detail="No CSV data found for the given user")
        
        # Convert query results to DataFrame
        data = [{
            'Timestamp': record.timestamp,
            'Consumption': record.consumption,
            'Production': record.production
        } for record in csv_data]

        df = pd.DataFrame(data)
        df['Month'] = df['Timestamp'].dt.to_period('M')

        # Calculate monthly statistics
        grouped = df.groupby('Month').agg(
            mean_consumption=pd.NamedAgg(column='Consumption', aggfunc='mean'),
            min_consumption=pd.NamedAgg(column='Consumption', aggfunc='min'),
            max_consumption=pd.NamedAgg(column='Consumption', aggfunc='max'),
            mean_production=pd.NamedAgg(column='Production', aggfunc='mean') if 'Production' in df.columns else None,
            min_production=pd.NamedAgg(column='Production', aggfunc='min') if 'Production' in df.columns else None,
            max_production=pd.NamedAgg(column='Production', aggfunc='max') if 'Production' in df.columns else None
        ).reset_index()

        # Fill missing values for production statistics with None
        if 'mean_production' in grouped.columns:
            grouped['mean_production'] = grouped['mean_production'].where(pd.notnull(grouped['mean_production']), None)
            grouped['min_production'] = grouped['min_production'].where(pd.notnull(grouped['min_production']), None)
            grouped['max_production'] = grouped['max_production'].where(pd.notnull(grouped['max_production']), None)
        else:
            grouped['mean_production'] = None
            grouped['min_production'] = None
            grouped['max_production'] = None

        # Convert results to list of MonthlyStats
        stats = [
            MonthlyStats(
                month=str(row['Month']),
                mean_consumption=row['mean_consumption'],
                min_consumption=row['min_consumption'],
                max_consumption=row['max_consumption'],
                mean_production=row['mean_production'],
                min_production=row['min_production'],
                max_production=row['max_production']
            )
            for index, row in grouped.iterrows()
        ]

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/csv-data/{user_id}/weekly-stats", tags=['Data Analysis'], response_model=List[WeeklyStats])
async def get_weekly_stats(user_id: str, db: Session = Depends(get_db)):
    """
    Retrieve weekly consumption and production statistics.

    **Description:**
    This endpoint calculates and returns weekly statistics including:
    - Mean, min, and max consumption.
    - Mean, min, and max production (if available).
    - Grouped by month and day of the week.

    **Path Parameters:**
    - **user_id** (str): The unique identifier of the user.

    **Returns:**
    - **List[WeeklyStats]**: Weekly statistics for the user.

    **Raises:**
    - **404 Not Found**: If no data is available.
    - **500 Internal Server Error**: If an unexpected error occurs.
    """
    try:
        csv_data = db.query(CsvData).filter(CsvData.user_id == user_id).all()
        
        if not csv_data:
            raise HTTPException(status_code=404, detail="No CSV data found for the given user")
        
        # Convert query results to DataFrame
        data = [{
            'Timestamp': record.timestamp,
            'Consumption': record.consumption,
            'Production': record.production
        } for record in csv_data]

        df = pd.DataFrame(data)
        df['Month'] = df['Timestamp'].dt.to_period('M')
        df['Day_of_Week'] = df['Timestamp'].dt.day_name()

        # Calculate weekly statistics
        grouped = df.groupby(['Month', 'Day_of_Week']).agg(
            mean_consumption=pd.NamedAgg(column='Consumption', aggfunc='mean'),
            min_consumption=pd.NamedAgg(column='Consumption', aggfunc='min'),
            max_consumption=pd.NamedAgg(column='Consumption', aggfunc='max'),
            mean_production=pd.NamedAgg(column='Production', aggfunc='mean') if 'Production' in df.columns else None,
            min_production=pd.NamedAgg(column='Production', aggfunc='min') if 'Production' in df.columns else None,
            max_production=pd.NamedAgg(column='Production', aggfunc='max') if 'Production' in df.columns else None
        ).reset_index()

        # Fill missing values for production statistics with None
        if 'mean_production' in grouped.columns:
            grouped['mean_production'] = grouped['mean_production'].where(pd.notnull(grouped['mean_production']), None)
            grouped['min_production'] = grouped['min_production'].where(pd.notnull(grouped['min_production']), None)
            grouped['max_production'] = grouped['max_production'].where(pd.notnull(grouped['max_production']), None)
        else:
            grouped['mean_production'] = None
            grouped['min_production'] = None
            grouped['max_production'] = None

        # Convert results to list of WeeklyStats
        stats = [
            WeeklyStats(
                month=str(row['Month']),
                day_of_week=row['Day_of_Week'],
                mean_consumption=row['mean_consumption'],
                min_consumption=row['min_consumption'],
                max_consumption=row['max_consumption'],
                mean_production=row['mean_production'],
                min_production=row['min_production'],
                max_production=row['max_production']
            )
            for index, row in grouped.iterrows()
        ]

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Total Consumption Stats endpoint (dataset)
@app.get("/csv-data/{user_id}/total-consumption", tags=['Data Analysis'])
async def get_total_consumption(user_id: str, db: Session = Depends(get_db)):
    """
    Retrieve total energy consumption in kWh.

    **Description:**
    This endpoint calculates the total energy consumption for a user.

    **Path Parameters:**
    - **user_id** (str): The unique identifier of the user.

    **Returns:**
    - **JSON Object**:
      ```json
      {
          "total_consumption": 12.34
      }
      ```

    **Raises:**
    - **404 Not Found**: If no data is found.
    - **500 Internal Server Error**: If an error occurs.
    """
    try:
        csv_data = db.query(CsvData).filter(CsvData.user_id == user_id).all()

        if not csv_data:
            raise HTTPException(status_code=404, detail="No CSV data found for the given user")

        total_consumption_kwh = sum(record.consumption for record in csv_data)
        total_consumption = round(total_consumption_kwh / 1000, 2)  # Convert to MWh and round to 2 decimal places

        return {"total_consumption_kWh": total_consumption}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Total Production Stats endpoint (dataset)
@app.get("/csv-data/{user_id}/total-production", tags=['Data Analysis'])
async def get_total_production(user_id: str, db: Session = Depends(get_db)):
    """
    Retrieve total energy production in kWh.

    **Description:**
    This endpoint calculates the total energy production for a user.

    **Path Parameters:**
    - **user_id** (str): The unique identifier of the user.

    **Returns:**
    - **JSON Object**:
      ```json
      {
          "total_production": 8.56
      }
      ```

    **Raises:**
    - **404 Not Found**: If no data is found.
    - **500 Internal Server Error**: If an error occurs.
    """
    try:
        csv_data = db.query(CsvData).filter(CsvData.user_id == user_id).all()

        if not csv_data:
            raise HTTPException(status_code=404, detail="No CSV data found for the given user")

        total_production_kwh = sum(record.production for record in csv_data if record.production is not None)
        total_production = round(total_production_kwh / 1000, 2)  # Convert to MWh and round to 2 decimal places

        return {"total_production_kWh": total_production}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Max consumption and max production days
@app.get("/csv-data/{user_id}/max-consumption-production-day", tags=['Data Analysis'])
async def get_max_consumption_production_day(user_id: str, db: Session = Depends(get_db)):
    """
    Retrieve the days with the highest consumption and production.

    **Description:**
    This endpoint finds the days with the maximum recorded energy consumption and production.

    **Path Parameters:**
    - **user_id** (str): The unique identifier of the user.

    **Returns:**
    - **JSON Object**:
      ```json
      {
          "max_consumption_day": {
              "date": "2023-06-15",
              "total_consumption_kw": 1023.5
          },
          "max_production_day": {
              "date": "2023-06-20",
              "total_production_kw": 850.4
          }
      }
      ```

    **Raises:**
    - **404 Not Found**: If no data is found.
    - **500 Internal Server Error**: If an error occurs.
    """
    try:
        csv_data = db.query(CsvData).filter(CsvData.user_id == user_id).all()

        if not csv_data:
            raise HTTPException(status_code=404, detail="No CSV data found for the given user")

        # Convert query results to DataFrame
        data = [{
            'Timestamp': record.timestamp,
            'Consumption': record.consumption,
            'Production': record.production
        } for record in csv_data]

        df = pd.DataFrame(data)

        # Ensure Timestamp is datetime and extract Date
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])

        df['Date'] = df['Timestamp'].dt.date

        # Group by Date and sum Consumption and Production
        grouped = df.groupby('Date').agg({
            'Consumption': 'sum',
            'Production': 'sum'
        }).reset_index()

        # Ensure numeric types and drop problematic rows
        grouped['Consumption'] = pd.to_numeric(grouped['Consumption'], errors='coerce')
        grouped['Production'] = pd.to_numeric(grouped['Production'], errors='coerce')
        grouped = grouped.dropna(subset=['Consumption', 'Production'])

        if grouped.empty:
            raise HTTPException(status_code=404, detail="No valid consumption/production data found.")

        # Compute maximums
        max_consumption_day = grouped.loc[grouped['Consumption'].idxmax()]
        max_production_day = grouped.loc[grouped['Production'].idxmax()]

        return {
            "max_consumption_day": {
                "date": str(max_consumption_day['Date']),
                "total_consumption_kWh": round(max_consumption_day['Consumption'] / 1000, 2)
            },
            "max_production_day": {
                "date": str(max_production_day['Date']),
                "total_production_kWh": round(max_production_day['Production'] / 1000, 2)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



# Data Duration endpoint (dataset)        
@app.get("/csv-data/{user_id}/data-duration", tags=['Data Analysis'])
async def get_data_duration(user_id: str, db: Session = Depends(get_db)):
    """
    Retrieve the duration of the available CSV dataset for a user.

    **Description:**
    This endpoint calculates the total duration (in days) of the stored energy dataset for a user,
    based on the earliest and latest timestamps.

    **Path Parameters:**
    - **user_id** (str): The unique identifier of the user.

    **Returns:**
    - **JSON Object**:
      ```json
      {
          "data_duration_days": 365
      }
      ```

    **Raises:**
    - **404 Not Found**: If no data is found.
    - **400 Bad Request**: If no timestamps are found in the dataset.
    - **500 Internal Server Error**: If an unexpected error occurs.

    **Process:**
    1. **Fetch CSV data from the database.**
    2. **Extract timestamps and determine start and end dates.**
    3. **Calculate duration in days.**
    """
    try:
        csv_data = db.query(CsvData).filter(CsvData.user_id == user_id).all()

        if not csv_data:
            raise HTTPException(status_code=404, detail="No CSV data found for the given user")

        timestamps = [record.timestamp for record in csv_data]
        if not timestamps:
            raise HTTPException(status_code=400, detail="No timestamps found in the CSV data")

        start_date = min(timestamps)
        end_date = max(timestamps)
        duration = (end_date - start_date).days

        return {"data_duration_days": duration}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# User Load/Production Profile
@app.get("/csv-data/{user_id}/average-daily-values", tags=['Data Analysis'], response_model=dict)
async def get_average_daily_values(user_id: str, db: Session = Depends(get_db)):
    """
    Retrieve the average daily energy consumption and production.

    **Description:**
    This endpoint calculates the average daily energy consumption and production for a user
    based on their stored CSV data.

    **Path Parameters:**
    - **user_id** (str): The unique identifier of the user.

    **Returns:**
    - **JSON Object**:
      ```json
      {
          "Average Daily Consumption (kWh)": 12.34,
          "Average Daily Production (kWh)": 8.56
      }
      ```

    **Raises:**
    - **404 Not Found**: If no data is found.
    - **500 Internal Server Error**: If an unexpected error occurs.

    **Process:**
    1. **Fetch CSV data from the database.**
    2. **Convert the data into a Pandas DataFrame for processing.**
    3. **Aggregate daily consumption and production.**
    4. **Compute the average values over all days.**
    """
    try:
        # Fetch CSV data
        csv_data = db.query(CsvData).filter(CsvData.user_id == user_id).all()

        if not csv_data:
            raise HTTPException(status_code=404, detail="No CSV data found for the given user")

        # Convert to DataFrame
        data = [{
            'Timestamp': record.timestamp,
            'Consumption': record.consumption,
            'Production': record.production
        } for record in csv_data]

        df = pd.DataFrame(data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Date'] = df['Timestamp'].dt.date

        # Group by date
        daily_totals = df.groupby('Date').agg(
            daily_consumption=('Consumption', 'sum'),
            daily_production=('Production', 'sum')
        ).reset_index()

        # Calculate averages
        average_daily_consumption = daily_totals['daily_consumption'].mean()
        average_daily_production = daily_totals['daily_production'].mean()

        return {
            'Average Daily Consumption (kWh)': round(average_daily_consumption / 1000, 2),
            'Average Daily Production (kWh)': round(average_daily_production / 1000, 2) if not pd.isna(average_daily_production) else None # converted in kWh
        }

    except Exception as e:
        logging.error(f"Error in average daily values: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Flexibility stats endpoint
@app.get("/csv-data/{user_id}/last-week-flexibility-efi-stats", tags=['Data Analysis'], response_model=List[DailyHourlyStats])
async def get_last_week_flexibility_efi_stats(user_id: str, db: Session = Depends(get_db)):
    """
    Retrieve Energy Flexibility Indicators for the last week.

    **Description:**
    This endpoint calculates energy flexibility indicators based on the last week's data.
    It computes:
    - **Baseline consumption & production**: Average values over the past 30 days for the same day-of-week and hour.
    - **Flexibility up & down values**: Maximum and minimum observed values over the last 30 days.
    - **Energy Flexibility Indicators (EFIs)**: The percentage change compared to the baseline.

    **Path Parameters:**
    - **user_id** (str): The unique identifier of the user.

    **Returns:**
    - **List[DailyHourlyStats]**: A structured list of daily and hourly flexibility statistics.

    **Raises:**
    - **404 Not Found**: If no data is available.
    - **500 Internal Server Error**: If an unexpected error occurs.

    **Process:**
    1. **Fetch and clean CSV data for the user.**
    2. **Extract dates, weekdays, and hours from timestamps.**
    3. **Compute baseline values from the last 30 days.**
    4. **Calculate flexibility up/down values and Energy Flexibility Indicators (EFIs).**
    5. **Format and return results per day and hour.**
    """
    try:
        # Fetch and clean the CSV data for the given user
        df = get_cleaned_data_for_user(user_id, db)

        df['Date'] = df['timestamp'].dt.date
        df['Day_of_Week'] = df['timestamp'].dt.day_name()
        df['Hour'] = df['timestamp'].dt.hour

        last_date = df['Date'].max()
        last_week_dates = pd.date_range(end=last_date, periods=7).date
        last_30_days = pd.date_range(end=last_date, periods=30).date

        results = []

        for date in last_week_dates:
            day_of_week = pd.Timestamp(date).day_name()
            daily_data = df[df['Date'] == date]
            
            if daily_data.empty:
                continue

            hourly_stats = []

            for hour in range(24):
                hour_data = daily_data[daily_data['Hour'] == hour]

                if hour_data.empty:
                    continue

                past_30_days_data = df[(df['Day_of_Week'] == day_of_week) & (df['Date'].isin(last_30_days)) & (df['Hour'] == hour)]

                # Calculate consumption statistics
                baseline_consumption = round(past_30_days_data['consumption'].mean(), 2)
                flexibility_up_consumption = round(past_30_days_data['consumption'].max(), 2)
                flexibility_down_consumption = round(past_30_days_data['consumption'].min(), 2)
                real_value_consumption = round(hour_data['consumption'].values[0], 2) if not hour_data.empty else None

                efi_up_consumption = round((flexibility_up_consumption - baseline_consumption) * 100 / baseline_consumption, 2) if baseline_consumption != 0 else None
                efi_down_consumption = round(abs(flexibility_down_consumption - baseline_consumption) * 100 / baseline_consumption, 2) if baseline_consumption != 0 else None

                # Calculate production statistics if data is available
                if 'production' in df.columns and not df['production'].isnull().all():
                    baseline_production = round(past_30_days_data['production'].mean(), 2)
                    flexibility_up_production = round(past_30_days_data['production'].max(), 2)
                    flexibility_down_production = round(past_30_days_data['production'].min(), 2)
                    real_value_production = round(hour_data['production'].values[0], 2) if 'production' in hour_data.columns and not hour_data['production'].isnull().all() else None

                    efi_up_production = round((flexibility_up_production - baseline_production) * 100 / baseline_production, 2) if baseline_production != 0 else None
                    efi_down_production = round(abs(flexibility_down_production - baseline_production) * 100 / baseline_production, 2) if baseline_production != 0 else None
                else:
                    baseline_production = None
                    flexibility_up_production = None
                    flexibility_down_production = None
                    real_value_production = None
                    efi_up_production = None
                    efi_down_production = None

                hourly_stats.append(HourlyStats(
                    hour=hour,
                    baseline_consumption=baseline_consumption,
                    flexibility_up_consumption=flexibility_up_consumption,
                    flexibility_down_consumption=flexibility_down_consumption,
                    real_value_consumption=real_value_consumption,
                    efi_up_consumption=efi_up_consumption,
                    efi_down_consumption=efi_down_consumption,
                    baseline_production=baseline_production,
                    flexibility_up_production=flexibility_up_production,
                    flexibility_down_production=flexibility_down_production,
                    real_value_production=real_value_production,
                    efi_up_production=efi_up_production,
                    efi_down_production=efi_down_production
                ))

            results.append(DailyHourlyStats(
                date=str(date),
                day_of_week=day_of_week,
                hourly_stats=hourly_stats
            ))

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Model Training - Routes
# ----------------------------- 

@app.post("/train-prosumer-model/", tags=['Aggregated Flexibility Model Training'])
async def train_prosumer_model(user_id: str, country: str, model_name: str, db: Session = Depends(get_db)):
    """
    Train an LSTM-based forecasting model for energy consumption and production.

    **Description:**
    - This endpoint trains a deep learning model to forecast **both** energy consumption and production.
    - The model uses an LSTM architecture with **aggregated** energy data (i.e., consumption & production).
    - The trained model is uploaded to MinIO storage for future inference.

    **Parameters:**
    - **user_id** (str): The unique identifier of the user.
    - **country** (str): The country associated with the dataset.
    - **model_name** (str): The name of the model to be trained.
    - **method** (str): The aggregation method (`Aggregated`).
    - **db** (Session): Database session dependency.

    **Returns:**
    - **JSON Object**:
      ```json
      {
          "message": "Prosumer model training successful"
      }
      ```

    **Raises:**
    - **500 Internal Server Error**: If an error occurs during model training.
    """
    try:
        method = "Aggregated"  # Hardcoded
        model = await train_model_logic(user_id, country, model_name, method, db, only_consumption=False)
        return {"message": "Prosumer model training successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

####### Second Attemp

async def train_model_logic(user_id: str, country: str, model_name: str, method: str, db: Session, only_consumption: bool = True):
    # Fetch and clean the data using the existing get_cleaned_data_for_user function
    df = get_cleaned_data_for_user(user_id, db)

    if df.empty:
        raise HTTPException(status_code=400, detail="No valid data available after cleaning")

    # Ensure the DataFrame has the necessary columns based on only_consumption flag
    if only_consumption:
        df = df[['timestamp', 'consumption']]  # Keep only consumption data
    else:
        df = df[['timestamp', 'consumption', 'production']]  # Keep both consumption and production

    # Preprocess the data
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['Year'] = df['timestamp'].dt.year
    df['Month'] = df['timestamp'].dt.month
    df['Hour'] = df['timestamp'].dt.hour
    df['Date'] = df['timestamp'].dt.date
    df['Day_of_Week'] = df['timestamp'].dt.dayofweek

    consumed_energy = df['consumption'].values.reshape(-1, 1)
    additional_features = df[['Month', 'Hour', 'Day_of_Week']]

    scaler = MinMaxScaler()
    additional_features_normalized = scaler.fit_transform(additional_features)
    consumed_energy_normalized = scaler.fit_transform(consumed_energy)

    if not only_consumption and 'production' in df.columns:
        produced_energy = df['production'].values.reshape(-1, 1)
        produced_energy_normalized = scaler.fit_transform(produced_energy)
        combined_energy_normalized = np.hstack([consumed_energy_normalized, produced_energy_normalized])
    else:
        combined_energy_normalized = consumed_energy_normalized

    # Create sequences for training
    look_back_per_day = 24
    X_train, y_train = await create_sequences_daily_with_features(
        combined_energy_normalized, additional_features_normalized, look_back=look_back_per_day)

    logging.info(f"X_train shape: {X_train.shape}")
    logging.info(f"y_train shape: {y_train.shape}")

    # Ensure y_train has the correct shape
    y_train = y_train.reshape(-1, look_back_per_day * (2 if not only_consumption and 'production' in df.columns else 1))

    logging.info(f"y_train reshaped: {y_train.shape}")

    # Align the lengths of X_train and y_train
    if len(X_train) > len(y_train):
        X_train = X_train[:len(y_train)]
    elif len(y_train) > len(X_train):
        y_train = y_train[:len(X_train)]

    logging.info(f"Aligned X_train shape: {X_train.shape}")
    logging.info(f"Aligned y_train shape: {y_train.shape}")

    # LSTM Model
    model = Sequential()
    model.add(LSTM(units=80, activation="relu", return_sequences=True, input_shape=(look_back_per_day, X_train.shape[2])))
    model.add(LSTM(units=20, activation="relu", return_sequences=True))
    model.add(LSTM(units=10, activation="relu", return_sequences=False))
    model.add(Dense(units=24 * (2 if not only_consumption and 'production' in df.columns else 1))) 
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Log with MLflow
    with mlflow.start_run():
        mlflow.tensorflow.autolog()
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=2, validation_split=0.2, callbacks=[early_stopping])

        # Save the model locally
        model_dir = "/tmp/model"
        os.makedirs(model_dir, exist_ok=True)
        model_file_name = f"{model_name}.h5"
        model_path = os.path.join(model_dir, model_file_name)
        model.save(model_path)

        # Get the user-specific bucket
        user_bucket = await create_user_bucket(user_id, country)

        # Upload the model to the user-specific MinIO bucket
        minio_client.fput_object(user_bucket, model_file_name, model_path)

        logging.info(f"Model {model_file_name} uploaded to bucket {user_bucket}")

    return model

# Function to create sequences for training with features
async def create_sequences_daily_with_features(dataset, additional_features, look_back=24):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        seq_X = dataset[i:(i + look_back), :]
        seq_features = additional_features[i + look_back - 1]
        combined_seq_X = np.concatenate((seq_X, np.tile(seq_features, (look_back, 1))), axis=1)
        dataX.append(combined_seq_X)
        if (i + look_back + look_back) <= len(dataset):
            seq_Y = dataset[i + look_back:i + look_back + look_back, :]
            dataY.append(seq_Y.flatten())
    return np.array(dataX), np.array(dataY)

# Train Model for disaggregation forecast
@app.post("/train-disaggregated-forecast", tags=['Disaggregated Flexibility Model Training'])
async def train_disaggregated_forecast(user_id: str, country: str, model_name: str, db: Session = Depends(get_db)):
    """
    Train a BiLSTM-based forecasting model for consumption and production (Disaggregated Method).

    **Description:**
    - This endpoint trains a deep learning model specifically for consumption data needed for disaggregation.
    - If production data exists, a separate model is trained for production.
    - The trained models are saved in MinIO for future inference.

    **Parameters:**
    - **user_id** (str): The unique identifier of the user.
    - **country** (str): The country associated with the dataset.
    - **model_name** (str): The name of the model to be trained.
    - **db** (Session): Database session dependency.

    **Returns:**
    - **JSON Object**:
      ```json
      {
          "message": "Disaggregated forecasting models trained successfully",
          "consumption_model": {
              "mse": 0.012,
              "mae": 0.045,
              "r2": 0.91
          },
          "production_model": {
              "mse": 0.015,
              "mae": 0.052,
              "r2": 0.87
          }
      }
      ```

    **Raises:**
    - **400 Bad Request**: If no valid data is available after cleaning.
    - **500 Internal Server Error**: If an error occurs during model training.
    """
    df = await get_cleaned_disaggregated_csv_data(user_id, db)
    df = pd.DataFrame(df)
    if df.empty:
        raise HTTPException(status_code=400, detail="No valid data available after cleaning")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'consumption' not in df.columns:
        raise HTTPException(status_code=400, detail="Missing 'consumption' column in dataset.")

    # Feature definitions
    base_features = [
        'hour', 'day_of_week', 'month', 'day', 'is_weekend', 'consumption',
        'rolling_mean_10min_consumption', 'rolling_std_10min_consumption', 'rolling_max_10min_consumption',
        'rolling_min_10min_consumption', 'rolling_mean_60min_consumption', 'rolling_std_60min_consumption',
        'lag_1min_consumption', 'lag_5min_consumption', 'lag_10min_consumption', 'lag_30min_consumption'
    ]
    prod_features = [
        'production', 'rolling_mean_10min_production', 'rolling_std_10min_production',
        'rolling_max_10min_production', 'rolling_min_10min_production',
        'rolling_mean_60min_production', 'rolling_std_60min_production',
        'lag_1min_production', 'lag_5min_production', 'lag_10min_production', 'lag_30min_production'
    ]

    has_production = 'production' in df.columns and not df['production'].isna().all()
    feature_columns = base_features + (prod_features if has_production else [])

    # Create features
    def create_features(df, col):
        df[f'rolling_mean_10min_{col}'] = df[col].rolling(window=10, min_periods=1).mean()
        df[f'rolling_std_10min_{col}'] = df[col].rolling(window=10, min_periods=1).std()
        df[f'rolling_max_10min_{col}'] = df[col].rolling(window=10, min_periods=1).max()
        df[f'rolling_min_10min_{col}'] = df[col].rolling(window=10, min_periods=1).min()
        df[f'rolling_mean_60min_{col}'] = df[col].rolling(window=60, min_periods=1).mean()
        df[f'rolling_std_60min_{col}'] = df[col].rolling(window=60, min_periods=1).std()
        for lag in [1, 5, 10, 30]:
            df[f'lag_{lag}min_{col}'] = df[col].shift(lag)

    create_features(df, 'consumption')
    if has_production:
        create_features(df, 'production')

    df.dropna(inplace=True)

    # Separate feature scaling from target scaling
    features_to_scale = [col for col in feature_columns if col not in ['consumption', 'production']]
    scaler = MinMaxScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    scaler_consumption = MinMaxScaler()
    df['consumption'] = scaler_consumption.fit_transform(df[['consumption']])

    scaler_production = None
    if has_production:
        scaler_production = MinMaxScaler()
        df['production'] = scaler_production.fit_transform(df[['production']])

    # Sequence generation
    look_back = 60
    step = 5
    X_consumption, y_consumption = create_sequences_np(df, 'consumption', feature_columns, look_back, step)

    split_idx_c = int(len(X_consumption) * 0.8)
    X_train_c, y_train_c = X_consumption[:split_idx_c], y_consumption[:split_idx_c]
    X_val_c, y_val_c = X_consumption[split_idx_c:], y_consumption[split_idx_c:]

    # Train model
    model_consumption = create_bilstm_model((look_back, X_train_c.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    with mlflow.start_run():
        mlflow.tensorflow.autolog()
        model_consumption.fit(X_train_c, y_train_c, epochs=1, batch_size=16,
                              validation_data=(X_val_c, y_val_c), callbacks=[early_stopping])

    # Save model & scalers
    model_dir = "/tmp/model"
    os.makedirs(model_dir, exist_ok=True)
    consumption_model_path = os.path.join(model_dir, f"{model_name}_consumption.h5")
    model_consumption.save(consumption_model_path)
    joblib.dump(scaler, os.path.join(model_dir, f"{model_name}_features_scaler.pkl"))
    joblib.dump(scaler_consumption, os.path.join(model_dir, f"{model_name}_consumption_scaler.pkl"))

    user_bucket = await create_user_bucket(user_id, country)
    minio_client.fput_object(user_bucket, f"{model_name}_consumption.h5", consumption_model_path)
    minio_client.fput_object(user_bucket, f"{model_name}_features_scaler.pkl", os.path.join(model_dir, f"{model_name}_features_scaler.pkl"))
    minio_client.fput_object(user_bucket, f"{model_name}_consumption_scaler.pkl", os.path.join(model_dir, f"{model_name}_consumption_scaler.pkl"))

    # Evaluate
    mse_c, mae_c, r2_c = evaluate_model(model_consumption, X_val_c, y_val_c)

    production_metrics = None
    if has_production:
        X_production, y_production = create_sequences_np(df, 'production', feature_columns, look_back, step)
        split_idx_p = int(len(X_production) * 0.8)
        X_train_p, y_train_p = X_production[:split_idx_p], y_production[:split_idx_p]
        X_val_p, y_val_p = X_production[split_idx_p:], y_production[split_idx_p:]

        model_production = create_bilstm_model((look_back, X_train_p.shape[2]))
        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            model_production.fit(X_train_p, y_train_p, epochs=1, batch_size=16,
                                 validation_data=(X_val_p, y_val_p), callbacks=[early_stopping])

        production_model_path = os.path.join(model_dir, f"{model_name}_production.h5")
        model_production.save(production_model_path)
        minio_client.fput_object(user_bucket, f"{model_name}_production.h5", production_model_path)

        if scaler_production:
            joblib.dump(scaler_production, os.path.join(model_dir, f"{model_name}_production_scaler.pkl"))
            minio_client.fput_object(user_bucket, f"{model_name}_production_scaler.pkl", os.path.join(model_dir, f"{model_name}_production_scaler.pkl"))

        mse_p, mae_p, r2_p = evaluate_model(model_production, X_val_p, y_val_p)
        production_metrics = {"mse": mse_p, "mae": mae_p, "r2": r2_p}

    return {
        "message": "Disaggregated forecasting models trained successfully",
        "consumption_model": {
            "mse": float(mse_c),
            "mae": float(mae_c),
            "r2": float(r2_c)
        },
        "production_model": production_metrics if production_metrics else None
    }

def create_sequences_np(data, target_column, feature_columns, look_back=60, step=5):
    data = data[feature_columns].values
    target_index = feature_columns.index(target_column)
    targets = data[:, target_index]

    sequences = np.lib.stride_tricks.sliding_window_view(data, (look_back, data.shape[1]))[:-1:step]
    targets = targets[look_back::step]

    sequences = sequences.reshape(sequences.shape[0], look_back, data.shape[1])
    return sequences.astype(np.float32), targets.astype(np.float32)

def create_bilstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(180, activation="tanh", return_sequences=True), input_shape=input_shape),
        Bidirectional(LSTM(60, activation="tanh", return_sequences=True)),
        Bidirectional(LSTM(10, activation="tanh", return_sequences=False)),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

# -----------------------------
# Forecast - Routes
# ----------------------------- 

# # Forecast Next Days
# Updated endpoint using cleaned data
@app.get("/csv-data/{user_id}/forecast-next-days", tags=['Aggegated Flexibility Forecasting'], response_model=dict)
async def forecast_next_days(user_id: str, model_name: str, country: str, days: int = 1, db: Session = Depends(get_db)):
    """
    Forecast hourly energy consumption for the next `days` using a trained model.

    **Description:**
    - This endpoint predicts hourly consumption and production for the next `days` days.
    - Forecasts are generated using a trained deep learning model.
    - The trained model is loaded from MinIO and applied to **cleaned** energy data.

    **Parameters:**
    - **user_id** (str): The unique identifier of the user.
    - **model_name** (str): The trained model's name.
    - **country** (str): The country associated with the dataset.
    - **method** (str): The aggregation method (in this case: `Aggregated`).
    - **days** (int, optional): Number of days to forecast (default: `1`). Must be between `1` and `7`.
    - **db** (Session): Database session dependency.

    **Returns:**
    - **JSON Object**:
      ```json
      {
          "forecasted_hourly_stats_next_days": [...]
      }
      ```

    **Raises:**
    - **400 Bad Request**: If `days` is outside the valid range (`1-7`).
    - **500 Internal Server Error**: If an unexpected error occurs.
    """
    try:
        if not (1 <= days <= 7):
            raise HTTPException(status_code=400, detail="Number of days must be between 1 and 7")

        method = "Aggregated"  # Hardcoded

        # Fetch and clean the CSV data for the given user
        df = get_cleaned_data_for_user(user_id, db)

        # Ensure consistent lowercase column names for model fitting and prediction
        df.columns = df.columns.str.lower()

        # After cleaning, proceed with the forecast logic
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['hour'] = df['timestamp'].dt.hour
        df['month'] = df['timestamp'].dt.month

        last_date = df['date'].max()
        next_days = pd.date_range(start=last_date + timedelta(days=1), periods=days * 24, freq='H')

        # Prepare the features for the next days
        next_day_features = pd.DataFrame({
            'month': next_days.month,
            'hour': next_days.hour,
            'day_of_week': next_days.dayofweek,
            'consumption': [0.0] * (days * 24),  # Placeholder for consumption
            'production': [0.0] * (days * 24)   # Placeholder for production
        })

        # Ensure the scaler fits the original cleaned data
        original_features = df[['month', 'hour', 'day_of_week', 'consumption', 'production']]
        scaler = MinMaxScaler()
        scaler.fit(original_features)

        # Normalize the next day features using the fitted scaler
        next_day_features_normalized = scaler.transform(next_day_features)

        # Reshape input data to match the model's expected input shape for multiple days
        X_next_days = next_day_features_normalized.reshape((days, 24, next_day_features_normalized.shape[1]))

        # Load the trained model
        user_bucket = f"{user_id}-{country}".lower().replace("_", "-")
        model_file_name = f"{model_name}.h5"
        model_path = f"/tmp/{model_file_name}"

        try:
            minio_client.fget_object(user_bucket, model_file_name, model_path)
        except S3Error as err:
            logging.error(f"Failed to get model {model_file_name} from bucket {user_bucket}: {err}")
            raise HTTPException(status_code=500, detail=f"MinIO error: {err}")

        model = load_model(model_path)

        # Predict consumption (and production if applicable) for the next days
        predictions_next_days = model.predict(X_next_days)

        # Check predictions shape and split into consumption and production if needed
        if predictions_next_days.shape[1] == 24:
            predictions_consumption = predictions_next_days.flatten()
            predictions_production = None
        elif predictions_next_days.shape[1] == 48:
            predictions_consumption = predictions_next_days[:, :24].flatten()
            predictions_production = predictions_next_days[:, 24:].flatten()
        else:
            raise ValueError(f"Model output shape is incorrect: expected 24 or 48 but got {predictions_next_days.shape[1]}")

        # Inverse transform the predictions to their original scale
        scaler_consumption = MinMaxScaler()
        scaler_consumption.fit(df[['consumption']])
        predictions_next_days_original_scale_consumption = scaler_consumption.inverse_transform(predictions_consumption.reshape(-1, 1)).flatten()

        if predictions_production is not None:
            scaler_production = MinMaxScaler()
            scaler_production.fit(df[['production']])
            predictions_next_days_original_scale_production = scaler_production.inverse_transform(predictions_production.reshape(-1, 1)).flatten()
        else:
            predictions_next_days_original_scale_production = None

        if predictions_next_days_original_scale_production is not None:
            for hour in range(days * 24):
                if hour % 24 < 6 or hour % 24 >= 20:
                    predictions_next_days_original_scale_production[hour] = 0.0

        # Calculate baseline, flexibility up, and flexibility down for each hour based on historical data
        last_30_days = pd.date_range(end=last_date, periods=30).date
        forecasted_hourly_stats = []

        for day in range(days):
            for hour in range(24):
                idx = day * 24 + hour
                # Filter the historical data for the past 30 days, matching the day of the week and the hour
                past_30_days_data = df[(df['day_of_week'] == next_days[idx].dayofweek) & (df['date'].isin(last_30_days)) & (df['hour'] == next_days[idx].hour)]

                # Calculate statistics
                baseline_consumption = round(past_30_days_data['consumption'].mean(), 2)
                flexibility_up_consumption = round(past_30_days_data['consumption'].max(), 2)
                flexibility_down_consumption = round(past_30_days_data['consumption'].min(), 2)

                if predictions_next_days_original_scale_production is not None:
                    baseline_production = round(past_30_days_data['production'].mean(), 2)
                    flexibility_up_production = round(past_30_days_data['production'].max(), 2)
                    flexibility_down_production = round(past_30_days_data['production'].min(), 2)
                else:
                    baseline_production = None
                    flexibility_up_production = None
                    flexibility_down_production = None

                # Calculate EFI values
                if baseline_consumption != 0 and not pd.isna(baseline_consumption):
                    efi_up_consumption = round((flexibility_up_consumption - baseline_consumption) * 100 / baseline_consumption, 2)
                    efi_down_consumption = round((baseline_consumption - flexibility_down_consumption) * 100 / baseline_consumption, 2)
                else:
                    efi_up_consumption = None
                    efi_down_consumption = None

                if baseline_production is not None and baseline_production != 0 and not pd.isna(baseline_production):
                    efi_up_production = round((flexibility_up_production - baseline_production) * 100 / baseline_production, 2)
                    efi_down_production = round((baseline_production - flexibility_down_production) * 100 / baseline_production, 2)
                else:
                    efi_up_production = None
                    efi_down_production = None

                # Append the statistics to the hourly stats list
                forecasted_hourly_stats.append({
                    'day': next_days[idx].date(),
                    'hour': next_days[idx].hour,
                    'real_value_consumption': round(float(predictions_next_days_original_scale_consumption[idx]), 2),
                    'baseline_consumption': baseline_consumption,
                    'flexibility_up_consumption': flexibility_up_consumption,
                    'flexibility_down_consumption': flexibility_down_consumption,
                    'efi_up_consumption': efi_up_consumption,
                    'efi_down_consumption': efi_down_consumption,
                    'real_value_production': round(float(predictions_next_days_original_scale_production[idx]), 2) if predictions_next_days_original_scale_production is not None else None,
                    'baseline_production': baseline_production,
                    'flexibility_up_production': flexibility_up_production,
                    'flexibility_down_production': flexibility_down_production,
                    'efi_up_production': efi_up_production,
                    'efi_down_production': efi_down_production
                })

        return {"forecasted_hourly_stats_next_days": forecasted_hourly_stats}

    except Exception as e:
        logging.error(f"Error forecasting next days: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Updated endpoint using cleaned data
@app.get("/csv-data/{user_id}/forecast-and-identify-hours", tags=['Aggegated Flexibility Forecasting'], response_model=dict)
async def forecast_and_identify_hours(user_id: str, model_name: str, country: str, days: int, db: Session = Depends(get_db)):
    """
    Forecast energy consumption and production for the next `days` and classify hours into peak/off-peak categories.

    **Description:**
    - This endpoint forecasts energy consumption and production using a trained model.
    - It identifies peak and off-peak hours for better **energy optimization**.
    - The trained model is retrieved from **MinIO** and applied to cleaned energy data.

    **Parameters:**
    - **user_id** (str): The unique identifier of the user.
    - **model_name** (str): The trained model's name.
    - **country** (str): The country associated with the dataset.
    - **method** (str): The aggregation method (in this case: `Aggregated`).
    - **days** (int): Number of days to forecast (`1-7`).
    - **db** (Session): Database session dependency.

    **Returns:**
    - **JSON Object**:
      ```json
      {
          "forecasted_hourly_stats_all_days": [
              {
                  "date": "YYYY-MM-DD",
                  "peak_peak_hours": [8, 9, 10],
                  "off_peak_peak_hours": [11, 12],
                  "peak_off_peak_hours": [17, 18, 19],
                  "off_off_hours": [2, 3, 4, 5],
                  "predicted_classes": [...],
                  "period_descriptions": {...}
              }
          ]
      }
      ```

    **Raises:**
    - **400 Bad Request**: If `days` is outside the valid range (`1-7`).
    - **500 Internal Server Error**: If an unexpected error occurs.
    """
    if not (1 <= days <= 7):
        raise HTTPException(status_code=400, detail="Days must be between 1 and 7")
    
    try:

        method = "Aggregated"  # Hardcoded

        # Fetch and clean the CSV data for the given user
        df = get_cleaned_data_for_user(user_id, db)

        # Ensure consistent lowercase column names for model fitting and prediction
        df.columns = df.columns.str.lower()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['hour'] = df['timestamp'].dt.hour
        df['month'] = df['timestamp'].dt.month

        last_date = df['date'].max()
        forecasted_hourly_stats_all_days = []

        for day in range(days):
            next_day = last_date + timedelta(days=day + 1)

            # Prepare the features for the next day
            next_day_hours = pd.date_range(start=next_day, periods=24, freq='H')
            next_day_features = pd.DataFrame({
                'month': next_day_hours.month,
                'hour': next_day_hours.hour,
                'day_of_week': next_day_hours.dayofweek,
                'consumption': [0.0] * 24,  # Placeholder for consumption
                'production': [0.0] * 24  # Placeholder for production
            })

            # Ensure the scaler fits the original data
            original_features = df[['month', 'hour', 'day_of_week', 'consumption', 'production']]
            scaler = MinMaxScaler()
            scaler.fit(original_features)

            # Normalize the next day features using the fitted scaler
            next_day_features_normalized = scaler.transform(next_day_features)

            # Reshape input data to match the model's expected input shape
            X_next_day = next_day_features_normalized.reshape((1, 24, next_day_features_normalized.shape[1]))

            # Load the trained model
            user_bucket = f"{user_id}-{country}".lower().replace("_", "-")
            model_file_name = f"{model_name}.h5"
            model_path = f"/tmp/{model_file_name}"

            try:
                minio_client.fget_object(user_bucket, model_file_name, model_path)
            except S3Error as err:
                logging.error(f"Failed to get model {model_file_name} from bucket {user_bucket}: {err}")
                raise HTTPException(status_code=500, detail=f"MinIO error: {err}")

            model = load_model(model_path)

            # Predict consumption (and production if applicable) for the next day
            predictions_next_day = model.predict(X_next_day)

            # Check predictions shape and split into consumption and production if needed
            if predictions_next_day.shape[1] == 24:
                predictions_consumption = predictions_next_day.flatten()
                predictions_production = None
            elif predictions_next_day.shape[1] == 48:
                predictions_consumption = predictions_next_day[0, :24]
                predictions_production = predictions_next_day[0, 24:]
            else:
                raise ValueError(f"Model output shape is incorrect: expected 24 or 48 but got {predictions_next_day.shape[1]}")

            # Inverse transform the predictions to their original scale
            scaler_consumption = MinMaxScaler()
            scaler_consumption.fit(df[['consumption']])
            predictions_next_day_original_scale_consumption = scaler_consumption.inverse_transform(predictions_consumption.reshape(-1, 1)).flatten()

            if predictions_production is not None:
                scaler_production = MinMaxScaler()
                scaler_production.fit(df[['production']])
                predictions_next_day_original_scale_production = scaler_production.inverse_transform(predictions_production.reshape(-1, 1)).flatten()
            else:
                predictions_next_day_original_scale_production = None

            # Set production to zero outside of 6:30 to 22:30
            if predictions_next_day_original_scale_production is not None:
                for hour in range(24):
                    if hour < 6 or hour >= 20:
                        predictions_next_day_original_scale_production[hour] = 0.0

            # Calculate peak and off-peak hours based on consumption and production for each hour
            mean_consumption = np.mean(predictions_next_day_original_scale_consumption)
            mean_production = np.mean(predictions_next_day_original_scale_production) if predictions_next_day_original_scale_production is not None else None

            threshold_consumption = 1.2 * mean_consumption
            threshold_production = 1.2 * mean_production if mean_production is not None else None

            peak_hours_consumption = [hour for hour in range(24) if predictions_next_day_original_scale_consumption[hour] > threshold_consumption]
            off_peak_hours_consumption = list(set(range(24)) - set(peak_hours_consumption))

            if predictions_next_day_original_scale_production is not None:
                peak_hours_production = [hour for hour in range(24) if predictions_next_day_original_scale_production[hour] > threshold_production]
                off_peak_hours_production = list(set(range(24)) - set(peak_hours_production))
            else:
                peak_hours_production = []
                off_peak_hours_production = []

            # Calculate intersection sets for analysis
            peak_peak_hours = set(peak_hours_consumption).intersection(peak_hours_production)
            off_peak_peak_hours = set(off_peak_hours_consumption).intersection(peak_hours_production)
            peak_off_peak_hours = set(peak_hours_consumption).intersection(off_peak_hours_production)
            off_off_hours = set(off_peak_hours_consumption).intersection(off_peak_hours_production)

            # Add descriptions and suggestions for each period
            descriptions = {
                'Peak Consumption & Peak Production': {
                    'description': 'High consumption and high production. Both are at peak levels.',
                    'suggestions': [
                        'Shift non-essential tasks to off-peak hours.'
                    ]
                },
                'Off-Peak Consumption & Peak Production': {
                    'description': 'Low consumption and high production. Excess production.',
                    'suggestions': [
                        'Increase energy usage during this time (e.g., charging EVs or appliances).'
                    ]
                },
                'Peak Consumption & Off-Peak Production': {
                    'description': 'High consumption and low production. You’re consuming more energy than you’re producing.',
                    'suggestions': [
                        'Reduce consumption or shift tasks to off-peak hours to save costs.'
                    ]
                },
                'Off-Peak Consumption & Off-Peak Production': {
                    'description': 'Low consumption and low production. Both energy demand and production are minimal.',
                    'suggestions': [
                        'Maintain low energy consumption.'
                    ]
                }
            }

            # Classify hours based on predictions and return suggestions
            predicted_classes = classify_values(predictions_next_day_original_scale_consumption, predictions_next_day_original_scale_production, threshold_consumption, threshold_production)

            # Add results for this day
            forecasted_hourly_stats_all_days.append({
                "date": str(next_day),
                "peak_peak_hours": sorted(list(peak_peak_hours)),
                "off_peak_peak_hours": sorted(list(off_peak_peak_hours)),
                "peak_off_peak_hours": sorted(list(peak_off_peak_hours)),
                "off_off_hours": sorted(list(off_off_hours)),
                "predicted_classes": predicted_classes,
                "period_descriptions": descriptions
            })

        return {"forecasted_hourly_stats_all_days": forecasted_hourly_stats_all_days}

    except Exception as e:
        logging.error(f"Error forecasting and identifying hours: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# Helper function to classify hours
def classify_values(consumption, production, threshold_consumption, threshold_production):
    classes = []
    for hour in range(24):
        cons = consumption[hour] if not isinstance(consumption[hour], np.ndarray) else consumption[hour].item()
        prod = production[hour] if production is not None and not isinstance(production[hour], np.ndarray) else production[hour].item() if production is not None else None

        if cons > threshold_consumption and (prod is not None and prod > threshold_production):
            classes.append('A: Peak-Peak')
        elif cons <= threshold_consumption and (prod is not None and prod > threshold_production):
            classes.append('B: Off-Peak-Peak')
        elif cons > threshold_consumption and (prod is not None and prod <= threshold_production):
            classes.append('C: Peak-Off-Peak')
        else:
            classes.append('D: Off-Peak-Off-Peak')
    return classes

# Forecast Disaggregated Consumption and Production Next Days
@app.get("/csv-data/{user_id}/forecast-disaggregated", tags=['Disaggegated Flexibility Forecasting'], response_model=dict)
async def forecast_disaggregated_next_days(
    user_id: str, country: str, model_name: str, db: Session = Depends(get_db)
):
    """
    Forecast energy **consumption** and **production** for disaggregation for the next `3 hours` using a one-minute resolution.

    **Description:**
    - Uses a trained **BiLSTM model** to forecast energy at a **minute-level resolution**.
    - The trained models are retrieved from **MinIO** and applied to cleaned energy data.
    - Forecasted values are **ensured to be non-negative**.

    **Parameters:**
    - **user_id** (str): The unique identifier of the user.
    - **model_name** (str): The trained model's name.
    - **country** (str): The country associated with the dataset.
    - **db** (Session): Database session dependency.

    **Returns:**
    - **JSON Object**:
      ```json
      {
          "message": "Forecast generated successfully",
          "forecast_timestamps": ["YYYY-MM-DD HH:MM:SS", ...],
          "forecast_consumption": [...],
          "forecast_production": [...] (optional)
      }
      ```

    **Raises:**
    - **400 Bad Request**: If no valid data is available after cleaning.
    - **500 Internal Server Error**: If an unexpected error occurs.
    """
    # Fetch latest cleaned data
    df = await get_cleaned_disaggregated_csv_data(user_id, db)
    if not df:
        raise HTTPException(status_code=400, detail="No valid data available after cleaning")

    df = pd.DataFrame(df)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Feature engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    def create_features(df, col):
        df[f'rolling_mean_10min_{col}'] = df[col].rolling(window=10, min_periods=1).mean()
        df[f'rolling_std_10min_{col}'] = df[col].rolling(window=10, min_periods=1).std()
        df[f'rolling_max_10min_{col}'] = df[col].rolling(window=10, min_periods=1).max()
        df[f'rolling_min_10min_{col}'] = df[col].rolling(window=10, min_periods=1).min()
        df[f'rolling_mean_60min_{col}'] = df[col].rolling(window=60, min_periods=1).mean()
        df[f'rolling_std_60min_{col}'] = df[col].rolling(window=60, min_periods=1).std()
        for lag in [1, 5, 10, 30]:
            df[f'lag_{lag}min_{col}'] = df[col].shift(lag)

    create_features(df, 'consumption')
    has_production = 'production' in df.columns and not df['production'].isna().all()
    if has_production:
        create_features(df, 'production')

    df = df.dropna()
    if df.empty:
        raise HTTPException(status_code=500, detail="No data left after feature generation")

    feature_columns = [
        'hour', 'day_of_week', 'month', 'day', 'is_weekend', 'consumption',
        'rolling_mean_10min_consumption', 'rolling_std_10min_consumption', 'rolling_max_10min_consumption',
        'rolling_min_10min_consumption', 'rolling_mean_60min_consumption', 'rolling_std_60min_consumption',
        'lag_1min_consumption', 'lag_5min_consumption', 'lag_10min_consumption', 'lag_30min_consumption'
    ]

    if has_production:
        feature_columns += [
            'production',
            'rolling_mean_10min_production', 'rolling_std_10min_production',
            'rolling_max_10min_production', 'rolling_min_10min_production',
            'rolling_mean_60min_production', 'rolling_std_60min_production',
            'lag_1min_production', 'lag_5min_production', 'lag_10min_production', 'lag_30min_production'
        ]

    # Load scalers and apply only to features
    user_bucket = await create_user_bucket(user_id, country)
    model_dir = "/tmp/model"

    features_scaler_path = os.path.join(model_dir, f"{model_name}_features_scaler.pkl")
    minio_client.fget_object(user_bucket, f"{model_name}_features_scaler.pkl", features_scaler_path)
    scaler_features = joblib.load(features_scaler_path)
    # These were the actual features scaled during training — exclude target columns
    features_to_scale = [col for col in feature_columns if col not in ['consumption', 'production']]
    df[features_to_scale] = scaler_features.transform(df[features_to_scale])

    scaler_path = os.path.join(model_dir, f"{model_name}_consumption_scaler.pkl")
    minio_client.fget_object(user_bucket, f"{model_name}_consumption_scaler.pkl", scaler_path)
    scaler_consumption = joblib.load(scaler_path)

    if has_production:
        prod_scaler_path = os.path.join(model_dir, f"{model_name}_production_scaler.pkl")
        minio_client.fget_object(user_bucket, f"{model_name}_production_scaler.pkl", prod_scaler_path)
        scaler_production = joblib.load(prod_scaler_path)

    # Load models
    consumption_model_path = os.path.join(model_dir, f"{model_name}_consumption.h5")
    minio_client.fget_object(user_bucket, f"{model_name}_consumption.h5", consumption_model_path)
    model_consumption = load_model(consumption_model_path)

    if has_production:
        production_model_path = os.path.join(model_dir, f"{model_name}_production.h5")
        minio_client.fget_object(user_bucket, f"{model_name}_production.h5", production_model_path)
        model_production = load_model(production_model_path)

    # Forecast loop
    forecast_steps = 180
    #forecast_steps = 60*24
    last_data = df[feature_columns].values[-60:]
    predictions_consumption = []
    predictions_production = [] if has_production else None

    for _ in range(forecast_steps):
        input_data = np.expand_dims(last_data, axis=0)
        pred_c = np.maximum(model_consumption.predict(input_data)[0, 0], 0)
        predictions_consumption.append(pred_c)

        new_input = np.roll(last_data, -1, axis=0)
        new_input[-1, feature_columns.index('consumption')] = pred_c

        if has_production:
            pred_p = np.maximum(model_production.predict(input_data)[0, 0], 0)
            predictions_production.append(pred_p)
            new_input[-1, feature_columns.index('production')] = pred_p

        last_data = new_input

    # Inverse transform predictions
    predictions_consumption = np.maximum(
        scaler_consumption.inverse_transform(np.array(predictions_consumption).reshape(-1, 1)).flatten(), 0
    )

    if has_production:
        predictions_production = np.maximum(
            scaler_production.inverse_transform(np.array(predictions_production).reshape(-1, 1)).flatten(), 0
        )

    # Timestamps
    last_timestamp = df['timestamp'].max().floor("T")
    timestamps = pd.date_range(start=last_timestamp, periods=forecast_steps + 1, freq='T')[1:]

    return {
        "message": "Forecast generated successfully",
        "forecast_timestamps": timestamps.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "forecast_consumption": predictions_consumption.tolist(),
        **({"forecast_production": predictions_production.tolist()} if has_production else {})
    }

# -----------------------------
# Error Analysis - Routes
# ----------------------------- 
@app.get("/csv-data/{user_id}/errors-in-time-periods", tags=['Aggegated Flexibility Forecasting'], response_model=dict)
async def errors_in_time_periods(user_id: str, model_name: str, country: str, db: Session = Depends(get_db)):
    """
    **Analyze forecasting errors for a user's energy data.**

    **Description:**
    - Compares the **actual vs. predicted** energy consumption and production.
    - Identifies **misclassified peak/off-peak periods**.
    - Computes a **confusion matrix** and **error metrics** for classification accuracy.
    - Generates a **heatmap visualization** of the confusion matrix.

    **Parameters:**
    - **user_id** (str): The unique identifier of the user.
    - **model_name** (str): The trained model's name.
    - **country** (str): The country associated with the dataset.
    - **method** (str): The aggregation method used (in this case: `Aggregated`).
    - **db** (Session): Database session dependency.

    **Returns:**
    - **JSON Object**:
      ```json
      {
          "Confusion Matrix": [[...], [...]],
          "Metrics": [
              {
                  "Class": "A: Peak-Peak",
                  "Accuracy": 0.85,
                  "Precision": 0.90,
                  "Recall": 0.80,
                  "F1 Score": 0.85,
                  "Weighted Average Index": 0.86
              },
              ...
          ]
      }
      ```

    **Raises:**
    - **400 Bad Request**: If insufficient data is available for analysis.
    - **500 Internal Server Error**: If an unexpected error occurs.
    """
    try:

        method = "Aggregated"  # Hardcoded
        # Fetch and clean the data using the get_cleaned_data_for_user function
        df = get_cleaned_data_for_user(user_id, db)

        if df.empty:
            raise HTTPException(status_code=400, detail="No valid data available after cleaning")

        df['Timestamp'] = pd.to_datetime(df['timestamp'])
        df['Date'] = df['Timestamp'].dt.date
        df['Hour'] = df['Timestamp'].dt.hour
        df['Month'] = df['Timestamp'].dt.month
        df['Day_of_Week'] = df['Timestamp'].dt.dayofweek  # Add Day_of_Week

        # Define thresholds for peak/non-peak classification
        mean_consumption_actual = df['consumption'].mean()
        mean_production_actual = df['production'].mean()

        threshold_actual_consumption = 1.2 * mean_consumption_actual
        threshold_actual_production = 1.2 * mean_production_actual

        # Load the model
        user_bucket = f"{user_id}-{country}".lower().replace("_", "-")
        model_file_name = f"{model_name}.h5"
        model_path = f"/tmp/{model_file_name}"

        try:
            minio_client.fget_object(user_bucket, model_file_name, model_path)
        except S3Error as err:
            logging.error(f"Failed to get model {model_file_name} from bucket {user_bucket}: {err}")
            raise HTTPException(status_code=500, detail=f"MinIO error: {err}")

        model = load_model(model_path)

        # Prepare features for the penultimate day and predict
        unique_dates = df['Date'].unique()
        if len(unique_dates) < 2:
            raise HTTPException(status_code=400, detail="Not enough data to analyze the penultimate day.")

        penultimate_date = unique_dates[-2]
        penultimate_day_data = df[df['Date'] == penultimate_date]

        # Ensure we have 24 hours of data; if not, fill missing hours with NaN
        all_hours = pd.DataFrame({'Hour': range(24)})
        penultimate_day_features = pd.merge(all_hours, penultimate_day_data[['Hour', 'consumption', 'production', 'Month', 'Day_of_Week']], on='Hour', how='left')

        if penultimate_day_features.shape[0] != 24:
            raise HTTPException(status_code=400, detail=f"Expected 24 rows for 24 hours, but got {penultimate_day_features.shape[0]}. Data may be incomplete.")

        # Normalize and predict
        scaler = MinMaxScaler()
        penultimate_day_features_normalized = scaler.fit_transform(penultimate_day_features.fillna(0))  # Fill NaNs with 0 for normalization

        # Reshape the data into 3D for the model input
        X_penultimate_day = penultimate_day_features_normalized.reshape((1, 24, penultimate_day_features_normalized.shape[1]))

        # Log the shape for debugging purposes
        logging.info(f"Shape of X_penultimate_day: {X_penultimate_day.shape}")

        # Predict using the model
        predictions_penultimate_day = model.predict(X_penultimate_day)

        # Log the predictions shape
        logging.info(f"Shape of predictions_penultimate_day: {predictions_penultimate_day.shape}")

        # Since the output is a flattened array, we need to reshape it accordingly
        predictions_penultimate_day = predictions_penultimate_day.reshape((24, 2))

        predictions_consumption = predictions_penultimate_day[:, 0]  # Consumption predictions
        predictions_production = predictions_penultimate_day[:, 1]  # Production predictions

        # Classify hours based on predicted and actual values
        predicted_classes = classify_values(predictions_consumption, predictions_production, threshold_actual_consumption, threshold_actual_production)
        actual_classes = classify_values(penultimate_day_features['consumption'].fillna(0).values, penultimate_day_features['production'].fillna(0).values, threshold_actual_consumption, threshold_actual_production)

        # Define all possible classes (A, B, C, D) manually
        all_possible_classes = ["A: Peak-Peak", "B: Off-Peak-Peak", "C: Peak-Off-Peak", "D: Off-Peak-Off-Peak"]

        # Calculate confusion matrix, ensuring all classes are represented
        conf_matrix = confusion_matrix(actual_classes, predicted_classes, labels=all_possible_classes)

        # Calculate metrics for each class
        metrics = []
        for i, label in enumerate(all_possible_classes):
            true_positive = conf_matrix[i, i]
            false_positive = conf_matrix[:, i].sum() - true_positive
            false_negative = conf_matrix[i, :].sum() - true_positive
            true_negative = conf_matrix.sum() - (true_positive + false_positive + false_negative)

            # Adjusting the calculation to handle the case where the class has no true positives
            if (true_positive + false_positive + false_negative) == 0:
                accuracy = None
                precision = None
                recall = None
                f1_score = None
            else:
                accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
                precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else None
                recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else None
                f1_score = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and (precision + recall) != 0 else None

            metrics.append({
                "Class": label,
                "Accuracy": accuracy if accuracy is not None else 0,
                "Precision": precision if precision is not None else 0,
                "Recall": recall if recall is not None else 0,
                "F1 Score": f1_score if f1_score is not None else 0
            })

        # Weighted Average Index Calculation
        def weighted_average_index(accuracy, precision, recall, f1_score):
            weights = {'accuracy': 0.30, 'precision': 0.20, 'recall': 0.30, 'f1_score': 0.20}
            return (
                weights['accuracy'] * (accuracy if accuracy is not None else 0) +
                weights['precision'] * (precision if precision is not None else 0) +
                weights['recall'] * (recall if recall is not None else 0) +
                weights['f1_score'] * (f1_score if f1_score is not None else 0)
            )

        # Add weighted average index to each metric
        for metric in metrics:
            metric['Weighted Average Index'] = weighted_average_index(
                metric['Accuracy'],
                metric['Precision'],
                metric['Recall'],
                metric['F1 Score']
            )

        # Display the confusion matrix as a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=all_possible_classes, yticklabels=all_possible_classes)
        plt.xlabel('Predicted Classes')
        plt.ylabel('Actual Classes')
        plt.title('Confusion Matrix')
        plt.savefig('/tmp/confusion_matrix.png')

        return {
            "Confusion Matrix": conf_matrix.tolist(),
            "Metrics": metrics
        }

    except Exception as e:
        logging.error(f"Error in error analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------
# EFIs in periods calculation - Routes
# ------------------------------------ 
@app.get("/csv-data/{user_id}/efi-in-time-periods", tags=['Aggegated Flexibility Forecasting'], response_model=dict)
async def errors_and_efi_in_time_periods(user_id: str, model_name: str, country: str, days: int = 1, db: Session = Depends(get_db)):
    """
    **Calculate Energy Flexibility Indicators (EFI) for Specific Time Periods.**
    
    **Description:**
    - Predicts **next-day** energy consumption and production.
    - Identifies **peak/off-peak consumption & production hours**.
    - Computes **EFI up/down values** for each period.
    - Provides **actionable insights & recommendations**.

    **Parameters:**
    - **user_id** (str): Unique user identifier.
    - **model_name** (str): Model used for prediction.
    - **country** (str): User's country.
    - **method** (str): Aggregation method (in this case: 'Aggregated').
    - **days** (int): Number of days to predict (1-7).
    - **db** (Session): Database session.

    **Returns:**
    - **EFI insights for multiple days**:
      ```json
      {
          "EFI Results for all days": [
              {
                  "day": "YYYY-MM-DD",
                  "efi_results": {
                      "EFI_peak_peak_hours": [10.5, 5.2],
                      "EFI_off_peak_peak_hours": [8.3, 3.1],
                      "EFI_peak_off_peak_hours": [6.7, 2.4],
                      "EFI_off_off_hours": [4.5, 1.8]
                  }
              }
          ],
          "EFI Descriptions": {
              "Peak Consumption & Peak Production": {
                  "description": "High consumption and high production.",
                  "efi_explanation": {
                      "EFI Up": "Increase energy usage in this period.",
                      "EFI Down": "Reduce energy usage for cost savings."
                  },
                  "suggestions": ["Shift tasks to off-peak hours."]
              }
          }
      }
      ```

    **Raises:**
    - **400 Bad Request**: If `days` is out of range or data is insufficient.
    - **500 Internal Server Error**: If an unexpected issue occurs.
    """
    try:
        method = "Aggregated"  # Hardcoded
        # Ensure the number of days is within the 1 to 7 range
        if not (1 <= days <= 7):
            raise HTTPException(status_code=400, detail="Days must be between 1 and 7")
        
        # Use the cleaned data from the get_cleaned_data_for_user function
        df = get_cleaned_data_for_user(user_id, db)

        if df.empty:
            raise HTTPException(status_code=400, detail="No valid data available after cleaning")

        df['Timestamp'] = pd.to_datetime(df['timestamp'])
        df['Date'] = df['Timestamp'].dt.date
        df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
        df['Hour'] = df['Timestamp'].dt.hour
        df['Month'] = df['Timestamp'].dt.month

        last_date = df['Date'].max()

        # Placeholder to store EFI results for each day
        efi_results_all_days = []

        for day in range(days):
            next_day = last_date + timedelta(days=day + 1)

            # Prepare the features for the next day
            next_day_hours = pd.date_range(start=next_day, periods=24, freq='H')
            next_day_features = pd.DataFrame({
                'Month': next_day_hours.month,
                'Hour': next_day_hours.hour,
                'Day_of_Week': next_day_hours.dayofweek,
                'consumption': [0.0] * 24,  # Placeholder value for consumption
                'production': [0.0] * 24    # Placeholder for production
            })

            # Normalize the next day features using the cleaned data
            original_features = df[['Month', 'Hour', 'Day_of_Week', 'consumption', 'production']]
            scaler = MinMaxScaler()
            scaler.fit(original_features)
            next_day_features_normalized = scaler.transform(next_day_features)
            X_next_day = next_day_features_normalized.reshape((1, 24, next_day_features_normalized.shape[1]))

            # Load the trained model
            user_bucket = f"{user_id}-{country}".lower().replace("_", "-")
            model_file_name = f"{model_name}.h5"
            model_path = f"/tmp/{model_file_name}"

            try:
                minio_client.fget_object(user_bucket, model_file_name, model_path)
            except S3Error as err:
                logging.error(f"Failed to get model {model_file_name} from bucket {user_bucket}: {err}")
                raise HTTPException(status_code=500, detail=f"MinIO error: {err}")

            model = load_model(model_path)

            # Predict consumption and production for the next day
            predictions_next_day = model.predict(X_next_day)
            if predictions_next_day.shape[1] == 24:
                predictions_consumption = predictions_next_day.flatten()
                predictions_production = None
            elif predictions_next_day.shape[1] == 48:
                predictions_consumption = predictions_next_day[0, :24]
                predictions_production = predictions_next_day[0, 24:]
            else:
                raise ValueError(f"Model output shape is incorrect: expected 24 or 48 but got {predictions_next_day.shape[1]}")

            # Inverse transform predictions to their original scale
            scaler_consumption = MinMaxScaler()
            scaler_consumption.fit(df[['consumption']])
            predictions_next_day_original_scale_consumption = scaler_consumption.inverse_transform(predictions_consumption.reshape(-1, 1)).flatten()

            if predictions_production is not None:
                scaler_production = MinMaxScaler()
                scaler_production.fit(df[['production']])
                predictions_next_day_original_scale_production = scaler_production.inverse_transform(predictions_production.reshape(-1, 1)).flatten()
            else:
                predictions_next_day_original_scale_production = None

            # Set production to zero outside of 6:30 to 22:30
            if predictions_next_day_original_scale_production is not None:
                for hour in range(24):
                    if hour < 6 or (hour == 6 and next_day_hours[hour].minute < 30) or hour >= 18 or (hour == 22 and next_day_hours[hour].minute > 30):
                        predictions_next_day_original_scale_production[hour] = 0.0

            # Fetch forecasted hourly stats from forecast-next-day logic
            last_30_days = pd.date_range(end=last_date, periods=30).date
            forecasted_hourly_stats = []
            for hour in range(24):
                past_30_days_data = df[(df['Day_of_Week'] == next_day_hours[0].dayofweek) & (df['Date'].isin(last_30_days)) & (df['Hour'] == hour)]
                baseline_consumption = round(past_30_days_data['consumption'].mean(), 2)
                flexibility_up_consumption = round(past_30_days_data['consumption'].max(), 2)
                flexibility_down_consumption = round(past_30_days_data['consumption'].min(), 2)

                if predictions_next_day_original_scale_production is not None:
                    baseline_production = round(past_30_days_data['production'].mean(), 2)
                    flexibility_up_production = round(past_30_days_data['production'].max(), 2)
                    flexibility_down_production = round(past_30_days_data['production'].min(), 2)
                else:
                    baseline_production = None
                    flexibility_up_production = None
                    flexibility_down_production = None

                # Calculate EFI
                efi_up_consumption = round((flexibility_up_consumption - baseline_consumption) * 100 / baseline_consumption, 2) if baseline_consumption else None
                efi_down_consumption = round((baseline_consumption - flexibility_down_consumption) * 100 / baseline_consumption, 2) if baseline_consumption else None

                efi_up_production = round((flexibility_up_production - baseline_production) * 100 / baseline_production, 2) if baseline_production else None
                efi_down_production = round((baseline_production - flexibility_down_production) * 100 / baseline_production, 2) if baseline_production else None

                forecasted_hourly_stats.append({
                    'hour': hour,
                    'real_value_consumption': round(float(predictions_next_day_original_scale_consumption[hour]), 2),
                    'baseline_consumption': baseline_consumption,
                    'flexibility_up_consumption': flexibility_up_consumption,
                    'flexibility_down_consumption': flexibility_down_consumption,
                    'efi_up_consumption': efi_up_consumption,
                    'efi_down_consumption': efi_down_consumption,
                    'real_value_production': round(float(predictions_next_day_original_scale_production[hour]), 2) if predictions_next_day_original_scale_production is not None else None,
                    'baseline_production': baseline_production,
                    'flexibility_up_production': flexibility_up_production,
                    'flexibility_down_production': flexibility_down_production,
                    'efi_up_production': efi_up_production,
                    'efi_down_production': efi_down_production
                })

            # Identify peak and off-peak hours
            mean_consumption = np.mean(predictions_next_day_original_scale_consumption)
            threshold_consumption = 1.2 * mean_consumption
            peak_hours_consumption = [hour for hour in range(24) if predictions_next_day_original_scale_consumption[hour] > threshold_consumption]
            off_peak_hours_consumption = list(set(range(24)) - set(peak_hours_consumption))

            mean_production = np.mean(predictions_next_day_original_scale_production) if predictions_next_day_original_scale_production is not None else None
            threshold_production = 1.2 * mean_production if mean_production is not None else None
            peak_hours_production = [hour for hour in range(24) if predictions_next_day_original_scale_production[hour] > threshold_production] if threshold_production else []
            off_peak_hours_production = list(set(range(24)) - set(peak_hours_production))

            # Calculate intersection sets for time periods
            peak_peak_hours = set(peak_hours_consumption).intersection(peak_hours_production)
            off_peak_peak_hours = set(off_peak_hours_consumption).intersection(peak_hours_production)
            peak_off_peak_hours = set(peak_hours_consumption).intersection(off_peak_hours_production)
            off_off_hours = set(off_peak_hours_consumption).intersection(off_peak_hours_production)

            # Calculate EFI for each time period
            def calculate_efi_for_hours(hours, forecasted_stats, key):
                efi_up = [forecasted_stats[hour][f"efi_up_{key}"] for hour in hours if forecasted_stats[hour][f"efi_up_{key}"] is not None]
                efi_down = [forecasted_stats[hour][f"efi_down_{key}"] for hour in hours if forecasted_stats[hour][f"efi_down_{key}"] is not None]

                avg_efi_up = round(sum(efi_up) / len(efi_up), 2) if efi_up else None
                avg_efi_down = round(sum(efi_down) / len(efi_down), 2) if efi_down else None
                return avg_efi_up, avg_efi_down

            efi_results = {
                'EFI_peak_peak_hours': calculate_efi_for_hours(peak_peak_hours, forecasted_hourly_stats, 'consumption'),
                'EFI_off_peak_peak_hours': calculate_efi_for_hours(off_peak_peak_hours, forecasted_hourly_stats, 'consumption'),
                'EFI_peak_off_peak_hours': calculate_efi_for_hours(peak_off_peak_hours, forecasted_hourly_stats, 'consumption'),
                'EFI_off_off_hours': calculate_efi_for_hours(off_off_hours, forecasted_hourly_stats, 'consumption')
            }

            # Add the results for the current day to the list
            efi_results_all_days.append({
                "day": next_day,
                "efi_results": efi_results
            })

        # Descriptions and explanations for each time period
        efi_descriptions = {
            'Peak Consumption & Peak Production': {
                'description': 'Both energy consumption and production are high.',
                'efi_explanation': {
                    'EFI Up': 'You can increase energy usage during this period if necessary, but it may come with higher costs.',
                    'EFI Down': 'You can reduce energy usage during this period to avoid high consumption costs.'
                },
                'suggestions': [
                    'Shift non-essential tasks to off-peak hours.'
                ]
            },
            'Off-Peak Consumption & Peak Production': {
                'description': 'Consumption is low, but production is high.',
                'efi_explanation': {
                    'EFI Up': 'Increase energy usage to take advantage of excess production.',
                    'EFI Down': 'Maintain low consumption and store excess energy if possible.'
                },
                'suggestions': [
                    'Use high-energy appliances from hours (A,C) to utilize available energy.'
                ]
            },
            'Peak Consumption & Off-Peak Production': {
                'description': 'High consumption but low production.',
                'efi_explanation': {
                    'EFI Up': 'You can increase consumption further, but it may increase costs due to low production.',
                    'EFI Down': 'Reducing consumption is beneficial as production is low.'
                },
                'suggestions': [
                    'Warning: Try to shift tasks to a,b periods to save costs.'
                ]
            },
            'Off-Peak Consumption & Off-Peak Production': {
                'description': 'Low consumption and low production.',
                'efi_explanation': {
                    'EFI Up': 'You can increase consumption slightly, but it’s already a low-demand period.',
                    'EFI Down': 'Minimal energy reduction is possible, but it may not have a significant impact.'
                },
                'suggestions': [
                    'Maintain low consumption and avoid unnecessary energy usage.'
                ]
            }
        }

        return {
            "EFI Results for all days": efi_results_all_days,
            "EFI Descriptions": efi_descriptions
        }

    except Exception as e:
        logging.error(f"Error calculating EFI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Day Summary - Routes
# ----------------------------- 
@app.get("/csv-data/{user_id}/day-summary", tags=['Data Analysis'], response_model=dict)
async def day_summary(user_id: str, model_name: str, country: str, method: str, days: int = 1, db: Session = Depends(get_db)):
    """
    **Generate a Daily Energy Summary Based on Forecasted Data.**
    
    **Description:**
    - Fetches **classified periods** from `forecast_and_identify_hours`.
    - Retrieves **EFI insights** from `errors_and_efi_in_time_periods`.
    - Identifies **dominant energy periods** in the day.
    - Suggests **energy shift optimizations** based on period occurrences.

    **Parameters:**
    - **user_id** (str): Unique user identifier.
    - **model_name** (str): Model used for prediction.
    - **country** (str): User's country.
    - **method** (str): Aggregation method (in this case: 'Aggregated').
    - **days** (int): Number of days to analyze (1-7).
    - **db** (Session): Database session.

    **Returns:**
    - **Daily summaries with suggestions**:
      ```json
      {
          "Daily Summaries": [
              {
                  "date": "YYYY-MM-DD",
                  "dominant_period": "Peak Consumption and Peak Production",
                  "suggestions": [
                      "Try to shift high-energy appliances to hours: ['10:00', '11:00'] to save on energy costs."
                  ],
                  "period_occurrences": {
                      "Peak Consumption and Peak Production": 10,
                      "Off-Peak Consumption and Peak Production": 5,
                      "Peak Consumption and Off-Peak Production": 4,
                      "Off-Peak Consumption and Off-Peak Production": 5
                  }
              }
          ]
      }
      ```

    **Raises:**
    - **400 Bad Request**: If `days` is out of range.
    - **500 Internal Server Error**: If an unexpected issue occurs.
    """
    try:
        # Fetch classified hours from forecast_and_identify_hours
        forecast_data = await forecast_and_identify_hours(user_id, model_name, country, days, db)

        # Fetch EFI suggestions and descriptions from errors_and_efi_in_time_periods
        efi_data = await errors_and_efi_in_time_periods(user_id, model_name, country, days, db)

        # Prepare a list to store summaries for each day
        day_summaries = []

        # Define sleep hours to exclude
        sleep_hours = list(range(23, 24)) + list(range(0, 6))

        # Loop through each day's data in forecasted_hourly_stats_all_days
        for day_data in forecast_data["forecasted_hourly_stats_all_days"]:
            # Initialize a dictionary to store period occurrences for the current day
            period_occurrences = {
                "Peak Consumption and Peak Production": 0,
                "Off-Peak Consumption and Peak Production": 0,
                "Peak Consumption and Off-Peak Production": 0,
                "Off-Peak Consumption and Off-Peak Production": 0
            }

            predicted_classes = [
                "Peak Consumption and Peak Production" if cls == "A: Peak-Peak" else
                "Off-Peak Consumption and Peak Production" if cls == "B: Off-Peak-Peak" else
                "Peak Consumption and Off-Peak Production" if cls == "C: Peak-Off-Peak" else
                "Off-Peak Consumption and Off-Peak Production" if cls == "D: Off-Peak-Off-Peak" else
                cls
                for cls in day_data["predicted_classes"]
            ]

            # Count occurrences of each period for the current day
            for period in predicted_classes:
                period_occurrences[period] += 1

            # Determine the dominant period for the day
            dominant_period = max(period_occurrences, key=period_occurrences.get)

            # Find hours for each period
            hours_A = [hour for hour, cls in enumerate(predicted_classes) if cls == "Peak Consumption and Peak Production"]
            hours_B = [hour for hour, cls in enumerate(predicted_classes) if cls == "Off-Peak Consumption and Peak Production"]
            hours_C = [hour for hour, cls in enumerate(predicted_classes) if cls == "Peak Consumption and Off-Peak Production"]
            hours_D = [hour for hour, cls in enumerate(predicted_classes) if cls == "Off-Peak Consumption and Off-Peak Production"]

            # Remove sleep hours from potential target and source hours
            hours_A = [hour for hour in hours_A if hour not in sleep_hours]
            hours_B = [hour for hour in hours_B if hour not in sleep_hours]
            hours_C = [hour for hour in hours_C if hour not in sleep_hours]
            hours_D = [hour for hour in hours_D if hour not in sleep_hours]

            # Combine and sort hours from A and C
            combined_hours_A_C = sorted(hours_A + hours_C)

            # Convert the hour values to "HH:MM" format
            combined_hours_A_C_str = [f"{hour:02d}:00" for hour in combined_hours_A_C]
            hours_C_str = [f"{hour:02d}:00" for hour in hours_C]

            # Generate suggestions based on the dominant period, limiting shifts to neighboring hours
            suggestions = []

            def get_nearby_hours(target_hours, source_hours, initial_window=2, max_window=6):
                """
                Get nearby hours within a specified window.
                Args:
                    target_hours (list): Hours that can be used for shifting.
                    source_hours (list): Hours to be shifted.
                    initial_window (int): The initial allowable shift window in hours.
                    max_window (int): The maximum allowable shift window in hours.
                Returns:
                    list: Filtered target hours that are within the window of each source hour.
                """
                filtered_hours = []
                current_window = initial_window

                # Gradually increase the window size until we find suitable target hours or reach the maximum window size
                while not filtered_hours and current_window <= max_window:
                    for source_hour in source_hours:
                        nearby_hours = [hour for hour in target_hours if abs(hour - source_hour) <= current_window]
                        filtered_hours.extend(nearby_hours)
                    current_window += 1

                return sorted(set(filtered_hours))

            # Generate suggestions for each dominant period
            if dominant_period == "Off-Peak Consumption and Peak Production":
                nearby_hours = get_nearby_hours(hours_B, combined_hours_A_C)
                if nearby_hours:
                    suggestions.append(
                        f"Try to shift high-energy appliances from hours: {combined_hours_A_C_str[:4]} to {[f'{hour:02d}:00' for hour in sorted(set(nearby_hours))[:4]]} to utilize available PV energy."
                    )
                else:
                    suggestions.append("Consider adjusting high-energy appliance usage to Off-Peak Consumption and Peak Production hours generally to optimize energy consumption.")

            elif dominant_period == "Off-Peak Consumption and Off-Peak Production":
                nearby_hours = get_nearby_hours(hours_B, hours_C)
                if nearby_hours:
                    suggestions.append(
                        f"Try to shift high-energy appliances from hours: {hours_C_str[:4]} to {[f'{hour:02d}:00' for hour in sorted(set(nearby_hours))[:4]]} to utilize available PV energy."
                    )
                else:
                    suggestions.append("Consider adjusting high-energy appliance usage to Off-Peak Consumption and Peak Production hours generally to optimize energy consumption.")

            elif dominant_period == "Peak Consumption and Off-Peak Production":
                nearby_hours = get_nearby_hours(hours_A + hours_B, hours_C)
                if nearby_hours:
                    suggestions.append(
                        f"Try to shift high-energy appliances from hours: {hours_C_str[:4]} to {[f'{hour:02d}:00' for hour in sorted(set(nearby_hours))[:4]]} to save on energy costs."
                    )
                else:
                    suggestions.append("Consider adjusting high-energy appliance usage to Off-Peak Consumption and Peak Production hours generally to save on energy costs.")

            elif dominant_period == "Peak Consumption and Peak Production":
                nearby_hours = get_nearby_hours(hours_B, hours_A)
                if nearby_hours:
                    suggestions.append(
                        f"Try to shift high-energy appliances to hours: {[f'{hour:02d}:00' for hour in sorted(set(nearby_hours))[:4]]} to save on energy costs."
                    )
                else:
                    suggestions.append("Consider adjusting high-energy appliance usage to Off-Peak Consumption and Peak Production hours generally to save on energy costs.")

            day_summary = {
                "date": day_data["date"],
                "dominant_period": dominant_period,
                "suggestions": suggestions,
                "period_occurrences": period_occurrences
            }

            # Add the summary for the current day to the list
            day_summaries.append(day_summary)

        # Return the list of day summaries
        return {"Daily Summaries": day_summaries}

    except Exception as e:
        logging.error(f"Error generating day summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Upload Disaggregation Model to MinIO
# -----------------------------
@app.post("/upload-disaggregation-model/", tags=['Disaggregated Flexibility Models'])
async def upload_disaggregation_model(user_id: str, file: UploadFile = File(...)):
    """
    Upload a trained disaggregation model (.h5 or .pkl) to MinIO under a user-specific bucket.

    **Query Parameters:**
    - **user_id**: Unique identifier of the user.
    - **file**: Upload model file (.h5 or .pkl).

    **Returns:**
    - Success message if uploaded correctly.
    """
    try:
        #bucket_name = f"{user_id}-disaggregation-models"
        bucket_name = "disaggregation-models"  # Shared bucket for all users

        # Create bucket if it doesn't exist
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        # Save model temporarily
        model_path = f"/tmp/{file.filename}"
        with open(model_path, "wb") as buffer:
            buffer.write(await file.read())

        # Upload to MinIO
        minio_client.fput_object(bucket_name, file.filename, model_path)
        os.remove(model_path)  # Cleanup

        return {"message": f"Model {file.filename} uploaded successfully to MinIO bucket {bucket_name}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
import pickle  # Ensure pickle is imported for loading .pkl models

# -----------------------------
# Disaggregate Forecasted Time Series
# -----------------------------
def download_model_from_minio(model_name, user_id):
    """
    Downloads a model (.h5 or .pkl) from MinIO and loads it accordingly.
    """
    bucket_name = f"disaggregation-models"
    model_path = f"/tmp/{model_name}"

    try:
        minio_client.fget_object(bucket_name, model_name, model_path)

        # Load model based on extension
        if model_name.endswith(".h5"):
            return load_model(model_path)  # Keras Model
        elif model_name.endswith(".pkl"):
            with open(model_path, "rb") as f:
                return pickle.load(f)  # Scikit-learn Model
        else:
            raise ValueError(f"Unsupported model format: {model_name}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading {model_name}: {str(e)}")

# # Forecast and Disaggregate Next Day
@app.post("/csv-data/{user_id}/forecast-disaggregated-disaggregation", tags=['Disaggegated Flexibility Forecasting'], response_model=dict)
async def forecast_disaggregated_disaggregation(
    user_id: str, country: str, model_name: str, selected_appliances: List[str], db: Session = Depends(get_db)
):
    """
    **Forecast & Disaggregate Energy Consumption & Production**

    **Description:**
    1. **Forecasts** next 3 hours minute-resolution energy consumption & production using BiLSTM models.
    2. **Performs disaggregation** based on user-selected appliances (e.g., TV, Dishwasher, etc.).
    3. **Retrieves models from MinIO** for forecasting and disaggregation.

    **Parameters:**
    - **user_id** (str): The unique user identifier.
    - **country** (str): The country associated with the data.
    - **model_name** (str): The trained forecasting model name.
    - **selected_appliances** (List[str]): List of selected appliances for disaggregation.
    - **db** (Session): Database session dependency.

    **Returns:**
    - **JSON Object**:
      ```json
      {
          "message": "Forecast and disaggregation successful",
          "forecast_timestamps": ["YYYY-MM-DD HH:MM:SS", ...],
          "forecast_consumption": [...],
          "forecast_production": [...],
          "disaggregated_consumption": {
              "TV": [...],
              "Dishwasher": [...],
              "EV": [...],
              "Total Remaining Consumption": [...]
          }
      }
      ```
    """
    import pickle
    import os

    try:
        # Step 1: Forecast
        forecast_result = await forecast_disaggregated_next_days(user_id, country, model_name, db)
        forecasted_timestamps = forecast_result["forecast_timestamps"]
        forecasted_consumption = np.array(forecast_result["forecast_consumption"], dtype=float)

        # Step 2: Disaggregation loop
        disaggregated_results = {}
        total_disaggregated = np.zeros_like(forecasted_consumption, dtype=float)

        for appliance in selected_appliances:
            model_file = f"{appliance}_model.h5"
            model = download_model_from_minio(model_file, user_id)

            if model is None:
                continue  # Skip if model not found

            # Load max_dataset_value from corresponding .pkl
            try:
                pkl_path = f"/tmp/model/{appliance}_model.pkl"
                minio_client.fget_object("disaggregation-models", f"{appliance}_model.pkl", pkl_path)

                with open(pkl_path, "rb") as f:
                    model_info = pickle.load(f)
                    max_value = model_info.get("max", 1.0)
            except Exception as e:
                print(f"Warning: Could not load max for {appliance}: {e}")
                max_value = 1.0

            # Prepare input
            X_test = forecasted_consumption.reshape(-1, 1).astype(float)

            time_steps = model.input_shape[1]
            num_features = model.input_shape[2]

            if X_test.shape[0] < time_steps:
                X_padded = np.zeros((time_steps, num_features), dtype=float)
                X_padded[:X_test.shape[0], :] = X_test
                X_test = np.expand_dims(X_padded, axis=0)
            else:
                X_test = np.array([X_test[i: i + time_steps] for i in range(len(X_test) - time_steps + 1)])

            # Predict and rescale
            appliance_power = model.predict(X_test).flatten().astype(float)
            appliance_power = appliance_power[: len(forecasted_consumption)]
            appliance_power = np.maximum(appliance_power * max_value, 0)  # Re-scale and clip negatives

            def enforce_min_cycle(mask, min_len=3):
                padded = np.pad(mask, (1, 1), constant_values=0)
                diff = np.diff(padded)
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                valid = np.zeros_like(mask)
                for s, e in zip(starts, ends):
                    if (e - s) >= min_len:
                        valid[s:e] = 1
                return valid

            # --- Clustering-based Refinement for Washing Machine ---
            if appliance.lower() == "washing_machine":
                timestamps = pd.to_datetime(forecasted_timestamps[:len(appliance_power)])
                df = pd.DataFrame({'timestamp': timestamps, 'wm_power': appliance_power})
                df['quarter'] = df['timestamp'].dt.floor('15min')

                features = df.groupby('quarter')['wm_power'].agg(
                    total_power='sum',
                    avg_power='mean',
                    peak_power='max',
                    std_power='std',
                    sample_count='count'
                ).fillna(0).reset_index()

                def rule_based_filter(row):
                    return int(
                        #row['total_power'] > 280000 and
                        row['total_power'] > 1800 and
                        # row['peak_power'] > 1900 and
                        row['peak_power'] > 140 and
                        # row['std_power'] > 160
                        row['std_power'] > 10
                    )

                features['wm_active'] = features.apply(rule_based_filter, axis=1)

                def enforce_min_cycle(mask, min_len=3):
                    padded = np.pad(mask, (1, 1), constant_values=0)
                    diff = np.diff(padded)
                    starts = np.where(diff == 1)[0]
                    ends = np.where(diff == -1)[0]
                    valid = np.zeros_like(mask)
                    for s, e in zip(starts, ends):
                        if (e - s) >= min_len:
                            valid[s:e] = 1
                    return valid

                print(f"--- {appliance} per-quarter stats ---")
                print(features[['quarter', 'total_power', 'peak_power', 'std_power']])

                #features['wm_active'] = enforce_min_cycle(features['wm_active'].values)
                features['wm_active'] = enforce_min_cycle(features['wm_active'].values, min_len=3)
                df = df.merge(features[['quarter', 'wm_active']], on='quarter', how='left')
                df['wm_active'] = df['wm_active'].fillna(0).astype(int)
                appliance_power = df['wm_power'] * df['wm_active']
                ####appliance_power = df['wm_power'] 
            # --- End WM Refinement ---
            # --- Clustering-based Refinement for Dish Washer ---
            elif appliance.lower() == "dish_washer":
                timestamps = pd.to_datetime(forecasted_timestamps[:len(appliance_power)])
                df = pd.DataFrame({'timestamp': timestamps, 'dw_power': appliance_power})
                df['quarter'] = df['timestamp'].dt.floor('15min')

                features = df.groupby('quarter')['dw_power'].agg(
                    total_power='sum',
                    avg_power='mean',
                    peak_power='max',
                    std_power='std',
                    sample_count='count'
                ).fillna(0).reset_index()

                def rule_based_filter_dw(row):
                    return int(
                        #row['total_power'] > 3191 and
                        row['total_power'] > 5000 and 
                        #row['peak_power'] > 92 and
                        row['peak_power'] > 150 and    
                        row['std_power'] > 50 and
                        row['sample_count'] > 10 and
                        7 <= row['quarter'].hour <= 23  # optional, restrict to daytime         
                        #row['std_power'] > 6
                    )

                features['dw_active'] = features.apply(rule_based_filter_dw, axis=1)

                print(f"--- {appliance} per-quarter stats ---")
                print(features[['quarter', 'total_power', 'peak_power', 'std_power']])

                features['dw_active'] = enforce_min_cycle(features['dw_active'].values, min_len=3)
                #features['dw_active'] = enforce_min_cycle(features['dw_active'].values)
                df = df.merge(features[['quarter', 'dw_active']], on='quarter', how='left')
                df['dw_active'] = df['dw_active'].fillna(0).astype(int)
                #appliance_power = df['dw_power'] * df['dw_active']
                scaling_factor = 1e5
                # appliance_power = df['dw_power'] 
                appliance_power_s = df['dw_power']/ scaling_factor
                appliance_power = appliance_power_s * df['dw_active']
            # --- End DW Refinement ---
            # --- Clustering-based Refinement for TV ---
            elif appliance.lower() == "tv":
                timestamps = pd.to_datetime(forecasted_timestamps[:len(appliance_power)])
                df = pd.DataFrame({'timestamp': timestamps, 'power': appliance_power})
                df['quarter'] = df['timestamp'].dt.floor('15min')
                features = df.groupby('quarter')['power'].agg(
                    total_power='sum', avg_power='mean', peak_power='max', std_power='std', sample_count='count'
                ).fillna(0).reset_index()

                def rule_based_filter_tv(row):
                    return int(
                        #row['total_power'] >= 2803 and 
                        row['total_power'] >= 1000 and 
                        row['total_power'] <= 50000 and
                        # row['peak_power'] >= 93 and 
                        row['peak_power'] >= 70 and 
                        # row['peak_power'] <= 116 and 
                        row['peak_power'] <= 150 and 
                        row['std_power'] <= 25)
                        #row['std_power'] >= 0 and 
                        #row['std_power'] <= 15)

                features['active'] = features.apply(rule_based_filter_tv, axis=1)

                print(f"--- {appliance} per-quarter stats ---")
                print(features[['quarter', 'total_power', 'peak_power', 'std_power']])

                features['active'] = enforce_min_cycle(features['active'].values, min_len=1)
                df = df.merge(features[['quarter', 'active']], on='quarter', how='left')
                df['active'] = df['active'].fillna(0).astype(int)
                appliance_power = df['power'] * df['active']
                # --- End TV Refinement ---
                # --- Clustering-based Refinement for Washer Dryer ---
            elif appliance.lower() == "washer_dryer":
                timestamps = pd.to_datetime(forecasted_timestamps[:len(appliance_power)])
                df = pd.DataFrame({'timestamp': timestamps, 'wd_power': appliance_power})
                df['quarter'] = df['timestamp'].dt.floor('15min')

                features = df.groupby('quarter')['wd_power'].agg(
                    total_power='sum',
                    avg_power='mean',
                    peak_power='max',
                    std_power='std',
                    sample_count='count'
                ).fillna(0).reset_index()

                def rule_based_filter_wd(row):
                    return int(
                        row['total_power'] >= 30000 and
                        row['peak_power'] >= 3100 and
                        row['std_power'] >= 200
                    )

                features['wd_active'] = features.apply(rule_based_filter_wd, axis=1)

                print(f"--- {appliance} per-quarter stats ---")
                print(features[['quarter', 'total_power', 'peak_power', 'std_power']])

                features['wd_active'] = enforce_min_cycle(features['wd_active'].values, min_len=3)
                df = df.merge(features[['quarter', 'wd_active']], on='quarter', how='left')
                df['wd_active'] = df['wd_active'].fillna(0).astype(int)
                appliance_power = df['wd_power'] * df['wd_active']
                appliance_power = appliance_power / 1e4  
                # --- End WD Refinement ---
            activated_power = appliance_power
            disaggregated_results[appliance] = activated_power.tolist()
            total_disaggregated += activated_power
            print(f"{appliance} max_value: {max_value}")
            print(f"{appliance} predicted sample: {appliance_power[:5]}")

        # Step 3: Remaining consumption
        remaining_consumption = forecasted_consumption - total_disaggregated
        remaining_consumption = np.maximum(remaining_consumption, 0)
        disaggregated_results["Total Remaining Consumption"] = remaining_consumption.tolist()

        # Step 4: Final response
        return {
            "message": "Forecast and disaggregation successful",
            "forecast_timestamps": forecasted_timestamps,
            "forecast_consumption": forecasted_consumption.tolist(),
            "forecast_production": forecast_result.get("forecast_production", []),
            "disaggregated_consumption": disaggregated_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/csv-data/{user_id}/disaggregated-disaggregation-last-day", tags=['Disaggegated Flexibility Forecasting'], response_model=dict)
async def disaggregate_last_day(
    user_id: str,
    country: str,
    model_name: str,  
    selected_appliances: List[str],
    db: Session = Depends(get_db)
):
    """
    Disaggregate real consumption data from the **last full day** into appliances.

    **Returns:**
    - Real timestamps for last 1440 minutes
    - Original consumption series
    - Disaggregated series per appliance
    - Remaining consumption
    """
    import pickle, os

    try:
        # Step 1: Get last 24h cleaned real data
        df = await get_cleaned_disaggregated_csv_data(user_id, db)
        if not df:
            raise HTTPException(status_code=400, detail="No valid data available after cleaning")

        df = pd.DataFrame(df)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values("timestamp")
        df = df.tail(1440)

        if len(df) < 1440:
            raise HTTPException(status_code=400, detail="Not enough data for last full day")

        timestamps = df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        real_consumption = df['consumption'].fillna(0).astype(float).values

        # Step 2: Disaggregation
        disaggregated_results = {}
        total_disaggregated = np.zeros_like(real_consumption, dtype=float)

        for appliance in selected_appliances:
            model_file = f"{appliance}_model.h5"
            model = download_model_from_minio(model_file, user_id)

            if model is None:
                continue  # Skip missing model

            # Load max scaling value
            try:
                pkl_path = f"/tmp/model/{appliance}_model.pkl"
                minio_client.fget_object("disaggregation-models", f"{appliance}_model.pkl", pkl_path)
                with open(pkl_path, "rb") as f:
                    model_info = pickle.load(f)
                    max_value = model_info.get("max", 1.0)
            except Exception as e:
                print(f"Warning: Max value for {appliance} not found: {e}")
                max_value = 1.0

            # Reshape input
            X_test = real_consumption.reshape(-1, 1).astype(float)
            time_steps = model.input_shape[1]
            num_features = model.input_shape[2]

            if X_test.shape[0] < time_steps:
                X_padded = np.zeros((time_steps, num_features), dtype=float)
                X_padded[:X_test.shape[0], :] = X_test
                X_test = np.expand_dims(X_padded, axis=0)
            else:
                X_test = np.array([X_test[i:i + time_steps] for i in range(len(X_test) - time_steps + 1)])

            appliance_power = model.predict(X_test).flatten().astype(float)
            appliance_power = appliance_power[:len(real_consumption)]
            appliance_power = np.maximum(appliance_power * max_value, 0)

            # Timestamp alignment
            aligned_timestamps = timestamps[-len(appliance_power):]

            # Appliance-specific refinement
            df_app = pd.DataFrame({'timestamp': pd.to_datetime(aligned_timestamps), 'power': appliance_power})
            df_app['quarter'] = df_app['timestamp'].dt.floor('15min')

            if appliance.lower() == "tv":
                stats = df_app.groupby('quarter')['power'].agg(['sum', 'mean', 'max', 'std', 'count']).reset_index()
                stats['active'] = stats.apply(lambda row: int(row['sum'] >= 1000 and row['max'] >= 70 and row['max'] <= 150 and row['std'] <= 25), axis=1)
                stats['active'] = enforce_min_cycle(stats['active'].values, min_len=1)
                df_app = df_app.merge(stats[['quarter', 'active']], on='quarter', how='left')
                df_app['active'] = df_app['active'].fillna(0).astype(int)
                appliance_power = df_app['power'] * df_app['active']

            elif appliance.lower() == "washing_machine":
                stats = df_app.groupby('quarter')['power'].agg(['sum', 'mean', 'max', 'std', 'count']).reset_index()
                stats['active'] = stats.apply(lambda row: int(row['sum'] > 1800 and row['max'] > 140 and row['std'] > 10), axis=1)
                stats['active'] = enforce_min_cycle(stats['active'].values, min_len=3)
                df_app = df_app.merge(stats[['quarter', 'active']], on='quarter', how='left')
                df_app['active'] = df_app['active'].fillna(0).astype(int)
                appliance_power = df_app['power'] * df_app['active']

            elif appliance.lower() == "dish_washer":
                stats = df_app.groupby('quarter')['power'].agg(['sum', 'mean', 'max', 'std', 'count']).reset_index()
                stats['active'] = stats.apply(lambda row: int(row['sum'] > 5000 and row['max'] > 150 and row['std'] > 50 and row['count'] > 10 and 7 <= row['quarter'].hour <= 23), axis=1)
                stats['active'] = enforce_min_cycle(stats['active'].values, min_len=3)
                df_app = df_app.merge(stats[['quarter', 'active']], on='quarter', how='left')
                df_app['active'] = df_app['active'].fillna(0).astype(int)
                appliance_power = (df_app['power'] / 1e5) * df_app['active']  # scaling as in original

            elif appliance.lower() == "washer_dryer":
                stats = df_app.groupby('quarter')['power'].agg(['sum', 'mean', 'max', 'std', 'count']).reset_index()
                stats['active'] = stats.apply(lambda row: int(row['sum'] >= 30000 and row['max'] >= 3100 and row['std'] >= 200), axis=1)
                stats['active'] = enforce_min_cycle(stats['active'].values, min_len=3)
                df_app = df_app.merge(stats[['quarter', 'active']], on='quarter', how='left')
                df_app['active'] = df_app['active'].fillna(0).astype(int)
                appliance_power = (df_app['power'] * df_app['active']) / 1e4

            activated_power = appliance_power
            disaggregated_results[appliance] = activated_power.tolist()
            total_disaggregated += activated_power

        # Step 3: Remaining
        remaining_consumption = real_consumption - total_disaggregated
        remaining_consumption = np.maximum(remaining_consumption, 0)
        disaggregated_results["Total Remaining Consumption"] = remaining_consumption.tolist()

        return {
            "message": "Disaggregation completed on last day’s data",
            "timestamps": timestamps[-len(real_consumption):],
            "real_consumption": real_consumption.tolist(),
            "disaggregated_consumption": disaggregated_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def enforce_min_cycle(mask, min_len=3):
    padded = np.pad(mask, (1, 1), constant_values=0)
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    valid = np.zeros_like(mask)
    for s, e in zip(starts, ends):
        if (e - s) >= min_len:
            valid[s:e] = 1
    return valid
    

@app.post("/csv-data/{user_id}/forecast-disaggregated-flexibility", tags=['Disaggegated Flexibility Forecasting'])
async def forecast_disaggregated_flexibility(
    user_id: str, 
    country: str, 
    model_name: str, 
    #days: int, 
    selected_appliances: List[str], 
    db: Session = Depends(get_db)
):
    """
    **Forecast Disaggregated Flexibility (3 Hours Ahead in 15-Min Steps)**

    **Description:**
    - Forecasts energy **consumption** and **production** for the next 3 hours using BiLSTM models.
    - Performs **disaggregation** based on user-selected flexible appliances.
    - Calculates **flexibility** as a percentage per **15-minute** step.

    **Formula:**
    - `Flexibility (%) = (Flexible Loads[15 min] / Total Loads[15 min]) * 100`

    **Parameters:**
    - `user_id` (str): Unique user identifier.
    - `country` (str): Country associated with the dataset.
    - `model_name` (str): Trained forecasting model name.
    - `days` (int): Number of days to forecast (`1-7`).
    - `selected_appliances` (List[str]): List of selected **flexible** appliances.
    - `db` (Session): Database session dependency.

    **Returns:**
    ```json
    {
        "message": "Flexibility calculation successful",
        "flexibility_timestamps": ["YYYY-MM-DD HH:MM:SS", ...],
        "flexibility_percentage": [10.5, 8.3, 12.7, ...]
    }
    ```
    """

    try:
        # Step 1: Perform Forecasting & Disaggregation
        forecast_result = await forecast_disaggregated_disaggregation(
            user_id, country, model_name, selected_appliances, db
        )

        # Step 2: Calculate Flexibility
        flexibility_result = await calculate_disaggregated_flexibility(forecast_result, selected_appliances)

        return flexibility_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def calculate_disaggregated_flexibility(forecast_data: dict, flexible_appliances: list):
    """
    **Calculate Disaggregated Flexibility (15-Minute Steps)**

    - Aggregates total and flexible consumption in **15-minute steps**.
    - Computes flexibility **percentage** per step.
    - Ensures all values stay in the **valid range (0-100%)**.

    **Returns:** JSON response with flexibility timestamps and percentages.
    """
    try:
        # Extract forecast data
        timestamps = pd.to_datetime(forecast_data["forecast_timestamps"])
        total_consumption = np.array(forecast_data["forecast_consumption"], dtype=float)
        disaggregated_consumption = forecast_data["disaggregated_consumption"]

        # Ensure valid data
        if len(total_consumption) == 0 or len(timestamps) == 0:
            raise HTTPException(status_code=400, detail="Forecast data is empty.")

        # Step 1: Aggregate into 15-minute intervals
        df = pd.DataFrame({"timestamp": timestamps, "total_consumption": total_consumption})

        # Sum flexible appliance loads
        df["flexible_loads"] = np.zeros_like(total_consumption)
        for appliance in flexible_appliances:
            if appliance in disaggregated_consumption:
                df["flexible_loads"] += np.array(disaggregated_consumption[appliance], dtype=float)

        # Resample into 15-minute intervals (sum of consumption in each period)
        df = df.set_index("timestamp").resample("15T").sum()

        # Step 2: Calculate Flexibility (%)
        df["flexibility_percentage"] = (df["flexible_loads"] / df["total_consumption"]) * 100
        df["flexibility_percentage"] = df["flexibility_percentage"].clip(0, 100)  # Ensure values stay in range

        # Convert timestamps back to string for JSON response
        df.reset_index(inplace=True)
        df["timestamp"] = df["timestamp"].astype(str)

        # Step 3: Format JSON response
        response = {
            "message": "Flexibility calculation successful",
            "flexibility_timestamps": df["timestamp"].tolist(),
            "flexibility_percentage": df["flexibility_percentage"].tolist()
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======== Prosumer ===========
async def compute_dynamic_forecast_errors(user_id: str, country: str, model_name: str, db: Session):
    """
    Computes dynamic forecast errors for the last 10 days using the trained BiLSTM model.
    - Fetches actual data.
    - Runs the trained model from MinIO on the last 10 days of data.
    - Computes RMSE for consumption and production.

    **Returns:**
    - RMSE for consumption
    - RMSE for production (if available)
    """

    # Step 1: Fetch actual data from the last 10 days
    df_actual = await get_cleaned_disaggregated_csv_data(user_id, db)
    df_actual = pd.DataFrame(df_actual)

    if df_actual.empty:
        raise HTTPException(status_code=400, detail="Not enough actual data for error computation.")

    # Step 2: Select the last 10 days of data
    df_actual['timestamp'] = pd.to_datetime(df_actual['timestamp'])
    last_timestamp = df_actual['timestamp'].max()
    first_timestamp = last_timestamp - pd.Timedelta(days=10)
    last_10_days = df_actual[df_actual['timestamp'] >= first_timestamp]

    if last_10_days.empty:
        raise HTTPException(status_code=400, detail="Insufficient data for error computation.")

    # Step 3: Fetch the trained model from MinIO
    user_bucket = await create_user_bucket(user_id, country)

    # Load MinIO Models
    consumption_model_path = f"/tmp/{model_name}_consumption.h5"
    minio_client.fget_object(user_bucket, f"{model_name}_consumption.h5", consumption_model_path)
    model_consumption = load_model(consumption_model_path)

    model_production = None
    production_model_path = f"/tmp/{model_name}_production.h5"
    try:
        minio_client.fget_object(user_bucket, f"{model_name}_production.h5", production_model_path)
        model_production = load_model(production_model_path)
    except Exception as e:
        print(f"[Warning] Production model not found: {e}")

    # production_model_path = f"/tmp/{model_name}_production.h5"
    # minio_client.fget_object(user_bucket, f"{model_name}_production.h5", production_model_path)
    # model_production = load_model(production_model_path) if "production" in last_10_days.columns else None

    # ==============================
    # APPLY SAME FEATURE ENGINEERING AS TRAINING
    # ==============================

    # Ensure Correct Data Types
    last_10_days['timestamp'] = pd.to_datetime(last_10_days['timestamp'])
    last_10_days['consumption'] = pd.to_numeric(last_10_days['consumption'], errors='coerce')
    if "production" in last_10_days.columns:
        last_10_days['production'] = pd.to_numeric(last_10_days['production'], errors='coerce')

    # Drop NaNs
    last_10_days = last_10_days.dropna()

    # Add Features (like training)
    last_10_days['hour'] = last_10_days['timestamp'].dt.hour
    last_10_days['day_of_week'] = last_10_days['timestamp'].dt.dayofweek
    last_10_days['month'] = last_10_days['timestamp'].dt.month
    last_10_days['day'] = last_10_days['timestamp'].dt.day
    last_10_days['is_weekend'] = last_10_days['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Create Rolling Stats & Lags
    def create_features(df, col):
        roll_10 = df[col].rolling(window=10, min_periods=1)
        roll_60 = df[col].rolling(window=60, min_periods=1)

        df[f'rolling_mean_10min_{col}'] = roll_10.mean()
        df[f'rolling_std_10min_{col}'] = roll_10.std()
        df[f'rolling_max_10min_{col}'] = roll_10.max()
        df[f'rolling_min_10min_{col}'] = roll_10.min()
        df[f'rolling_mean_60min_{col}'] = roll_60.mean()
        df[f'rolling_std_60min_{col}'] = roll_60.std()

        for lag in [1, 5, 10, 30]:
            df[f'lag_{lag}min_{col}'] = df[col].shift(lag)

    create_features(last_10_days, "consumption")
    has_production = "production" in last_10_days.columns and model_production is not None
    if has_production:
        create_features(last_10_days, "production")
    # if "production" in last_10_days.columns:
    #     create_features(last_10_days, "production")

    # Drop NaN values caused by shifting
    last_10_days = last_10_days.dropna()

    # 6Define Feature Sets (like training)
    consumption_features = [
        'hour', 'day_of_week', 'month', 'day', 'is_weekend', 'consumption',
        'rolling_mean_10min_consumption', 'rolling_std_10min_consumption', 'rolling_max_10min_consumption',
        'rolling_min_10min_consumption', 'rolling_mean_60min_consumption', 'rolling_std_60min_consumption',
        'lag_1min_consumption', 'lag_5min_consumption', 'lag_10min_consumption', 'lag_30min_consumption'
    ]
    if has_production:

    # has_production = "production" in last_10_days.columns
        consumption_features += [
            'rolling_mean_10min_production', 'rolling_std_10min_production', 'rolling_max_10min_production', 'production',
            'rolling_min_10min_production', 'rolling_mean_60min_production', 'rolling_std_60min_production',
            'lag_1min_production', 'lag_5min_production', 'lag_10min_production', 'lag_30min_production'
        ]

    # Scale Features
    scaler = MinMaxScaler()
    last_10_days[consumption_features] = scaler.fit_transform(last_10_days[consumption_features])

    # ==============================
    # GENERATE SEQUENCES (LIKE TRAINING)
    # ==============================
    look_back = 60
    step = 5
    X_test_c, y_test_c = create_sequences_np(last_10_days, "consumption", consumption_features, look_back, step)

    if X_test_c.shape[0] == 0:
        raise HTTPException(status_code=500, detail="No valid consumption test samples for error computation.")

    # Debugging Output
    print(f"Expected Model Input Shape: (None, {look_back}, {len(consumption_features)})")
    print(f"Actual Input Shape Passed: {X_test_c.shape}")

    if X_test_c.shape[2] != len(consumption_features):
        raise HTTPException(status_code=500, detail=f"Shape mismatch: Expected {len(consumption_features)} features, got {X_test_c.shape[2]}")

    # Predict & Compute Errors
    y_pred_c = model_consumption.predict(X_test_c)
    mse_consumption = mean_squared_error(y_test_c, y_pred_c)
    rmse_consumption = np.sqrt(mse_consumption)

    # ==============================
    # PRODUCTION FORECAST ERROR
    # ==============================
    rmse_production = 0  # Default if no production data
    if has_production:
    #if has_production and model_production:
        X_test_p, y_test_p = create_sequences_np(last_10_days, "production", consumption_features, look_back, step)
        if X_test_p.shape[0] > 0:
            y_pred_p = model_production.predict(X_test_p)
            mse_production = mean_squared_error(y_test_p, y_pred_p)
            rmse_production = np.sqrt(mse_production)

    return rmse_consumption, rmse_production

async def calculate_prosumer_flexibility(forecast_data: dict, flexible_appliances: list, user_id: str, country: str, model_name: str, db: Session):
    """
    **Calculate Prosumer Flexibility with Dynamic Error Adjustments (15-Minute Steps)**

    - Aggregates total and flexible consumption in **15-minute steps**.
    - Identifies **high PV production periods**.
    - **Shifts flexible loads** to align with peak PV production.
    - Computes flexibility **percentage** per step.
    - Adjusts flexibility calculation dynamically based on **rolling forecast errors**.

    **Returns:** JSON response with flexibility timestamps, adjusted flexibility, and error bounds.
    """
    try:
        # Step 1: Extract forecasted data
        timestamps = pd.to_datetime(forecast_data["forecast_timestamps"])
        total_consumption = np.array(forecast_data["forecast_consumption"], dtype=float)
        # forecasted_production = np.array(forecast_data.get("forecast_production", np.zeros_like(total_consumption)), dtype=float)
        # disaggregated_consumption = forecast_data["disaggregated_consumption"]

        # Align production to consumption length
        forecasted_production = np.array(forecast_data.get("forecast_production", []), dtype=float)
        if len(forecasted_production) < len(total_consumption):
            forecasted_production = np.pad(forecasted_production, (0, len(total_consumption) - len(forecasted_production)), constant_values=0)
        elif len(forecasted_production) > len(total_consumption):
            forecasted_production = forecasted_production[:len(total_consumption)]

        disaggregated_consumption = forecast_data["disaggregated_consumption"]

        if len(total_consumption) == 0 or len(timestamps) == 0:
            raise HTTPException(status_code=400, detail="Forecast data is empty.")

        # Step 2: Compute dynamic forecasting errors
        rmse_forecast, rmse_pv = await compute_dynamic_forecast_errors(user_id, country, model_name, db)

        # Step 3: Create DataFrame for processing
        df = pd.DataFrame({
            "timestamp": timestamps,
            "previous_load": total_consumption,
            "production": forecasted_production
        })

        # Step 4: Compute flexible loads per appliance
        df["flexible_loads"] = np.zeros_like(total_consumption)
        for appliance in flexible_appliances:
            if appliance in disaggregated_consumption:
                df["flexible_loads"] += np.array(disaggregated_consumption[appliance], dtype=float)

        # Step 5: Resample into 15-minute intervals
        df = df.set_index("timestamp").resample("15T").sum()

        # Step 6: Identify High PV Production Periods
        production_threshold = df["production"].quantile(0.75)
        df["high_production"] = df["production"] >= production_threshold

        # Step 7: Shift Flexible Loads to High PV Production Periods
        df["new_flexible_loads"] = df["flexible_loads"].copy()
        for index in df.index:
            if df.loc[index, "high_production"]:
                df.loc[index, "new_flexible_loads"] += df.loc[index, "flexible_loads"]
                df.loc[index, "flexible_loads"] = 0

        # Step 8: Compute New Load
        df["new_load"] = df["previous_load"] - df["flexible_loads"] + df["new_flexible_loads"]

        # Step 9: Compute Adjusted Flexibility
        df["flexibility_adjusted"] = ((df["new_load"] - df["previous_load"]) / df["previous_load"]) * 100

        # Step 10: Compute Corrected Error Bounds
        k = 1.96  # 95% Confidence Interval
        delta_forecast_percent = rmse_forecast
        delta_pv_percent = rmse_pv
        delta_disagg_percent = np.sqrt(0.06482**2 + 0.28956**2 + 0.48426**2 + 0.23471**2) 

        df["error_factor"] = k * np.sqrt(delta_forecast_percent**2 + delta_pv_percent**2 + delta_disagg_percent**2)

        df["flexibility_upper"] = ((df["new_load"] + k * np.sqrt(delta_forecast_percent**2 + delta_pv_percent**2 + delta_disagg_percent**2) - df["previous_load"]- delta_forecast_percent) / (df["previous_load"]+ delta_forecast_percent)) * 100
        df["flexibility_lower"] = ((df["new_load"] - k * np.sqrt(delta_forecast_percent**2 + delta_pv_percent**2 + delta_disagg_percent**2) - df["previous_load"]+ delta_forecast_percent) / (df["previous_load"]- delta_forecast_percent)) * 100


        # Convert timestamps back to string for JSON response
        df.reset_index(inplace=True)
        df["timestamp"] = df["timestamp"].astype(str)

        return {
            "message": "Prosumer flexibility calculation successful",
            "timestamps": df["timestamp"].tolist(),
            "previous_load": df["previous_load"].tolist(),
            "new_load": df["new_load"].tolist(),
            "flexibility_adjusted": df["flexibility_adjusted"].tolist(),
            "flexibility_upper": df["flexibility_upper"].tolist(),
            "flexibility_lower": df["flexibility_lower"].tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/csv-data/{user_id}/forecast-disaggregated-prosumer-flexibility", tags=['Disaggegated Flexibility Forecasting'])
async def forecast_disaggregated_prosumer_flexibility(
    user_id: str, 
    country: str, 
    model_name: str, 
    #days: int, 
    selected_appliances: List[str], 
    db: Session = Depends(get_db)
):
    """
    **Forecast Prosumer Flexibility (3 Hours Ahead in 15-Minute Steps)**

    **Description:**
    - Forecasts **consumption** and **production** for next 3 hours using BiLSTM models.
    - Performs **disaggregation** for flexible appliances.
    - **Shifts loads** to periods of **high PV production**.
    - Calculates **prosumer flexibility** as the difference between original and shifted consumption.
    - **Includes uncertainty propagation** from **dynamic forecasting errors**.

    **Returns:**
    ```json
    {
        "message": "Prosumer flexibility calculation successful",
        "timestamps": ["YYYY-MM-DD HH:MM:SS", ...],
        "previous_load": [5.0, 5.2, ...],
        "new_load": [2.0, 2.2, ...],
        "flexibility_adjusted": [-60.5, -58.3, ...],
        "flexibility_upper": [-55.1, -52.8, ...],
        "flexibility_lower": [-65.8, -63.2, ...]
    }
    ```
    """
    try:
        # Step 1: Compute Dynamic Forecasting Errors
        delta_forecast, delta_pv = await compute_dynamic_forecast_errors(user_id, country, model_name, db)

        # Step 2: Perform Forecasting & Disaggregation
        forecast_result = await forecast_disaggregated_disaggregation(
            user_id, country, model_name, selected_appliances, db
        )

        # Step 3: Calculate Prosumer Flexibility with Errors
        flexibility_result = await calculate_prosumer_flexibility(
            forecast_result, selected_appliances, user_id, country, model_name, db
        )

        return flexibility_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





