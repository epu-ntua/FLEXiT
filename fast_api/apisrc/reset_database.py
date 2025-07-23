from sqlalchemy import create_engine, MetaData
from service import Base, my_database_connection  

# Create an engine and metadata
engine = create_engine(my_database_connection, pool_pre_ping=True)
metadata = MetaData()

# Reflect the existing database
metadata.reflect(bind=engine)

# Drop the 'csv_data' table if it exists
if 'csv_data' in metadata.tables:
    metadata.tables['csv_data'].drop(bind=engine)
    print("Dropped existing 'csv_data' table.")

# Recreate the tables based on the updated models
Base.metadata.create_all(bind=engine)
print("Recreated all tables.")

