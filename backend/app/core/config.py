import os
from dotenv import load_dotenv

load_dotenv()

MONGO_ROOT_USER = os.getenv("MONGO_ROOT_USER")
MONGO_ROOT_PASSWORD = os.getenv("MONGO_ROOT_PASSWORD")
MONGO_PORT = os.getenv("MONGO_PORT")
MONGO_HOST = os.getenv("MONGO_HOST")

MONGO_URL = "mongodb://" + MONGO_ROOT_USER + ":" + MONGO_ROOT_PASSWORD + \
    "@" + MONGO_HOST + ":" + MONGO_PORT + "/"

DATABASE_NAME = os.getenv("MONGO_ROOT_DATABASE")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")

ML_URL = os.getenv("ML_URL")


settings = {
    'MONGO_URL': MONGO_URL,
    'DATABASE_NAME': DATABASE_NAME,
    'SECRET_KEY': SECRET_KEY,
    'ALGORITHM': ALGORITHM,
    'ACCESS_TOKEN_EXPIRE_MINUTES': ACCESS_TOKEN_EXPIRE_MINUTES,
    'ML_URL': ML_URL,
}

# envs = {key: os.getenv(key) for key in os.environ.keys()}
