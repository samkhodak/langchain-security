from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    CLIENT_ID = os.getenv('CLIENT_ID')
    CLIENT_SECRET = os.getenv('CLIENT_SECRET')
    AUTH_BASE_URL = os.getenv('AUTH_BASE_URL')
    TOKEN_URL = os.getenv('TOKEN_URL')
    REDIRECT_URI = os.getenv('REDIRECT_URI')
