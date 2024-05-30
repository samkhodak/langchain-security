import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client_id = os.getenv('GOOGLE_CLIENT_ID')
client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
redirect_uri = os.getenv('REDIRECT_URI', 'http://127.0.0.1:5000/login/authorized')
authorization_base_url = os.getenv('AUTHORIZATION_BASE_URL', 'https://accounts.google.com/o/oauth2/auth')
token_url = os.getenv('TOKEN_URL', 'https://accounts.google.com/o/oauth2/token')
