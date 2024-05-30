from flask import Flask, redirect, url_for, session, request, render_template
from flask.views import MethodView
from config import client_id, client_secret, redirect_uri, authorization_base_url, token_url
from index import IndexView
from requests_oauthlib import OAuth2Session
from login import LoginView
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
scope = ['email']

app.add_url_rule('/', view_func=IndexView.as_view('index'))
app.add_url_rule('/login', view_func=LoginView.as_view('login'))

@app.route('/authenticate')
def authenticate():
    google = OAuth2Session(client_id, redirect_uri=redirect_uri, scope=scope)
    print(authorization_base_url)
    authorization_url, state = google.authorization_url(authorization_base_url)
    session['oauth_state'] = state
    return redirect(authorization_url)

@app.route('/login/authorized')
def authorized():
    google = OAuth2Session(client_id, state=session['oauth_state'], redirect_uri=redirect_uri)
    token = google.fetch_token(token_url, client_secret=client_secret, authorization_response=request.url)
    session['oauth_token'] = token
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('oauth_token', None)
    return redirect(url_for('index'))


if __name__ == '__main__':
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    app.run(debug=True)
