from flask import render_template, session, redirect, url_for
from requests_oauthlib import OAuth2Session
from config import client_id
from flask.views import MethodView

class IndexView(MethodView):
    def get(self):
        if 'oauth_token' in session:
            google = OAuth2Session(client_id, token=session['oauth_token'])
            userinfo = google.get('https://www.googleapis.com/oauth2/v1/userinfo').json()
            return render_template('index.html', email=userinfo["email"])
        return redirect(url_for('login'))
