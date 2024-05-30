import os
from flask import Flask, redirect, url_for, session, request, render_template
from flask.views import MethodView
from requests_oauthlib import OAuth2Session
from config import Config

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config.from_object(Config)

class LoginView(MethodView):
    def get(self):
        google = OAuth2Session(app.config['CLIENT_ID'], redirect_uri=app.config['REDIRECT_URI'], scope=['openid', 'email'])
        authorization_url, state = google.authorization_url(app.config['AUTH_BASE_URL'], access_type="offline", prompt="select_account")
        session['oauth_state'] = state
        return redirect(authorization_url)

class AuthorizedView(MethodView):
    def get(self):
        google = OAuth2Session(app.config['CLIENT_ID'], state=session['oauth_state'], redirect_uri=app.config['REDIRECT_URI'])
        token = google.fetch_token(app.config['TOKEN_URL'], client_secret=app.config['CLIENT_SECRET'], authorization_response=request.url)
        session['oauth_token'] = token
        user_info = google.get('https://www.googleapis.com/oauth2/v1/userinfo').json()
        session['email'] = user_info['email']
        return redirect(url_for('main'))

class MainView(MethodView):
    def get(self):
        if 'email' not in session:
            return render_template('login.html')
        return render_template('main.html', email=session['email'])

class LogoutView(MethodView):
    def get(self):
        session.clear()
        return render_template('login.html')

app.add_url_rule('/', view_func=MainView.as_view('main'))
app.add_url_rule('/login', view_func=LoginView.as_view('login'))
app.add_url_rule('/login/authorized', view_func=AuthorizedView.as_view('authorized'))
app.add_url_rule('/logout', view_func=LogoutView.as_view('logout'))

if __name__ == '__main__':
    app.run(debug=True)
