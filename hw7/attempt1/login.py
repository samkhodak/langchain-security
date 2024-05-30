from flask import render_template, session, redirect, url_for
from flask.views import MethodView

class LoginView(MethodView):
    def get(self):
        if 'oauth_token' in session:
            return redirect(url_for('index'))
        return render_template('login.html')
