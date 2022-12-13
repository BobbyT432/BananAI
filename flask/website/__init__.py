from flask import Flask
from .views import routes

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'boeing' # encrypts the data for our website (cookies, etc.)

    # Before it can access the decorator location it goes
    # through here first (example, if it was /route, and the original was home,
    # then it would be /route/home)
    app.register_blueprint(routes, url_prefix='/') 

    return app
    