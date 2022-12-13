# put our routes here
from flask import Blueprint, render_template

routes = Blueprint('routes', __name__)

@routes.route('/') # decorator
def home():
    return render_template("index.html")

@routes.route('/test')
def test():
    return render_template("test.html")
    