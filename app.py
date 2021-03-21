import os
from flask import Flask, render_template, session, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField
from wtforms.validators import NumberRange
import numpy as np
from model.train import train_model
import joblib
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets

matplotlib.use('Agg')


# Iris photo album
IRIS_ALBUM = 'static'
# Scatter plots colours
COLOURS = ['green', 'orange', 'pink']
# Iris features names
FEATURES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

def make_prediction(model, scaler, sample_json):
    sepal_length = sample_json['sepal_length']
    sepal_width = sample_json['sepal_width']
    petal_length = sample_json['petal_length']
    petal_width = sample_json['petal_width']

    flower = [[sepal_length, sepal_width, petal_length, petal_width]]
    flower = scaler.transform(flower)

    classes = np.array(['setosa', 'versicolor', 'virginica'])
    class_ind = model.predict(flower)

    return classes[np.argmax(class_ind, axis=1)][0]

def return_iris_photo(iris_class):
    if iris_class == 'setosa':
        filename = 'iris_setosa.jpg'
    elif iris_class == 'versicolor':
        filename = 'iris_versicolor.jpg'
    else:
        filename = 'iris_virginica.jpg'
    
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return full_filename

def plot_scatter(user_input, x_axis, y_axis):
    iris_dataset = datasets.load_iris()
    
    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Scatter plot of 3 types of Iris
    for i in range(0, 3):
        plt.scatter(
            iris_dataset.data[iris_dataset.target==i, FEATURES.index(x_axis)], 
            iris_dataset.data[iris_dataset.target==i, FEATURES.index(y_axis)], 
            c=COLOURS[i], alpha=0.5, label=list(iris_dataset.target_names)[i])

    # YOUR INPUT with annotation
    plt.scatter(user_input[x_axis], user_input[y_axis], c='r')
    plt.text(user_input[x_axis]+0.03, user_input[y_axis]+0.03, 'YOUR INPUT', color='r')

    plt.title(f'Iris dataset: {iris_dataset.feature_names[FEATURES.index(y_axis)]} vs {iris_dataset.feature_names[FEATURES.index(x_axis)]}')
    plt.xlabel(iris_dataset.feature_names[FEATURES.index(x_axis)])
    plt.ylabel(iris_dataset.feature_names[FEATURES.index(y_axis)])
    plt.legend()

    scatter_filename = f'scatter-{y_axis}-{x_axis}.png'
    scatter_full_filename = os.path.join(app.config['UPLOAD_FOLDER'], scatter_filename)
    plt.savefig(scatter_full_filename)
    return scatter_full_filename


# Make an instance of Flask
app = Flask(__name__)

# Configure SECRET_KEY, which is needed to keep the client-side sessions secure in Flask. 
app.config['SECRET_KEY'] = 'someRandomKey'
app.config['UPLOAD_FOLDER'] = IRIS_ALBUM

# check whether the model is already trained or not
if not os.path.isfile('trained_models/iris-model.pkl'):
    train_model()

# Load the model and scaler
trained_model = joblib.load('trained_models/iris-model.pkl')
trained_scaler = joblib.load('trained_models/iris-scaler.pkl')

# Creat an WTForm Class, TextField Represents <input type = 'text'>
class FlowerForm(FlaskForm):
    sep_len = TextField('Sepal Length (cm): ')
    sep_wid = TextField('Sepal Width (cm): ')
    pet_len = TextField('Petal Length (cm): ')
    pet_wid = TextField('Petal Width (cm): ')

    submit = SubmitField('Analyze')

# Endpoint: homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    # Create instance of the form
    form = FlowerForm()
    
    # If the form is valid on submission
    if form.validate_on_submit():
        session['sep_len'] = form.sep_len.data
        session['sep_wid'] = form.sep_wid.data
        session['pet_len'] = form.pet_len.data
        session['pet_wid'] = form.pet_wid.data
        
        return redirect(url_for('result_page'))
    
    return render_template('home.html', form=form)

# Endpoint: prediction results page
@app.route('/prediction')
def result_page():
    content = {}

    content['sepal_length'] = float(session['sep_len'])
    content['sepal_width'] = float(session['sep_wid'])
    content['petal_length'] = float(session['pet_len'])
    content['petal_width'] =  float(session['pet_wid'])

    results = make_prediction(trained_model, trained_scaler, sample_json=content)
    iris_image = return_iris_photo(results)
    scatter1 = plot_scatter(user_input=content, x_axis='sepal_length', y_axis='sepal_width')
    scatter2 = plot_scatter(user_input=content, x_axis='petal_length', y_axis='petal_width')

    return render_template(
        'prediction.html', results=results, image=iris_image,
        plot1=scatter1, plot2=scatter2)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)