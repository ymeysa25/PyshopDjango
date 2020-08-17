from django.http import HttpResponse
from django.shortcuts import render
from .models import Product
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

# Create your views here.


def index(request):
    products = Product.objects.all() #to get all data from database
    return render(request, 'index.html',
                  {'products': products}
                  )

    # return HttpResponse('Hello World dddddd')


def new(request):
    music_data = pd.read_csv('music.csv')
    # memisahkan input dan output

    x = music_data.drop('genre', axis=1)  # input columns, tipe data dataframe
    y = music_data['genre']  # output column, tipe data series

    # train a dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # allocing 20% of data to train

    # create a model
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    # make prediction
    # predictions = model.predict([[21,1],[22,0]]) #male(21) and Female(22)
    predictions = model.predict(x_test)
    print(predictions)
    # calculate accuracy
    score = accuracy_score(y_test, predictions)

    return HttpResponse(score)
