from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict.html', views.pred, name='pred'),
    path('index.html', views.index, name='index'),
    path('about.html', views.about, name='about'),
    path('contact.html', views.contact, name='contact'),
]