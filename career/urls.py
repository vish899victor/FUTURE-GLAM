from django.urls import path
from . import views

urlpatterns = [
    path('career/<str:data>',views.career,name="data"),]
