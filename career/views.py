from django.shortcuts import render
from Home.models import fields

# Create your views here.
def career(request,data):
    fieldx=fields.objects.all()
    if data=='Blockchain':
        data='Blockchain Developer'
    if data=='ML':
        data='Machine Learning Experts'
    if data=='medical':
        data='Medical Professionals'
    if data=='DS':
        data='Data Scientist'
    return render(request,'career.html',{'fieldx': fieldx,'data':data})

