from django.shortcuts import render
from .models import fields

# Create your views here.
def index(request):
    fieldx = fields.objects.all()
    return render(request,'index.html', {'fieldx': fieldx})


