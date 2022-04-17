from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import loader

def show_index(request):
    return render(request=request, template_name="index.html")
def show_home(request):
    return render(request=request, template_name="climap.html")