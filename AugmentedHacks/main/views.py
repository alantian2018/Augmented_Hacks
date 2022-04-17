from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import loader

def show_home(request):
    return render(request=request, template_name="home.html")
def show_explore(request):
    return render(request=request, template_name="climap.html")
def show_about (request):
    return render (request , 'about_us.html')