from django.conf.urls import url
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import *

urlpatterns = [
path('',chat,name="chat"),
]+static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
