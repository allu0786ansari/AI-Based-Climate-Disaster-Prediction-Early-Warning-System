from django.urls import path
from .views import predict_flood, detect_wildfire, get_heatmap

urlpatterns = [
    path('predict_flood/', predict_flood),
    path('detect_wildfire/', detect_wildfire),
    path('get_heatmap/', get_heatmap),
]
