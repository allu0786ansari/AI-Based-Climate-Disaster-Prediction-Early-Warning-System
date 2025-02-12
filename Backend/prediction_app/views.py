from django.http import JsonResponse
from .ai.lstm_flood_model import train_flood_model
from .ai.yolo_wildfire_model import detect_fire
from .ai.disaster_heatmap import generate_heatmap

def predict_flood(request):
    train_flood_model()
    return JsonResponse({"message": "Flood Prediction Model Trained!"})

def detect_wildfire(request):
    detect_fire("../dataset/satellite_image.jpg")
    return JsonResponse({"message": "Wildfire Detection Completed!"})

def get_heatmap(request):
    generate_heatmap()
    return JsonResponse({"message": "Heatmap Generated!", "url": "/static/heatmap.png"})
