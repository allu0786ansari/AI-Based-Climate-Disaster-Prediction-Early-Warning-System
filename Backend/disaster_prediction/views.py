from rest_framework.response import Response
from rest_framework.decorators import api_view
from .ai.lstm_flood_model import predict_flood
from .ai.yolo_wildfire_model import detect_wildfire

@api_view(['POST'])
def disaster_prediction(request):
    data = request.data
    flood_risk = predict_flood(data['climate_factors'])
    wildfire_risk = detect_wildfire(data['image'])
    
    return Response({
        "flood_risk": flood_risk,
        "wildfire_risk": wildfire_risk
    })
