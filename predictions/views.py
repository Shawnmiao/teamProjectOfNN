from django.shortcuts import render
from django.http import JsonResponse
import json
import torch
import joblib
from django.views.decorators.csrf import csrf_exempt

from .train_models import CarPricePredictor

# Load the trained model and scaler
model = CarPricePredictor()
model.load_state_dict(torch.load('car_price_predictor.pth'))
model.eval()
scaler = joblib.load('scaler.pkl')  # Load your trained scaler

# def index(request):
#     return render(request, 'predictions/homepage.html')

@csrf_exempt
def predict_price(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        features = [
            data['year'],
            data['mileage'],
            data['tax'],
            data['mpg'],
            data['engineSize']
        ]
        print(features)
        # One-hot encode categorical features based on user input
        model_features = [1 if data['model'] == m else 0 for m in ['A1','A6', 'A4','A3','Q3','Q5','A5','S4','Q2','A7','TT','Q7',
 'RS6','RS3','A8','Q8','RS4','RS5','R8','SQ5','S8',',SQ7','S3','S5','A2','RS7']]
        transmission_features = [1 if data['transmission'] == t else 0 for t in ['Manual', 'Automatic','Semi-auto',]]
        fuelType_features = [1 if data['fuelType'] == f else 0 for f in ['Petrol', 'Diesel','Hybrid']]

        # Combine all features
        features.extend(model_features + transmission_features + fuelType_features)

        # Scale the features
        features = scaler.transform([features])

        # Make prediction
        inputs = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(inputs).item()
        print('prediction:',prediction)
        return JsonResponse({'predicted_price': prediction})
    else:
        return render(request, 'predictions/homepage.html')