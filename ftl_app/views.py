# ftl_app/views.py
from django.shortcuts import render
from .ftl_core.ftl_model import run_ftl_simulation
from .ftl_core.evaluation import simulate_inference_attack
from .models import SimulationResult
import json

def index(request):
    return render(request, 'index.html')

def test_model(request):
    if request.method == 'POST':
        num_rounds = int(request.POST.get('num_rounds', 5))
        num_devices = int(request.POST.get('num_devices', 3))
        noise_multiplier = float(request.POST.get('noise_multiplier', 1.1))
        mode = request.POST.get('mode', 'live')
        
        accuracy, loss, leakage, latency, epsilon = run_ftl_simulation(
            num_rounds=num_rounds, num_devices=num_devices, 
            noise_multiplier=noise_multiplier, precomputed=(mode == 'precomputed')
        )
        
        result = SimulationResult(
            num_rounds=num_rounds, num_devices=num_devices, noise_multiplier=noise_multiplier,
            accuracy=json.dumps(accuracy), loss=json.dumps(loss), 
            leakage=json.dumps(leakage), latency=json.dumps(latency), epsilon=json.dumps(epsilon)
        )
        result.save()
        
        attack_success = simulate_inference_attack(num_devices)
        
        context = {
            'accuracy': accuracy, 'loss': loss, 'leakage': leakage, 'latency': latency,
            'epsilon': epsilon, 'attack_success': attack_success, 'num_rounds': num_rounds,
            'num_devices': num_devices, 'noise_multiplier': noise_multiplier, 'mode': mode
        }
        return render(request, 'test_model.html', context)
    return render(request, 'test_model.html')