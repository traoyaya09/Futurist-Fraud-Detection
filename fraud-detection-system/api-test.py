
"""
Fraud Detection API - Test Script
==================================

Quick testing script for the FastAPI fraud detection service.

Usage:
    1. Start the API:
       uvicorn fraud_detection_service:app --reload
    
    2. Run tests:
       python api_test.py

Author: Fraud Detection System v2.0.1
Date: 2026-03-16
"""

import sys
import json
import time
from typing import Dict, List
import requests
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

API_BASE_URL = "http://localhost:8000"
TIMEOUT = 10  # seconds

# ═══════════════════════════════════════════════════════════════════════════
# TEST DATA
# ═══════════════════════════════════════════════════════════════════════════

# Sample transactions (realistic values)
TEST_TRANSACTIONS = {
    "low_risk": {
        "time": 123.45,
        "v1": -1.0, "v2": 0.5, "v3": 0.2, "v4": -0.8, "v5": 0.3,
        "v6": 0.1, "v7": -0.4, "v8": 0.0, "v9": 0.6, "v10": -0.2,
        "v11": 0.4, "v12": -0.1, "v13": 0.0, "v14": 0.5, "v15": -0.3,
        "v16": 0.2, "v17": -0.5, "v18": 0.1, "v19": 0.0, "v20": 0.3,
        "v21": -0.2, "v22": 0.4, "v23": -0.1, "v24": 0.0, "v25": 0.2,
        "v26": -0.3, "v27": 0.1, "v28": 0.0,
        "amount": 50.00
    },
    "medium_risk": {
        "time": 456.78,
        "v1": -2.5, "v2": 1.8, "v3": 1.2, "v4": -1.5, "v5": 0.9,
        "v6": 0.7, "v7": -1.2, "v8": 0.5, "v9": 1.4, "v10": -0.8,
        "v11": 1.1, "v12": -0.6, "v13": 0.3, "v14": 1.3, "v15": -0.9,
        "v16": 0.8, "v17": -1.1, "v18": 0.6, "v19": 0.4, "v20": 0.9,
        "v21": -0.7, "v22": 1.0, "v23": -0.5, "v24": 0.3, "v25": 0.7,
        "v26": -0.8, "v27": 0.4, "v28": 0.2,
        "amount": 500.00
    },
    "high_risk": {
        "time": 789.12,
        "v1": -5.0, "v2": 3.5, "v3": 2.8, "v4": -3.2, "v5": 2.1,
        "v6": 1.8, "v7": -2.5, "v8": 1.5, "v9": 2.9, "v10": -1.8,
        "v11": 2.4, "v12": -1.5, "v13": 1.2, "v14": 2.7, "v15": -2.0,
        "v16": 1.9, "v17": -2.3, "v18": 1.6, "v19": 1.3, "v20": 2.1,
        "v21": -1.7, "v22": 2.2, "v23": -1.4, "v24": 1.1, "v25": 1.8,
        "v26": -1.9, "v27": 1.2, "v28": 0.9,
        "amount": 2500.00
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def print_header(text: str) -> None:
    """Print colored header."""
    print("\n" + "=" * 80)
    print(Fore.CYAN + Style.BRIGHT + text.center(80))
    print("=" * 80)

def print_success(text: str) -> None:
    """Print success message."""
    print(Fore.GREEN + "✓ " + text)

def print_error(text: str) -> None:
    """Print error message."""
    print(Fore.RED + "✗ " + text)

def print_info(text: str) -> None:
    """Print info message."""
    print(Fore.YELLOW + "ℹ " + text)

def format_json(data: Dict) -> str:
    """Format JSON with indentation."""
    return json.dumps(data, indent=2)

def get_risk_color(risk_level: str) -> str:
    """Get color for risk level."""
    colors = {
        "LOW": Fore.GREEN,
        "MEDIUM": Fore.YELLOW,
        "HIGH": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT
    }
    return colors.get(risk_level, Fore.WHITE)

# ═══════════════════════════════════════════════════════════════════════════
# TEST FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def test_root_endpoint() -> bool:
    """Test root endpoint (GET /)."""
    print_header("TEST 1: Root Endpoint")
    
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Root endpoint accessible")
            print(f"  API Name: {data.get('name')}")
            print(f"  Version: {data.get('version')}")
            print(f"  Status: {data.get('status')}")
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        return False

def test_health_endpoint() -> bool:
    """Test health check endpoint (GET /health)."""
    print_header("TEST 2: Health Check")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Health check passed")
            print(f"  Status: {data.get('status')}")
            print(f"  Models Loaded: {data.get('models_loaded')}")
            print(f"  Model Version: {data.get('model_version')}")
            return data.get('models_loaded', False)
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        return False

def test_model_info() -> bool:
    """Test model info endpoint (GET /models/info)."""
    print_header("TEST 3: Model Information")
    
    try:
        response = requests.get(f"{API_BASE_URL}/models/info", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Model info retrieved")
            print(f"  Model Version: {data.get('model_version')}")
            print(f"  Base Models: {', '.join(data.get('base_models', []))}")
            print(f"  Meta Model: {data.get('meta_model')}")
            print(f"  Feature Count: {data.get('feature_count')}")
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        return False

def test_prediction(name: str, transaction: Dict) -> bool:
    """Test fraud prediction endpoint."""
    print_header(f"TEST 4.{name.upper()}: Prediction ({name.replace('_', ' ').title()})")
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=transaction,
            timeout=TIMEOUT
        )
        
        response_time = (time.time() - start_time) * 1000  # ms
        
        if response.status_code == 200:
            data = response.json()
            
            fraud_prob = data.get('fraud_probability', 0)
            risk_level = data.get('risk_level', 'UNKNOWN')
            contributions = data.get('base_model_contributions', {})
            inference_time = data.get('inference_time_ms', 0)
            
            # Print results
            print_success(f"Prediction successful")
            print(f"  Fraud Probability: {fraud_prob*100:.2f}%")
            print(get_risk_color(risk_level) + f"  Risk Level: {risk_level}")
            print(f"  Response Time: {response_time:.2f}ms")
            print(f"  Inference Time: {inference_time:.2f}ms")
            
            # Print contributions
            print(f"\n  Base Model Contributions:")
            for model, contrib in contributions.items():
                bar_length = int(contrib * 50)  # 50 chars max
                bar = "█" * bar_length
                print(f"    {model:20s}: {bar} {contrib*100:.2f}%")
            
            # Performance check
            if response_time < 50:
                print_success(f"Performance: EXCELLENT (< 50ms)")
            elif response_time < 100:
                print_info(f"Performance: GOOD (< 100ms)")
            else:
                print_error(f"Performance: NEEDS IMPROVEMENT (> 100ms)")
            
            return True
            
        else:
            print_error(f"Prediction failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        return False

def test_batch_prediction() -> bool:
    """Test batch prediction endpoint."""
    print_header("TEST 5: Batch Prediction")
    
    try:
        transactions = [
            TEST_TRANSACTIONS["low_risk"],
            TEST_TRANSACTIONS["medium_risk"],
            TEST_TRANSACTIONS["high_risk"]
        ]
        
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=transactions,
            timeout=TIMEOUT
        )
        
        response_time = (time.time() - start_time) * 1000  # ms
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            count = data.get('count', 0)
            
            print_success(f"Batch prediction successful")
            print(f"  Transactions: {count}")
            print(f"  Response Time: {response_time:.2f}ms")
            print(f"  Avg Time per Transaction: {response_time/count:.2f}ms")
            
            # Print summary
            print(f"\n  Results:")
            for i, pred in enumerate(predictions, 1):
                risk_level = pred.get('risk_level', 'UNKNOWN')
                fraud_prob = pred.get('fraud_probability', 0)
                color = get_risk_color(risk_level)
                print(f"    {i}. {color}{risk_level:8s} - {fraud_prob*100:.2f}%")
            
            return True
            
        else:
            print_error(f"Batch prediction failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        return False

def test_error_handling() -> bool:
    """Test API error handling with invalid input."""
    print_header("TEST 6: Error Handling")
    
    try:
        # Test with missing fields
        invalid_transaction = {
            "Time": 123.45,
            "Amount": 50.00
            # Missing V1-V28
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=invalid_transaction,
            timeout=TIMEOUT
        )
        
        if response.status_code == 422:  # Validation error
            print_success("API correctly rejected invalid input (422 Unprocessable Entity)")
            return True
        else:
            print_error(f"Expected 422, got {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        return False

def test_performance(iterations: int = 100) -> bool:
    """Test API performance with multiple requests."""
    print_header(f"TEST 7: Performance Test ({iterations} requests)")
    
    try:
        transaction = TEST_TRANSACTIONS["low_risk"]
        
        times = []
        print_info(f"Running {iterations} predictions...")
        
        for i in range(iterations):
            start_time = time.time()
            
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=transaction,
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                times.append((time.time() - start_time) * 1000)
            else:
                print_error(f"Request {i+1} failed")
                return False
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{iterations}")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        p50 = sorted(times)[len(times) // 2]
        p95 = sorted(times)[int(len(times) * 0.95)]
        p99 = sorted(times)[int(len(times) * 0.99)]
        
        print_success("Performance test completed")
        print(f"\n  Statistics:")
        print(f"    Total Requests: {iterations}")
        print(f"    Average: {avg_time:.2f}ms")
        print(f"    Min: {min_time:.2f}ms")
        print(f"    Max: {max_time:.2f}ms")
        print(f"    P50 (Median): {p50:.2f}ms")
        print(f"    P95: {p95:.2f}ms")
        print(f"    P99: {p99:.2f}ms")
        
        # Performance rating
        if avg_time < 20:
            print_success(f"  Rating: EXCELLENT (avg < 20ms)")
        elif avg_time < 50:
            print_success(f"  Rating: GOOD (avg < 50ms)")
        elif avg_time < 100:
            print_info(f"  Rating: ACCEPTABLE (avg < 100ms)")
        else:
            print_error(f"  Rating: NEEDS IMPROVEMENT (avg > 100ms)")
        
        return True
        
    except Exception as e:
        print_error(f"Performance test failed: {str(e)}")
        return False

# ═══════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_all_tests() -> None:
    """Run all tests and print summary."""
    print("\n")
    print(Fore.CYAN + Style.BRIGHT + "╔" + "═" * 78 + "╗")
    print(Fore.CYAN + Style.BRIGHT + "║" + " " * 20 + "FRAUD DETECTION API - TEST SUITE" + " " * 26 + "║")
    print(Fore.CYAN + Style.BRIGHT + "╚" + "═" * 78 + "╝")
    
    print_info(f"Testing API at: {API_BASE_URL}")
    print_info(f"Timeout: {TIMEOUT}s\n")
    
    # Run tests
    results = {
        "Root Endpoint": test_root_endpoint(),
        "Health Check": test_health_endpoint(),
        "Model Info": test_model_info(),
        "Prediction (Low Risk)": test_prediction("low_risk", TEST_TRANSACTIONS["low_risk"]),
        "Prediction (Medium Risk)": test_prediction("medium_risk", TEST_TRANSACTIONS["medium_risk"]),
        "Prediction (High Risk)": test_prediction("high_risk", TEST_TRANSACTIONS["high_risk"]),
        "Batch Prediction": test_batch_prediction(),
        "Error Handling": test_error_handling(),
        "Performance Test": test_performance(iterations=100)
    }
    
    # Print  summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        if result:
            print_success(f"{test_name:30s} PASSED")
        else:
            print_error(f"{test_name:30s} FAILED")
    
    print("\n" + "=" * 80)
    
    if passed == total:
        print(Fore.GREEN + Style.BRIGHT + f"✓ ALL TESTS PASSED ({passed}/{total})")
    else:
        print(Fore.RED + Style.BRIGHT + f"✗ SOME TESTS FAILED ({passed}/{total})")
    
    print("=" * 80 + "\n")

# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n" + Fore.YELLOW + "Tests interrupted by user")
        sys.exit(1)

