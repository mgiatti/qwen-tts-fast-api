import requests
import time
import sys

BASE_URL = "http://127.0.0.1:8008"

def wait_for_server():
    print("Waiting for server...")
    for _ in range(10):
        try:
            requests.get(BASE_URL)
            print("Server is up!")
            return True
        except requests.ConnectionError:
            time.sleep(1)
    return False

def test_open_vision():
    print("Testing Open Vision...")
    payload = {
        "text": "Hello world, this is a test of Open Vision.",
        "type": "open_vision",
        "language": "English"
    }
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    if response.status_code != 200:
        print(f"Failed to submit job: {response.text}")
        return
    
    data = response.json()
    job_id = data["job_id"]
    print(f"Job submitted: {job_id}")
    
    while True:
        status_res = requests.get(f"{BASE_URL}/status/{job_id}")
        status_data = status_res.json()
        status = status_data["status"]
        print(f"Status: {status}")
        
        if status in ["completed", "failed"]:
            print(f"Final Result: {status_data}")
            break
        time.sleep(2)

if __name__ == "__main__":
    if not wait_for_server():
        print("Server failed to start.")
        sys.exit(1)
    
    test_open_vision()
