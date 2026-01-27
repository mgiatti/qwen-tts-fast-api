import requests
import sys

BASE_URL = "http://127.0.0.1:8008"

def check():
    print(f"Checking {BASE_URL}...")
    
    # Check Root
    try:
        resp = requests.get(f"{BASE_URL}/")
        print(f"GET / : Status {resp.status_code}")
        if resp.status_code == 200:
            print("Response:", resp.json())
        else:
            print("Response:", resp.text)
    except Exception as e:
        print(f"GET / Failed: {e}")
        return

    # Check Docs
    try:
        resp = requests.get(f"{BASE_URL}/docs")
        print(f"GET /docs : Status {resp.status_code}")
    except Exception as e:
        print(f"GET /docs Failed: {e}")

    # Check Generate (Method Not Allowed test)
    try:
        resp = requests.get(f"{BASE_URL}/generate")
        print(f"GET /generate (Expected 405) : Status {resp.status_code}")
    except:
        pass

    # Check Generate (valid)
    try:
        resp = requests.post(f"{BASE_URL}/generate", json={
            "text": "test",
            "type": "open_vision",
            "language": "English"
        })
        print(f"POST /generate : Status {resp.status_code}")
        if resp.status_code == 200:
            print("Response:", resp.json())
        else:
            print("Response:", resp.text)
    except Exception as e:
        print(f"POST /generate Failed: {e}")

if __name__ == "__main__":
    check()
