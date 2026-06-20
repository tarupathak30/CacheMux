import httpx
import json
import time

URL = "http://127.0.0.1:8000/v1/chat/completions"


def run_tests():
    # -------------------------------------------------------------
    # TEST 1: System Prompts & Extra Parameters (Routing Validation)
    # -------------------------------------------------------------
    print("\n=== TEST 1: SYSTEM PROMPT & EXTRA PARAMS ===")
    payload_params = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a pirate. Answer in one short sentence."},
            {"role": "user", "content": "Hello!"}
        ],
        "temperature": 0.2,
        "max_tokens": 50,
        "stream": False
    }
    
    try:
        res = httpx.post(URL, json=payload_params, timeout=30.0)
        print(f"Status Code: {res.status_code}")
        print(f"Content: {res.json()['choices'][0]['message']['content']}")
        print(f"Provider Header: {res.headers.get('X-Provider', 'None')}")
    except Exception as e:
        print(f"Test 1 Failed: {e}")

    # -------------------------------------------------------------
    # TEST 2: Active Streaming (Primary Provider)
    # -------------------------------------------------------------
    print("\n=== TEST 2: NON-CACHED LIVE STREAMING ===")
    # Using unique text to guarantee a Cache MISS and trigger fresh provider stream
    unique_timestamp = time.time()
    payload_stream = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": f"Count from 1 to 5 slowly. Timestamp: {unique_timestamp}"}
        ],
        "stream": True
    }

    full_stream_text = ""
    try:
        with httpx.Client() as client:
            with client.stream("POST", URL, json=payload_stream, timeout=45.0) as response:
                print(f"Status Code: {response.status_code}")
                print(f"Cache Header: {response.headers.get('X-Cache', 'MISS')}")
                
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_content = line[6:]
                        if data_content.strip() == "[DONE]":
                            break
                        
                        chunk_json = json.loads(data_content)
                        delta = chunk_json["choices"][0]["delta"].get("content", "")
                        full_stream_text += delta
                        print(delta, end="", flush=True)
                print() # Newline after stream finishes
    except Exception as e:
        print(f"\nTest 2 Failed: {e}")

    # -------------------------------------------------------------
    # TEST 3: Streaming from Cache (Emulated Latency Validation)
    # -------------------------------------------------------------
    print("\n=== TEST 3: STREAMING CACHE HIT ===")
    # Sending exact same payload as Test 2 to enforce a Cache HIT on Upstash Redis
    try:
        with httpx.Client() as client:
            with client.stream("POST", URL, json=payload_stream, timeout=15.0) as response:
                print(f"Status Code: {response.status_code}")
                print(f"Cache Header: {response.headers.get('X-Cache', 'None')}")
                
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_content = line[6:]
                        if data_content.strip() == "[DONE]":
                            break
                        chunk_json = json.loads(data_content)
                        print(chunk_json["choices"][0]["delta"].get("content", ""), end="", flush=True)
                print()
    except Exception as e:
        print(f"\nTest 3 Failed: {e}")

    # -------------------------------------------------------------
    # TEST 4: Failover / Error Fallback (Simulation)
    # -------------------------------------------------------------
    print("\n=== TEST 4: INVALID MODEL FALLBACK / ERROR HANDLING ===")
    payload_bad = {
        "model": "non-existent-broken-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False
    }
    try:
        res = httpx.post(URL, json=payload_bad, timeout=30.0)
        print(f"Status Code: {res.status_code}")
        print(f"Response data: {res.json()}")
    except Exception as e:
        print(f"Test 4 Exception: {e}")

if __name__ == "__main__":
    run_tests()
