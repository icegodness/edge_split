import requests
import time


def measure_latency(server_url):
    start_time = time.time()
    response = requests.get(server_url)
    end_time = time.time()

    latency = end_time - start_time
    return latency


if __name__ == "__main__":
    server_url = "http://192.168.31.174:4980"  # Replace with your server URL
    latency = measure_latency(server_url)
    print(f"Latency: {latency} seconds")
