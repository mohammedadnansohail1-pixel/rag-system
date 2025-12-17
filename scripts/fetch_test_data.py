"""Fetch new test data for validation."""
import os
import requests
import time

DATA_DIR = "data/validation"
os.makedirs(DATA_DIR, exist_ok=True)

# 1. FINANCIAL - Apple 10-K (different company)
print("Fetching Apple 10-K...")
from src.loaders import SECLoader
loader = SECLoader(download_dir=DATA_DIR)
try:
    docs = loader.download("AAPL", "10-K", num_filings=1)
    print(f"  Downloaded: {len(docs)} Apple filings")
except Exception as e:
    print(f"  Error: {e}")

# 2. TECHNICAL - FastAPI docs
print("\nFetching FastAPI documentation...")
fastapi_urls = [
    "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "https://fastapi.tiangolo.com/tutorial/path-params/",
    "https://fastapi.tiangolo.com/tutorial/query-params/",
]
for url in fastapi_urls:
    try:
        resp = requests.get(url, timeout=10)
        filename = url.split("/")[-2] + ".html"
        with open(f"{DATA_DIR}/{filename}", "w") as f:
            f.write(resp.text)
        print(f"  Saved: {filename}")
        time.sleep(0.5)
    except Exception as e:
        print(f"  Error fetching {url}: {e}")

# 3. LEGAL - GitHub Terms of Service
print("\nFetching GitHub ToS...")
github_tos_url = "https://docs.github.com/en/site-policy/github-terms/github-terms-of-service"
try:
    resp = requests.get(github_tos_url, timeout=10)
    with open(f"{DATA_DIR}/github_tos.html", "w") as f:
        f.write(resp.text)
    print(f"  Saved: github_tos.html")
except Exception as e:
    print(f"  Error: {e}")

print("\nDone! Files in data/validation/:")
for f in os.listdir(DATA_DIR):
    size = os.path.getsize(f"{DATA_DIR}/{f}")
    print(f"  {f}: {size:,} bytes")
