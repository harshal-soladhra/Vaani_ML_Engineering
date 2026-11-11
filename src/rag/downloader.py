import os
import sys
import httpx
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import PAPERS, PAPERS_DIR

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 (Research paper client for educational analysis)"
}

def download_paper(paper_key: str, info: dict) -> Path:
    target_path = PAPERS_DIR / info["filename"]
    if target_path.exists() and target_path.stat().st_size > 10000:
        print(f"Paper {info['title']} already exists at {target_path} ({target_path.stat().st_size} bytes).")
        return target_path

    urls_to_try = [info["url"], info.get("fallback_url")]
    if "alt_url" in info:
        urls_to_try.append(info["alt_url"])

    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=60.0) as client:
        for url in urls_to_try:
            if not url:
                continue
            print(f"Attempting download for {paper_key} from {url}...")
            try:
                resp = client.get(url)
                if resp.status_code == 200 and resp.content.startswith(b"%PDF"):
                    with open(target_path, "wb") as f:
                        f.write(resp.content)
                    print(f"Successfully downloaded {info['title']} to {target_path} ({len(resp.content)} bytes)")
                    return target_path
                else:
                    print(f"Failed {url}: status {resp.status_code}, header {resp.headers.get('content-type')}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")

    raise RuntimeError(f"Could not download paper {paper_key} from any attempted URLs.")

def download_all_papers():
    downloaded = {}
    for key, info in PAPERS.items():
        downloaded[key] = download_paper(key, info)
    return downloaded

if __name__ == "__main__":
    download_all_papers()
