import requests
from urllib.parse import urlparse
from config import GOOGLE_API_KEY, GOOGLE_CSE_ID, SERPER_API_KEY, BLACKLIST_DOMAINS

def domain_allowed(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    # Only block blacklisted domains
    if any(b in domain for b in BLACKLIST_DOMAINS):
        return False
    return True

def serper_search(query: str, num_results: int):
    urls = []
    page = 1

    while len(urls) < num_results:
        remaining = num_results - len(urls)
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "q": query,
                "num": min(10, remaining),
                "page": page,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("organic", [])

        for item in items:
            url = item.get("link")
            if not url:
                continue
            if domain_allowed(url):
                urls.append(url)
                if len(urls) >= num_results:
                    break

        if not items:
            break
        page += 1

    return urls

def google_search(query: str, num_results: int):
    if SERPER_API_KEY:
        return serper_search(query, num_results)

    urls = []
    start = 1

    while len(urls) < num_results:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_CSE_ID,
                "q": query,
                "start": start,
            },
            timeout=10
        )

        data = resp.json()
        items = data.get("items", [])

        for item in items:
            url = item["link"]
            if domain_allowed(url):
                urls.append(url)
                if len(urls) >= num_results:
                    break

        start += 10
        if not items:
            break

    return urls
