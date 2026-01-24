import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not SERPER_API_KEY and (not GOOGLE_API_KEY or not GOOGLE_CSE_ID):
    raise RuntimeError("Missing SERPER_API_KEY or GOOGLE_API_KEY/GOOGLE_CSE_ID in .env")

MAX_URLS_PER_SUBSUBTOPIC = 5

# Domains to exclude
BLACKLIST_DOMAINS = {
    "reddit.com",
    "quora.com",
    "medium.com",
    "facebook.com",
    "twitter.com",
    "x.com",
}

# Domains to prioritize
WHITELIST_DOMAINS = {
    "nasdaq.com",
    "investopedia.com",
    "ig.com",
    "sgx.com",
    "cmegroup.com",
    "corporatefinanceinstitute.com",
    "sec.gov"
}

