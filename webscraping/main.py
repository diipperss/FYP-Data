import asyncio
from webscraping.topics import TOPICS
from search import google_search
from crawler import crawl_and_chunk
from config import MAX_URLS_PER_SUBSUBTOPIC

async def main():
    for topic, subs in TOPICS.items():
        for sub, queries in subs.items():
            for query in queries:
                urls = google_search(query, MAX_URLS_PER_SUBSUBTOPIC)

                output_path = f"data/raw/{topic}/{query.replace(' ', '_')}"
                await crawl_and_chunk(urls, output_path)

asyncio.run(main())
