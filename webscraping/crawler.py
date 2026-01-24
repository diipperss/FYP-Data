import os
import re
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

async def crawl_and_chunk(urls, output_path):
    os.makedirs(output_path, exist_ok=True)

    # 1. Setup global browser settings
    browser_config = BrowserConfig(
        headless=True,
        verbose=True,
        # This helps mimic a real browser
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    )
    pruning_filter = PruningContentFilter(
        threshold=0.45,           # Lower values are more 'aggressive' at pruning
        min_word_threshold=50,    # Ignore blocks with fewer than 50 words
        threshold_type="fixed"
    )
    
    md_generator = DefaultMarkdownGenerator(
        content_filter=pruning_filter
    )

    # 2. Setup per-run settings
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,        # Ensure fresh data
        markdown_generator=md_generator,
        excluded_tags=["nav", "footer", "header", "aside", "form", "button", "ul", "li", "script", "style"],
        word_count_threshold=30,             # Filters out junk/error pages
        process_iframes=True
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        chunks = []
        for url in urls:
            print(f"Crawl starting: {url}")
            result = await crawler.arun(url=url, config=run_config)
            print(f"Result = {result}")

            if result.success and result.markdown:
                clean_content = getattr(result, 'fit_markdown', result.markdown)
               
                # Remove dictionary junk and common boilerplate
                clean_content = re.sub(r'([A-Z] \| ){5,}', '', clean_content)
                clean_content = re.sub(r"(Table of Contents|Read more|Partner Links|More Videos).*?\n", "", clean_content, flags=re.DOTALL)

                # Remove lines with too many URLs
                clean_content = remove_lines_with_too_many_urls(clean_content, max_urls=1)

                # Remove clickbait/ads
                clean_content = remove_clickbait_lines(clean_content)

                lower_content = clean_content.lower()

                #quality gate
                quality_signals = ["definition", "what is", "explained", "formula", "example","step","overview"]
                if any(k in lower_content for k in quality_signals):
                    # Filter out the 'AMLP' style ticker tables
                    if lower_content.count("ticker") > 5 and lower_content.count("%") > 10:
                        print(f"Skipping {url} - Likely a data table.")
                        continue

                    chunks.append(f"SOURCE: {url}\n{clean_content}\n\n{'='*50}\n\n")
                    print(f"Successfully scraped {len(clean_content)} chars from {url}")
                else:
                    print(f"Skipping {url} - Likely a data table or irrelevant page.")
            else:
                print(f"Failed to scrape {url}: {result.error_message}")

        if chunks:
            with open(os.path.join(output_path, "chunks.txt"), "w", encoding='utf-8') as f:
                f.writelines(chunks)



# Helper functions
def remove_lines_with_too_many_urls(text, max_urls=1):
    clean_lines = []
    for line in text.splitlines():
        url_count = len(re.findall(r'https?://\S+', line))
        if url_count <= max_urls:
            clean_lines.append(line)
    return "\n".join(clean_lines)

def remove_clickbait_lines(text):
    blacklist_phrases = [
        "Sponsored", "Advisors:", "click here", "Partner Links", "Advertisement"
    ]
    clean_lines = [
        line for line in text.splitlines()
        if not any(phrase.lower() in line.lower() for phrase in blacklist_phrases)
    ]
    return "\n".join(clean_lines)