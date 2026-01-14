import asyncio
import json
import pandas as pd
from datetime import datetime
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
import re


class StockMarketScraper:
    """
    A comprehensive stock market news scraper using Crawl4AI
    Extracts: header, text, source, reactions/comments count, publication date
    """

    def __init__(self):
        self.crawler = None
        self.results = []

    async def __aenter__(self):
        """Async context manager entry"""
        browser_config = BrowserConfig(headless=True, verbose=False)
        crawler_strategy = AsyncPlaywrightCrawlerStrategy(browser_config=browser_config)
        self.crawler = AsyncWebCrawler(crawler_strategy=crawler_strategy)
        await self.crawler.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.crawler:
            await self.crawler.__aexit__(exc_type, exc_val, exc_tb)

    def clean_text(self, text):
        """Clean and normalize extracted text"""
        if not text:
            return ""
        # Remove extra whitespace and newlines
        text = re.sub(r"\s+", " ", str(text))
        text = text.strip()
        return text

    def extract_reactions_count(self, content):
        """Extract reaction/comment counts from content"""
        # Common patterns for reactions/comments
        patterns = [
            r"(\d+)\s*comments?",
            r"(\d+)\s*replies?",
            r"(\d+)\s*reactions?",
            r"(\d+)\s*likes?",
            r"Comments\s*\((\d+)\)",
            r"(\d+)\s*upvotes?",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return 0

    async def scrape_yahoo_finance(self, symbol="AAPL"):
        """Scrape Yahoo Finance news for a specific stock symbol"""
        url = f"https://finance.yahoo.com/quote/{symbol}/news"

        # Define extraction strategy for structured data
        extraction_strategy = LLMExtractionStrategy(
            llm_config=LLMConfig(
                provider="ollama/llama2",
                api_token="",
            ),
            instruction="""
            You are a data extraction assistant. Analyze the web page content and extract stock market news articles.

            For each article, extract and return ONLY the following fields in valid JSON format:
            1. headline: The main news title as plain text, no markdown or HTML.
            2. summary: Brief summary or first paragraph as plain text, no images or links.
            3. source: Publisher or website name.
            4. publish_date: Date the article was published.
            5. url: Full article URL if available.

            Return a JSON array of article objects ONLY, no additional text or explanation.
            """,
        )

        config = CrawlerRunConfig(
            extraction_strategy=extraction_strategy,
            js_code="""
            // Scroll to load more content
            window.scrollTo(0, document.body.scrollHeight);
            await new Promise(resolve => setTimeout(resolve, 2000));
            """,
        )

        try:
            result = await self.crawler.arun(url=url, config=config)

            # Parse basic content even if LLM extraction fails
            articles = self.parse_yahoo_finance_content(result.markdown, result.html)

            for article in articles:
                article["source"] = "Yahoo Finance"
                article["symbol"] = symbol
                article["scraped_at"] = datetime.now().isoformat()

            self.results.extend(articles)
            return articles

        except Exception as e:
            print(f"Error scraping Yahoo Finance for {symbol}: {e}")
            return []

    def parse_yahoo_finance_content(self, markdown_content, html_content):
        """Parse Yahoo Finance content when LLM extraction is not available"""
        articles = []

        # Split content by common news separators
        sections = markdown_content.split("\n\n")

        for i, section in enumerate(sections):
            if len(section.strip()) < 50:  # Skip short sections
                continue

            lines = section.strip().split("\n")
            if len(lines) < 2:
                continue

            # Extract headline (usually the first line or has title formatting)
            headline = ""
            content_text = ""

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if it looks like a headline (short, capitalized, etc.)
                if len(line) < 200 and (
                    line.isupper() or line.istitle() or line.startswith("#")
                ):
                    if not headline:
                        headline = line.replace("#", "").strip()
                else:
                    content_text += line + " "

            if headline and content_text:
                reactions_count = self.extract_reactions_count(content_text)

                article = {
                    "headline": self.clean_text(headline),
                    "text": self.clean_text(content_text[:500]),  # Limit to 500 chars
                    "source": "Yahoo Finance",
                    "reactions_count": reactions_count,
                    "publish_date": "Not specified",
                    "url": "N/A",
                }
                articles.append(article)

        return articles[:10]  # Return top 10 articles

    async def scrape_reddit_stocks(self, subreddit="stocks", limit=20):
        """Scrape Reddit stock discussions"""
        url = f"https://www.reddit.com/r/{subreddit}/hot/"

        config = CrawlerRunConfig(
            js_code="""
            // Scroll to load more posts
            for(let i = 0; i < 3; i++) {
                window.scrollTo(0, document.body.scrollHeight);
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
            """
        )

        try:
            result = await self.crawler.arun(url=url, config=config)
            articles = self.parse_reddit_content(result.markdown, subreddit)

            for article in articles:
                article["source"] = f"Reddit r/{subreddit}"
                article["scraped_at"] = datetime.now().isoformat()

            self.results.extend(articles[:limit])
            return articles[:limit]

        except Exception as e:
            print(f"Error scraping Reddit r/{subreddit}: {e}")
            return []

    def parse_reddit_content(self, markdown_content, subreddit):
        """Parse Reddit content to extract posts"""
        articles = []

        # Reddit posts often have specific patterns
        lines = markdown_content.split("\n")
        current_post = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for post titles (often bold or have specific formatting)
            if (line.startswith("**") and line.endswith("**")) or len(line) > 20:
                # If we have a current post, save it
                if current_post.get("headline"):
                    articles.append(current_post)

                # Start new post
                current_post = {
                    "headline": self.clean_text(line.replace("**", "")),
                    "text": "",
                    "source": f"Reddit r/{subreddit}",
                    "reactions_count": 0,
                    "publish_date": "Not specified",
                    "url": "N/A",
                }

            # Look for comments/upvotes counts
            elif re.search(r"\d+\s*(comments?|upvotes?|points?)", line, re.IGNORECASE):
                if current_post:
                    current_post["reactions_count"] = self.extract_reactions_count(line)

            # Add to post text
            elif current_post.get("headline") and len(line) > 10:
                current_post["text"] += line + " "

        # Don't forget the last post
        if current_post.get("headline"):
            articles.append(current_post)

        # Clean up text fields
        for article in articles:
            article["text"] = self.clean_text(article["text"][:300])  # Limit text

        return articles

    async def scrape_multiple_sources(self, symbols=None, subreddits=None):
        """Scrape multiple sources for comprehensive stock market data"""
        if symbols is None:
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        if subreddits is None:
            subreddits = ["stocks", "investing", "SecurityAnalysis"]

        print("Starting comprehensive stock market data scraping...")

        # Scrape Yahoo Finance for each symbol
        for symbol in symbols:
            print(f"Scraping Yahoo Finance for {symbol}...")
            await self.scrape_yahoo_finance(symbol)
            await asyncio.sleep(2)  # Be respectful with requests

        # Scrape Reddit subreddits
        for subreddit in subreddits:
            print(f"Scraping Reddit r/{subreddit}...")
            await self.scrape_reddit_stocks(subreddit)
            await asyncio.sleep(2)  # Be respectful with requests

        return self.results

    def save_to_csv(self, filename="stock_market_data.csv"):
        """Save scraped data to CSV file"""
        if not self.results:
            print("No data to save")
            return

        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False, encoding="utf-8")
        print(f"Saved {len(self.results)} articles to {filename}")
        return filename

    def get_summary(self):
        """Get summary of scraped data"""
        if not self.results:
            return "No data scraped yet"

        summary = {
            "total_articles": len(self.results),
            "sources": list(set([article["source"] for article in self.results])),
            "avg_reactions": sum(
                [article.get("reactions_count", 0) for article in self.results]
            )
            / len(self.results),
            "sample_headlines": [
                article["headline"][:100] for article in self.results[:5]
            ],
        }

        return summary


# Example usage function
async def main():
    """Example usage of the StockMarketScraper"""
    async with StockMarketScraper() as scraper:
        # Scrape data from multiple sources
        results = await scraper.scrape_multiple_sources(
            symbols=["AAPL", "GOOGL", "MSFT"], subreddits=["stocks", "investing"]
        )

        # Save to CSV
        filename = scraper.save_to_csv("my_stock_data.csv")

        # Print summary
        summary = scraper.get_summary()
        print("\n" + "=" * 50)
        print("SCRAPING SUMMARY")
        print("=" * 50)
        print(json.dumps(summary, indent=2))

        # Display first few results
        if results:
            print("\n" + "=" * 50)
            print("SAMPLE RESULTS")
            print("=" * 50)
            for i, article in enumerate(results[:3]):
                print(f"\n{i + 1}. {article['headline']}")
                print(f"   Source: {article['source']}")
                print(f"   Reactions: {article['reactions_count']}")
                print(f"   Text: {article['text'][:150]}...")


# Run the scraper
if __name__ == "__main__":
    asyncio.run(main())
