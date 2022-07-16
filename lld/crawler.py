# lld/crawler.py

import asyncio
from urllib.parse import urlparse

import aiohttp
from extraction import Extractor
from tqdm.asyncio import tqdm, tqdm_asyncio


def get_domain(url: str):
    parsed = urlparse(url)
    return parsed.netloc


async def gather_with_concurrency(n: int, tasks):
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task):
        async with semaphore:
            return await task

    results = asyncio.as_completed(tasks)
    return results


async def scrape_description(
    url: str, session: aiohttp.ClientSession, timeout: int = 15
) -> str:
    """Scrape description of the given url from <head> and <meta> tags."""

    extractor = Extractor()

    domain = get_domain(url)

    try:
        async with session.get(url, timeout=timeout) as response:
            html = await response.text()
            data = extractor.extract(html, source_url=url)

            collected = [
                *data.titles,
                data.description,
                domain,
            ]

            results = "\n".join(filter(bool, collected))
            return results
    except:
        return domain


async def crawl(urls: list[str], batch_size: int = 100):

    connector = aiohttp.TCPConnector(limit=None, ttl_dns_cache=300)

    async with aiohttp.ClientSession(connector=connector) as session:
        gen_results = await gather_with_concurrency(
            batch_size, [scrape_description(url, session) for url in urls]
        )

        for result in gen_results:
            yield await result


async def run(urls: list[str], batch_size: int = 100):

    crawler = crawl(urls, batch_size)
    results = [c async for c in tqdm_asyncio(crawler, total=len(urls))]

    return results
