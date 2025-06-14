# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import requests
from bs4 import BeautifulSoup
from .base import ToolInterface


ENDPOINT = "***"


def extract_text_from(url, max_len: int = 2000) -> str:
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)[:max_len]


def search_hn(query: str, crawl_urls: bool) -> str:
    params = {
        "query": query,
        "tags": "story",
        "numericFilters": "points>100"
    }

    response = requests.get(ENDPOINT, params=params)

    hits = response.json()["hits"]

    result = ""
    for hit in hits[:5]:
        title = hit["title"]
        url = hit["url"]
        result += f"Title: {title}\n"
        
        if url is not None and crawl_urls:
            result += f"\tExcerpt: {extract_text_from(url)}\n"
        else:
            objectID = hit["objectID"]
            comments_url = f"{ENDPOINT}?tags=comment,story_{objectID}&hitsPerPage=1"
            comments_response = requests.get(comments_url)
            comment = comments_response.json()["hits"][0]['comment_text']
            
            result += f"\tComment: {comment}\n"
    return result


class HackerNewsSearchTool(ToolInterface):
    """Tool to get some insight from Hacker News users"""

    name: str = "hacker news search"  # Add type annotation
    description: str = ("Get insight from hacker news users to specific search terms. "
                        "Input should be a search term (e.g. How to get rich?). "
                        "The output will be the most recent stories related to it with a user comment.")  # Add type annotation
    crawl_urls: bool = False  # Add type annotation

    def use(self, input_text: str) -> str:
        return search_hn(input_text, self.crawl_urls)


if __name__ == '__main__':
    hn = HackerNewsSearchTool()
    res = hn.use('GPT-4')
    print(res)
