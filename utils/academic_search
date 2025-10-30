"""
Academic Paper Search Module
Integrates with Semantic Scholar and arXiv APIs
"""

import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class AcademicSearcher:
    """Handles academic paper searches across multiple sources"""
    
    def __init__(self):
        self.semantic_scholar_base = "https://api.semanticscholar.org/graph/v1"
        self.arxiv_base = "http://export.arxiv.org/api/query"
        
    def search_semantic_scholar(
        self, 
        query: str, 
        limit: int = 10,
        year_from: Optional[int] = None
    ) -> List[Dict]:
        """
        Search Semantic Scholar for papers
        
        Args:
            query: Search query string
            limit: Maximum number of results
            year_from: Filter papers from this year onwards
            
        Returns:
            List of paper dictionaries
        """
        papers = []
        
        try:
            # Build search URL
            url = f"{self.semantic_scholar_base}/paper/search"
            
            params = {
                "query": query,
                "limit": limit,
                "fields": "paperId,title,abstract,authors,year,citationCount,venue,publicationDate,url,externalIds"
            }
            
            if year_from:
                params["year"] = f"{year_from}-"
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if "data" in data:
                for paper in data["data"]:
                    papers.append({
                        "title": paper.get("title", ""),
                        "authors": [a.get("name", "") for a in paper.get("authors", [])],
                        "abstract": paper.get("abstract", ""),
                        "year": paper.get("year"),
                        "citation_count": paper.get("citationCount", 0),
                        "venue": paper.get("venue", ""),
                        "publication_date": paper.get("publicationDate", ""),
                        "url": paper.get("url", ""),
                        "arxiv_id": paper.get("externalIds", {}).get("ArXiv"),
                        "doi": paper.get("externalIds", {}).get("DOI"),
                        "source": "Semantic Scholar"
                    })
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
        
        return papers
    
    def search_arxiv(
        self, 
        query: str, 
        max_results: int = 10,
        days_back: int = 30
    ) -> List[Dict]:
        """
        Search arXiv for papers
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            days_back: Look back this many days
            
        Returns:
            List of paper dictionaries
        """
        papers = []
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Build query with date filter
            search_query = f'{query} AND submittedDate:[{start_date.strftime("%Y%m%d")}0000 TO {end_date.strftime("%Y%m%d")}2359]'
            
            params = {
                "search_query": search_query,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }
            
            response = requests.get(self.arxiv_base, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            # Define namespaces
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            for entry in root.findall('atom:entry', ns):
                try:
                    # Extract arXiv ID from id URL
                    arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
                    
                    # Extract authors
                    authors = [
                        author.find('atom:name', ns).text 
                        for author in entry.findall('atom:author', ns)
                    ]
                    
                    # Extract categories
                    categories = [
                        cat.get('term') 
                        for cat in entry.findall('atom:category', ns)
                    ]
                    
                    paper = {
                        "title": entry.find('atom:title', ns).text.strip(),
                        "authors": authors,
                        "abstract": entry.find('atom:summary', ns).text.strip(),
                        "year": int(entry.find('atom:published', ns).text[:4]),
                        "citation_count": 0,  # arXiv doesn't provide this
                        "venue": f"arXiv {categories[0] if categories else ''}",
                        "publication_date": entry.find('atom:published', ns).text.split('T')[0],
                        "url": f"https://arxiv.org/abs/{arxiv_id}",
                        "arxiv_id": arxiv_id,
                        "doi": None,
                        "source": "arXiv",
                        "categories": categories
                    }
                    
                    papers.append(paper)
                    
                except Exception as e:
                    print(f"Error parsing arXiv entry: {e}")
                    continue
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"arXiv search error: {e}")
        
        return papers
    
    def search_all_sources(
        self, 
        query: str, 
        limit_per_source: int = 5,
        days_back: int = 30
    ) -> List[Dict]:
        """
        Search all available sources and combine results
        
        Args:
            query: Search query string
            limit_per_source: Max results per source
            days_back: Look back this many days
            
        Returns:
            Combined and sorted list of papers
        """
        all_papers = []
        
        # Calculate year threshold
        year_from = (datetime.now() - timedelta(days=days_back)).year
        
        # Search Semantic Scholar
        ss_papers = self.search_semantic_scholar(query, limit_per_source, year_from)
        all_papers.extend(ss_papers)
        
        # Search arXiv
        arxiv_papers = self.search_arxiv(query, limit_per_source, days_back)
        all_papers.extend(arxiv_papers)
        
        # Remove duplicates based on title similarity
        unique_papers = []
        seen_titles = set()
        
        for paper in all_papers:
            title_key = paper["title"].lower().strip()[:50]  # First 50 chars
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        # Sort by citation count (highest first), then by date
        unique_papers.sort(
            key=lambda x: (
                -x.get("citation_count", 0),
                x.get("publication_date", "")
            ),
            reverse=False
        )
        
        return unique_papers


def search_academic_papers_by_topics(
    topics: Dict[str, str],
    papers_per_topic: int = 5,
    days_back: int = 30
) -> Dict[str, List[Dict]]:
    """
    Search for academic papers across multiple topics
    
    Args:
        topics: Dictionary of {topic_name: search_query}
        papers_per_topic: Number of papers to retrieve per topic
        days_back: How many days back to search
        
    Returns:
        Dictionary of {topic_name: [papers]}
    """
    searcher = AcademicSearcher()
    results = {}
    
    for topic_name, search_query in topics.items():
        print(f"Searching papers for: {topic_name}")
        papers = searcher.search_all_sources(
            query=search_query,
            limit_per_source=papers_per_topic,
            days_back=days_back
        )
        results[topic_name] = papers[:papers_per_topic]  # Limit final results
        
        # Brief pause between topics
        time.sleep(1)
    
    return results


# Predefined academic search topics
ACADEMIC_TOPICS = {
    "AI in Finance": "artificial intelligence machine learning finance financial services",
    "Data Management": "data management governance architecture metadata financial",
    "Financial Technology": "fintech blockchain digital payments financial technology",
    "Regulatory Technology": "regtech compliance regulation financial technology",
    "Machine Learning Applications": "machine learning deep learning neural networks finance"
}


if __name__ == "__main__":
    # Test the module
    print("Testing Academic Search Module...\n")
    
    searcher = AcademicSearcher()
    
    # Test Semantic Scholar
    print("Testing Semantic Scholar...")
    ss_results = searcher.search_semantic_scholar("AI in finance", limit=3)
    print(f"Found {len(ss_results)} papers from Semantic Scholar\n")
    
    # Test arXiv
    print("Testing arXiv...")
    arxiv_results = searcher.search_arxiv("machine learning finance", max_results=3)
    print(f"Found {len(arxiv_results)} papers from arXiv\n")
    
    # Test combined search
    print("Testing combined search...")
    all_results = searcher.search_all_sources("AI finance", limit_per_source=2)
    print(f"Found {len(all_results)} total papers\n")
    
    if all_results:
        print("Sample paper:")
        paper = all_results[0]
        print(f"Title: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'][:3])}")
        print(f"Year: {paper['year']}")
        print(f"Citations: {paper['citation_count']}")
        print(f"Source: {paper['source']}")