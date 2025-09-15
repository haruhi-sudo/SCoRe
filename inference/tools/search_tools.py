import os
import re
import json
import sqlite3
import time
from typing import Any, Optional, Dict, List
import requests
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()
load_dotenv(".local_env", override=True)

# ======== 数据结构定义 ========
class SearchResult(BaseModel):
    """单条搜索结果"""
    id: str
    title: str
    url: str
    snippet: str
    source: str
    display_link: Optional[str] = None
    formatted_url: Optional[str] = None

class SearchResponse(BaseModel):
    """搜索响应"""
    success: bool
    query: str
    results: List[SearchResult]
    count: int
    search_time: float
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# ======== 搜索类 ========
class GoogleSearch:
    """Google搜索封装类"""
    
    def __init__(self, api_key: Optional[str] = None):
        """初始化搜索类
        
        Args:
            api_key: Google API密钥，如果不提供则从环境变量读取
        """
        self.api_key = api_key or os.environ.get("SEARCH_API_KEY")
        if not self.api_key:
            raise ValueError("缺少 SEARCH_API_KEY，请设置环境变量或传入参数")
        
        self.api_url = 'https://idealab.alibaba-inc.com/api/v1/search/search'
    
    def search(
        self,
        query: str,
        num_results: int = 5,
        language: str = "en-us",
        country: str = "us",
        safe_search: bool = True
    ) -> SearchResponse:
        """执行搜索
        
        Args:
            query: 搜索关键词
            num_results: 返回结果数量 (1-10)
            language: 搜索结果语言
            country: 搜索结果国家/地区
            safe_search: 是否启用安全搜索
        
        Returns:
            SearchResponse: 搜索结果响应对象
        """
        # 参数验证
        if not query or not query.strip():
            return SearchResponse(
                success=False,
                query=query,
                results=[],
                count=0,
                search_time=0,
                message="搜索关键词不能为空"
            )
        
        query = query.strip()
        num_results = max(1, min(num_results, 10))
        
        start_time = time.time()
        
        try:
            # 构建请求
            headers = {
                'X-AK': self.api_key,
                'Content-Type': 'application/json'
            }
            
            data = {
                'query': query,
                'num': num_results,
                'extendParams': {
                    'country': country,
                    'locale': language,
                },
                'platformInput': {
                    'model': 'google-search'
                }
            }
            
            # 发送请求
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            search_time = time.time() - start_time
            
            # 解析响应
            json_data = response.json()
            search_results = []
            
            if "data" in json_data and 'originalOutput' in json_data['data']:
                organic_results = json_data['data']['originalOutput'].get('organic', [])
                
                for i, item in enumerate(organic_results):
                    search_results.append(SearchResult(
                        id=f"google-{i}",
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source="google",
                        display_link=item.get("displayLink"),
                        formatted_url=item.get("formattedUrl")
                    ))
            
            return SearchResponse(
                success=True,
                query=query,
                results=search_results,
                count=len(search_results),
                search_time=search_time,
                metadata={
                    "language": language,
                    "country": country,
                    "safe_search": safe_search,
                    "search_engine": "google"
                }
            )
            
        except requests.exceptions.Timeout:
            return SearchResponse(
                success=False,
                query=query,
                results=[],
                count=0,
                search_time=time.time() - start_time,
                message="搜索请求超时"
            )
        except requests.exceptions.RequestException as e:
            return SearchResponse(
                success=False,
                query=query,
                results=[],
                count=0,
                search_time=time.time() - start_time,
                message=f"搜索API错误: {str(e)}"
            )
        except Exception as e:
            return SearchResponse(
                success=False,
                query=query,
                results=[],
                count=0,
                search_time=time.time() - start_time,
                message=f"搜索失败: {str(e)}"
            )
    
    def search_simple(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """简单搜索接口，返回字典列表
        
        Args:
            query: 搜索关键词
            num_results: 返回结果数量
        
        Returns:
            搜索结果列表，每个结果是包含title, url, snippet的字典
        """
        response = self.search(query, num_results)
        
        if response.success:
            return [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet
                }
                for r in response.results
            ]
        else:
            return []

# ======== 便捷函数 ========
def quick_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """快速搜索函数
    
    Args:
        query: 搜索关键词
        num_results: 返回结果数量
    
    Returns:
        搜索结果列表
    """
    try:
        searcher = GoogleSearch()
        return searcher.search_simple(query, num_results)
    except Exception as e:
        return []


class SearchTools:
    # Class variable for cache file path
    _cache_file = None
    # Flag to track if database has been initialized
    _db_initialized = False

    @classmethod
    def set_cache_file(cls, cache_file_path):
        """Set the cache file path"""
        cls._cache_file = cache_file_path
    @staticmethod
    def _init_cache_db():
        """Initialize the cache database only once"""
        if SearchTools._db_initialized:
            return
            
        with sqlite3.connect(SearchTools._cache_file) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS search_cache (
                    query TEXT PRIMARY KEY,
                    results TEXT,
                    timestamp REAL
                )
                """
            )
            conn.commit()
        SearchTools._db_initialized = True
    
    @staticmethod
    def _get_cached_results(query: str) -> Optional[list]:
        """Retrieve cached search results"""
        try:
            with sqlite3.connect(SearchTools._cache_file) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT results FROM search_cache WHERE query = ?",
                    (query,)
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
            return None
        except Exception:
            return None
    
    @staticmethod
    def _cache_results(query: str, results: list):
        """Cache search results"""
        try:
            with sqlite3.connect(SearchTools._cache_file) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO search_cache (query, results, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    (query, json.dumps(results), time.time())
                )
                conn.commit()
        except Exception:
            pass  # Silently fail caching if issues occur
    
    @staticmethod
    def web_search(query):
        """web search with cache support"""
        # Initialize cache database only once
        SearchTools._init_cache_db()

        query = query.strip().lower()
        query = re.sub(r"[^\w\s]", "", query)
        
        # Check cache first
        search_dicts = SearchTools._get_cached_results(query)
        retrieved_content = ""
        if search_dicts is not None:
            retrieved_content = "hit cache."
        else:
            # Perform search if not in cache
            search_dicts = quick_search(query, num_results=10)

        
        if len(search_dicts) == 0:
            retrieved_content = "Failed to retrieve content. Please try again later."
        else:
            # Add result to cache
            SearchTools._cache_results(query, search_dicts)
            for idx, search_res in enumerate(search_dicts):
                retrieved_content += f"Page: {idx}\nTitle: {search_res['title']}\nSnippet: {search_res['snippet']}\n"
        
        return retrieved_content.strip()


if __name__ == '__main__':
    print(quick_search('who is the president of usa?'))
