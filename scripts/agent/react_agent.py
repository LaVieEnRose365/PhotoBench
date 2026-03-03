"""
ReAct Agent for Complex Image Retrieval using LangGraph.
Supports FaceInfo, Metadata, and Embedding search tools.
Supports concurrent tool execution and batch query processing.
"""

import asyncio
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import Annotated, Any, Callable, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Optional imports for Claude and Gemini native APIs
try:
    from langchain_anthropic import ChatAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "embeddings"))
from image_search import ImageSearchEngine


# =============================================================================
# Data Loaders
# =============================================================================

@dataclass
class FaceInfoDB:
    """Database for face information lookup."""
    face_id_to_nicknames: Dict[str, List[str]] = field(default_factory=dict)
    image_to_face_ids: Dict[str, List[str]] = field(default_factory=dict)
    nickname_to_face_ids: Dict[str, List[str]] = field(default_factory=dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> "FaceInfoDB":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        db = cls(
            face_id_to_nicknames=data.get("face_id_to_nicknames", {}),
            image_to_face_ids=data.get("image_to_face_ids", {}),
        )
        
        # Build reverse mapping: nickname -> face_ids
        for face_id, nicknames in db.face_id_to_nicknames.items():
            for nickname in nicknames:
                nickname_lower = nickname.lower()
                if nickname_lower not in db.nickname_to_face_ids:
                    db.nickname_to_face_ids[nickname_lower] = []
                db.nickname_to_face_ids[nickname_lower].append(face_id)
        
        return db
    
    def find_face_ids_by_name(self, name: str) -> List[str]:
        """Find face IDs that match a name/nickname."""
        name_lower = name.lower()
        matched_face_ids = set()
        
        # Exact match first
        if name_lower in self.nickname_to_face_ids:
            matched_face_ids.update(self.nickname_to_face_ids[name_lower])
        
        # Partial match
        for nickname, face_ids in self.nickname_to_face_ids.items():
            if name_lower in nickname or nickname in name_lower:
                matched_face_ids.update(face_ids)
        
        return list(matched_face_ids)
    
    def find_images_with_face_ids(self, face_ids: List[str]) -> List[str]:
        """Find images containing any of the given face IDs."""
        face_ids_set = set(face_ids)
        matched_images = []
        
        for image, img_face_ids in self.image_to_face_ids.items():
            if face_ids_set.intersection(set(img_face_ids)):
                matched_images.append(image)
        
        return matched_images
    
    def find_images_with_all_face_ids(self, face_ids: List[str]) -> List[str]:
        """Find images containing ALL of the given face IDs."""
        face_ids_set = set(face_ids)
        matched_images = []
        
        for image, img_face_ids in self.image_to_face_ids.items():
            if face_ids_set.issubset(set(img_face_ids)):
                matched_images.append(image)
        
        return matched_images
    
    def get_all_nicknames(self) -> Dict[str, List[str]]:
        """Get all face_id -> nicknames mapping."""
        return self.face_id_to_nicknames


@dataclass
class MetadataDB:
    """Database for image metadata (time, location) lookup."""
    metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> "MetadataDB":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(metadata=data)
    
    def filter_by_time_range(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[str]:
        """Filter images by time range. Time format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS."""
        matched_images = []
        
        start_dt = self._parse_time(start_time) if start_time else None
        end_dt = self._parse_time(end_time) if end_time else None
        
        for image, meta in self.metadata.items():
            img_time = meta.get("time", "")
            if not img_time:
                continue
            
            img_dt = self._parse_time(img_time)
            if img_dt is None:
                continue
            
            if start_dt and img_dt < start_dt:
                continue
            if end_dt and img_dt > end_dt:
                continue
            
            matched_images.append(image)
        
        return matched_images
    
    def filter_by_year(self, year: int) -> List[str]:
        """Filter images by year."""
        matched_images = []
        
        for image, meta in self.metadata.items():
            img_time = meta.get("time", "")
            if not img_time:
                continue
            
            img_dt = self._parse_time(img_time)
            if img_dt and img_dt.year == year:
                matched_images.append(image)
        
        return matched_images
    
    def filter_by_month(self, year: int, month: int) -> List[str]:
        """Filter images by year and month."""
        matched_images = []
        
        for image, meta in self.metadata.items():
            img_time = meta.get("time", "")
            if not img_time:
                continue
            
            img_dt = self._parse_time(img_time)
            if img_dt and img_dt.year == year and img_dt.month == month:
                matched_images.append(image)
        
        return matched_images
    
    def filter_by_location(self, location_keywords: List[str]) -> List[str]:
        """Filter images by location keywords (fuzzy match)."""
        matched_images = []
        
        for image, meta in self.metadata.items():
            location = meta.get("location", "")
            if not location:
                continue
            
            # Check if any keyword is in the location
            if any(kw.lower() in location.lower() for kw in location_keywords):
                matched_images.append(image)
        
        return matched_images
    
    def get_image_metadata(self, image: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific image."""
        return self.metadata.get(image)
    
    def get_all_locations(self) -> List[str]:
        """Get all unique locations."""
        locations = set()
        for meta in self.metadata.values():
            loc = meta.get("location", "")
            if loc:
                locations.add(loc)
        return list(locations)
    
    @staticmethod
    def _parse_time(time_str: str) -> Optional[datetime]:
        """Parse time string to datetime."""
        if not time_str:
            return None
        
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y年%m月%d日",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        return None


# =============================================================================
# Global Instances (will be initialized)
# =============================================================================

_face_db: Optional[FaceInfoDB] = None
_metadata_db: Optional[MetadataDB] = None
_embedding_engine: Optional[ImageSearchEngine] = None


def init_databases(
    face_info_path: str,
    metadata_path: str,
    model_name: str,
    index_dir: str,
    device: str = "cuda",
):
    """Initialize all databases and search engine."""
    global _face_db, _metadata_db, _embedding_engine
    
    print(f"Loading FaceInfo from {face_info_path}...")
    _face_db = FaceInfoDB.from_json(face_info_path)
    
    print(f"Loading Metadata from {metadata_path}...")
    _metadata_db = MetadataDB.from_json(metadata_path)
    
    print(f"Loading Embedding engine: {model_name} from {index_dir}...")
    _embedding_engine = ImageSearchEngine.from_pretrained(
        model_name=model_name,
        index_dir=index_dir,
        device=device,
    )
    _embedding_engine.load()
    
    print("All databases initialized!")


# Target number of results
TARGET_RESULTS = 50


def fill_results_with_embedding(
    results: List[str],
    query_text: str,
    target_count: int = TARGET_RESULTS,
) -> List[str]:
    """
    Fill results up to target_count using embedding search.
    
    Args:
        results: Current result list
        query_text: Original query text for embedding search
        target_count: Target number of results (default: 50)
    
    Returns:
        List of exactly target_count image filenames.
    """
    if _embedding_engine is None:
        return results[:target_count]
    
    if len(results) >= target_count:
        return results[:target_count]
    
    # Get embedding results
    existing_set = set(results)
    needed = target_count - len(results)
    
    # Search with enough buffer to account for duplicates
    embedding_results = _embedding_engine.retrieve(
        text=query_text,
        top_k=target_count + len(results),
        prompt_suffix="Represent the query for image retrieval.",
    )
    
    # Add non-duplicate results
    filled_results = list(results)
    for r in embedding_results:
        if len(filled_results) >= target_count:
            break
        filename = r["filename"]
        if filename not in existing_set:
            filled_results.append(filename)
            existing_set.add(filename)
    
    return filled_results[:target_count]


# =============================================================================
# Tools Definition
# =============================================================================

@tool
def search_by_face(
    names: List[str],
    require_all: bool = False,
) -> str:
    """
    Search for images containing specific people by their names or nicknames.
    
    Args:
        names: List of person names/nicknames to search for (e.g., ["妈妈", "爸爸"], ["小明"])
        require_all: If True, only return images containing ALL specified people.
                     If False, return images containing ANY of the specified people.
    
    Returns:
        JSON string with matching images and matched face info.
    """
    if _face_db is None:
        return json.dumps({"error": "Face database not initialized"})
    
    all_face_ids = []
    name_to_face_ids = {}
    
    for name in names:
        face_ids = _face_db.find_face_ids_by_name(name)
        if face_ids:
            all_face_ids.extend(face_ids)
            name_to_face_ids[name] = face_ids
    
    if not all_face_ids:
        return json.dumps({
            "found": False,
            "message": f"No person found matching names: {names}",
            "available_nicknames": _face_db.get_all_nicknames(),
        })
    
    if require_all:
        # Need at least one face_id from each name
        images_sets = []
        for name, face_ids in name_to_face_ids.items():
            imgs = set()
            for fid in face_ids:
                imgs.update(_face_db.find_images_with_face_ids([fid]))
            if imgs:
                images_sets.append(imgs)
        
        if images_sets:
            matched_images = list(set.intersection(*images_sets))
        else:
            matched_images = []
    else:
        matched_images = _face_db.find_images_with_face_ids(list(set(all_face_ids)))
    
    return json.dumps({
        "found": True,
        "matched_names": name_to_face_ids,
        "images": matched_images,
        "count": len(matched_images),
    }, ensure_ascii=False)


@tool
def search_by_metadata(
    location_keywords: Optional[List[str]] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Search for images by time and/or location metadata.
    
    Args:
        location_keywords: Keywords to match in location (e.g., ["北京", "故宫"])
        year: Filter by year (e.g., 2023)
        month: Filter by month (1-12), must be used with year
        start_date: Start date for range filter (format: YYYY-MM-DD)
        end_date: End date for range filter (format: YYYY-MM-DD)
    
    Returns:
        JSON string with matching images.
    """
    if _metadata_db is None:
        return json.dumps({"error": "Metadata database not initialized"})
    
    # Start with all images
    all_images = set(_metadata_db.metadata.keys())
    
    # Apply location filter
    if location_keywords:
        location_images = set(_metadata_db.filter_by_location(location_keywords))
        all_images = all_images.intersection(location_images)
    
    # Apply time filters
    if start_date or end_date:
        time_images = set(_metadata_db.filter_by_time_range(start_date, end_date))
        all_images = all_images.intersection(time_images)
    elif year and month:
        time_images = set(_metadata_db.filter_by_month(year, month))
        all_images = all_images.intersection(time_images)
    elif year:
        time_images = set(_metadata_db.filter_by_year(year))
        all_images = all_images.intersection(time_images)
    
    matched_images = list(all_images)
    
    return json.dumps({
        "filters_applied": {
            "location_keywords": location_keywords,
            "year": year,
            "month": month,
            "start_date": start_date,
            "end_date": end_date,
        },
        "images": matched_images,
        "count": len(matched_images),
    }, ensure_ascii=False)


@tool
def search_by_embedding(
    query: str,
    top_k: int = 30,
) -> str:
    """
    Search for images using semantic embedding similarity.
    Use this for content-based queries like "sunset on beach", "birthday party", etc.
    
    Args:
        query: Natural language description of the image content
        top_k: Maximum number of results to return (default: 30)
    
    Returns:
        JSON string with ranked images and their similarity scores.
    """
    if _embedding_engine is None:
        return json.dumps({"error": "Embedding engine not initialized"})
    
    results = _embedding_engine.retrieve(
        text=query,
        top_k=top_k,
        prompt_suffix="Represent the query for image retrieval.",
    )
    
    return json.dumps({
        "query": query,
        "results": results,
        "count": len(results),
    }, ensure_ascii=False)


@tool
def get_available_people() -> str:
    """
    Get a list of all known people and their nicknames in the photo album.
    Use this to understand what person names can be searched.
    
    Returns:
        JSON string with face_id to nicknames mapping.
    """
    if _face_db is None:
        return json.dumps({"error": "Face database not initialized"})
    
    return json.dumps({
        "people": _face_db.get_all_nicknames(),
    }, ensure_ascii=False)


@tool
def intersect_results(
    image_lists: List[List[str]],
) -> str:
    """
    Intersect multiple image lists to find images that appear in ALL lists.
    Use this to combine results from different tools.
    
    Args:
        image_lists: List of image filename lists to intersect
    
    Returns:
        JSON string with intersected images.
    """
    if not image_lists:
        return json.dumps({"images": [], "count": 0})
    
    # Filter out None values and empty lists
    valid_lists = [lst for lst in image_lists if lst is not None and isinstance(lst, list)]
    
    if not valid_lists:
        return json.dumps({"images": [], "count": 0})
    
    result = set(valid_lists[0])
    for img_list in valid_lists[1:]:
        result = result.intersection(set(img_list))
    
    return json.dumps({
        "images": list(result),
        "count": len(result),
    }, ensure_ascii=False)


# =============================================================================
# Agent State and Graph
# =============================================================================

class AgentState(TypedDict):
    """State of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    final_results: Optional[List[str]]


# All tools
ALL_TOOLS = [
    search_by_face,
    search_by_metadata,
    search_by_embedding,
    get_available_people,
    intersect_results,
]

# Tool mode configurations
TOOL_MODES = {
    "V": [search_by_embedding],  # Vision only
    "VM": [search_by_embedding, search_by_metadata, intersect_results],  # Vision + Metadata
    "VF": [search_by_embedding, search_by_face, get_available_people],  # Vision + Face
    "ALL": ALL_TOOLS,  # All tools (default)
}


def get_tools_for_mode(tool_mode: str = "ALL") -> List:
    """Get the list of tools for the specified mode."""
    mode = tool_mode.upper()
    if mode not in TOOL_MODES:
        raise ValueError(f"Unknown tool_mode: {tool_mode}. Available modes: {list(TOOL_MODES.keys())}")
    return TOOL_MODES[mode]


def create_react_agent(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    tool_mode: str = "ALL",
):
    """Create a ReAct agent for image retrieval.
    
    Supports multiple LLM backends:
    - OpenAI/OpenAI-compatible APIs (default): gpt-*, deepseek-*, qwen*, o3-*, etc.
    - Claude native API: claude-* models (requires ANTHROPIC_API_KEY or api_key)
    - Gemini native API: gemini-* models (requires GOOGLE_API_KEY or api_key)
    
    If base_url is provided, OpenAI-compatible API is used regardless of model name.
    
    Tool modes:
    - "V": Only search_by_embedding
    - "VM": search_by_embedding + search_by_metadata + intersect_results
    - "VF": search_by_embedding + search_by_face + get_available_people
    - "ALL": All tools (default)
    """
    # Get tools based on tool_mode
    tools = get_tools_for_mode(tool_mode)
    
    model_name_lower = model_name.lower()
    
    # Determine which LLM backend to use
    # If base_url is provided, always use OpenAI-compatible API (proxy mode)
    use_claude_native = (
        "claude" in model_name_lower 
        and base_url is None 
        and HAS_ANTHROPIC
    )
    use_gemini_native = (
        "gemini" in model_name_lower 
        and base_url is None 
        and HAS_GOOGLE_GENAI
    )
    
    if use_claude_native:
        # Use Claude native API (Anthropic)
        llm_kwargs = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        if api_key:
            llm_kwargs["anthropic_api_key"] = api_key
        # Also check environment variable
        elif os.environ.get("ANTHROPIC_API_KEY"):
            llm_kwargs["anthropic_api_key"] = os.environ["ANTHROPIC_API_KEY"]
        
        llm = ChatAnthropic(**llm_kwargs)
        
    elif use_gemini_native:
        # Use Gemini native API (Google)
        llm_kwargs = {
            "model": model_name,
            "temperature": temperature,
        }
        if api_key:
            llm_kwargs["google_api_key"] = api_key
        # Also check environment variable
        elif os.environ.get("GOOGLE_API_KEY"):
            llm_kwargs["google_api_key"] = os.environ["GOOGLE_API_KEY"]
        
        llm = ChatGoogleGenerativeAI(**llm_kwargs)
        
    else:
        # Use OpenAI-compatible API (default)
        llm_kwargs = {
            "model": model_name,
            "temperature": temperature,
        }
        if api_key:
            llm_kwargs["api_key"] = api_key
        if base_url:
            llm_kwargs["base_url"] = base_url
        
        # Qwen3 models require enable_thinking=False for non-streaming calls
        if "qwen" in model_name_lower:
            llm_kwargs["extra_body"] = {"enable_thinking": False}
        
        llm = ChatOpenAI(**llm_kwargs)
    
    llm_with_tools = llm.bind_tools(tools)
    
    # Build dynamic system prompt based on available tools
    tool_names = [t.name for t in tools]
    
    tool_descriptions = []
    tool_idx = 1
    if "search_by_face" in tool_names:
        tool_descriptions.append(f"{tool_idx}. **search_by_face**: Search for photos containing specific people by their names/nicknames")
        tool_idx += 1
    if "search_by_metadata" in tool_names:
        tool_descriptions.append(f"{tool_idx}. **search_by_metadata**: Search for photos by time (year, month, date range) and/or location")
        tool_idx += 1
    if "search_by_embedding" in tool_names:
        tool_descriptions.append(f"{tool_idx}. **search_by_embedding**: Search for photos by semantic content description (e.g., \"beach sunset\", \"birthday cake\")")
        tool_idx += 1
    if "get_available_people" in tool_names:
        tool_descriptions.append(f"{tool_idx}. **get_available_people**: Get a list of all known people and their nicknames")
        tool_idx += 1
    if "intersect_results" in tool_names:
        tool_descriptions.append(f"{tool_idx}. **intersect_results**: Combine results from multiple searches (find photos matching ALL criteria)")
        tool_idx += 1
    
    tool_desc_str = "\n".join(tool_descriptions)
    
    # Build strategy section based on available tools
    strategy_items = []
    strategy_items.append("1. Analyze the user's query to determine which tool(s) to use:")
    if "search_by_face" in tool_names:
        strategy_items.append("   - If the query mentions specific people by name → use search_by_face")
    if "search_by_metadata" in tool_names:
        strategy_items.append("   - If the query mentions time/date or location → use search_by_metadata")
    if "search_by_embedding" in tool_names:
        strategy_items.append("   - If the query describes image content/scene → use search_by_embedding")
    if len(tools) > 1:
        strategy_items.append("   - Complex queries may require multiple tools")
    
    if "intersect_results" in tool_names and len(tools) > 1:
        strategy_items.append("""
2. For multi-criteria queries:
   - First, call each relevant tool separately
   - Then use intersect_results to find photos matching ALL criteria
   - Or rank by combining scores if appropriate

3. Return the final ordered list of photos that best match the query.""")
    else:
        strategy_items.append("\n2. Return the final ordered list of photos that best match the query.")
    
    strategy_str = "\n".join(strategy_items)
    
    system_prompt = f"""You are an intelligent photo retrieval assistant. Your task is to help users find photos from their album based on their queries.

You have access to the following tools:
{tool_desc_str}

## Strategy:
{strategy_str}

## Important:
- Always explain your reasoning before calling tools
- If no results are found, try alternative approaches or explain why
- Present results clearly with the image filenames
- For embedding search, consider the similarity scores for ranking"""

    def parse_json_tool_calls(content: str) -> List[Dict]:
        """
        Parse tool calls from various formats in message content.
        Handles:
        1. Hammer format: [{"name": "func", "arguments": {...}}]
        2. ToolACE vLLM format: {"name": "func", "parameters": {...}}
        3. ToolACE native format: [func_name(param1="value1", param2=value2), ...]
        """
        if not content:
            return []
        
        # Try to find content in markdown code blocks or raw
        content = content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        
        # Method 1: Try to parse as JSON
        try:
            parsed = json.loads(content)
            
            # Case 1a: Array of tool calls (Hammer format)
            if isinstance(parsed, list):
                valid_calls = []
                for item in parsed:
                    if isinstance(item, dict) and "name" in item:
                        # Support both "arguments" and "parameters" keys
                        args = item.get("arguments") or item.get("parameters") or {}
                        call = {
                            "id": f"call_{len(valid_calls)}",
                            "name": item["name"],
                            "args": args,
                        }
                        valid_calls.append(call)
                if valid_calls:
                    return valid_calls
            
            # Case 1b: Single tool call object (ToolACE vLLM format)
            elif isinstance(parsed, dict) and "name" in parsed:
                # Support both "arguments" and "parameters" keys
                args = parsed.get("arguments") or parsed.get("parameters") or {}
                return [{
                    "id": "call_0",
                    "name": parsed["name"],
                    "args": args,
                }]
                
        except json.JSONDecodeError:
            pass
        
        # Method 2: Parse ToolACE format [func_name(param1=value1, ...), ...]
        # Pattern: func_name(key1="val1", key2=val2, ...)
        if content.startswith("[") and content.endswith("]"):
            inner = content[1:-1].strip()
            if inner:
                valid_calls = []
                # Split by "), " to get individual function calls
                # Use regex to match function calls: name(args)
                func_pattern = r'(\w+(?:\.\w+)*)\(([^)]*)\)'
                matches = re.findall(func_pattern, inner)
                
                for func_name, args_str in matches:
                    args = {}
                    if args_str.strip():
                        # Parse arguments: key="value" or key=value
                        # Handle string values with quotes and numeric/boolean values
                        arg_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\[.*?\])|(\{.*?\})|([^,\s]+))'
                        arg_matches = re.findall(arg_pattern, args_str)
                        for match in arg_matches:
                            key = match[0]
                            # Find the non-empty value group
                            value = match[1] or match[2] or match[3] or match[4] or match[5]
                            # Try to convert to appropriate type
                            if value.lower() == 'true':
                                args[key] = True
                            elif value.lower() == 'false':
                                args[key] = False
                            elif value.lower() == 'none' or value.lower() == 'null':
                                args[key] = None
                            else:
                                try:
                                    # Try integer
                                    args[key] = int(value)
                                except ValueError:
                                    try:
                                        # Try float
                                        args[key] = float(value)
                                    except ValueError:
                                        # Try JSON (for lists/dicts)
                                        try:
                                            args[key] = json.loads(value)
                                        except:
                                            # Keep as string
                                            args[key] = value
                    
                    call = {
                        "id": f"call_{len(valid_calls)}",
                        "name": func_name,
                        "args": args,
                    }
                    valid_calls.append(call)
                
                if valid_calls:
                    return valid_calls
        
        return []
    
    def should_continue(state: AgentState):
        """Check if agent should continue or stop."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if isinstance(last_message, AIMessage):
            # First check structured tool_calls
            if last_message.tool_calls:
                return "tools"
            # Then check for JSON tool calls in content (for models like Hammer)
            if last_message.content:
                json_calls = parse_json_tool_calls(last_message.content)
                if json_calls:
                    return "tools"
        return "end"
    
    def call_model(state: AgentState):
        """Call the LLM."""
        messages = state["messages"]
        
        # Add system message if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + list(messages)
        
        response = llm_with_tools.invoke(messages)
        
        # If response has no tool_calls but content looks like JSON tool calls, convert them
        if isinstance(response, AIMessage) and not response.tool_calls and response.content:
            json_calls = parse_json_tool_calls(response.content)
            if json_calls:
                # Create new AIMessage with parsed tool_calls
                response = AIMessage(
                    content="",  # Clear content since it's just the JSON
                    tool_calls=json_calls,
                )
        
        return {"messages": [response]}
    
    def extract_final_results(state: AgentState):
        """Extract final image list from the last message or tool results."""
        messages = state["messages"]
        
        all_images = []
        
        # First, try to find images in the last AI message
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                content = msg.content
                try:
                    # Look for image filenames pattern
                    images = re.findall(r'[\w\-]+(?:\s[\w\-]+)*\.(?:jpg|JPG|jpeg|JPEG|png|PNG|heic|HEIC)', content)
                    if images:
                        return {"final_results": images}
                except:
                    pass
                break  # Only check the last non-tool-calling AI message
        
        # If no images found in AI message, extract from tool results
        # This handles cases where the model doesn't properly output the final list
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                try:
                    content = msg.content
                    # Try to parse as JSON first
                    try:
                        result = json.loads(content)
                        if isinstance(result, dict):
                            # Handle search results format: {"images": [...]}
                            if "images" in result:
                                imgs = result["images"]
                                if isinstance(imgs, list):
                                    all_images.extend(imgs)
                            # Handle intersect results format: {"intersected_images": [...]}
                            if "intersected_images" in result:
                                imgs = result["intersected_images"]
                                if isinstance(imgs, list):
                                    all_images.extend(imgs)
                            # Handle ranked results format: {"ranked_images": [...]}
                            if "ranked_images" in result:
                                imgs = result["ranked_images"]
                                if isinstance(imgs, list):
                                    # ranked_images is list of [filename, score] pairs
                                    for item in imgs:
                                        if isinstance(item, list) and len(item) >= 1:
                                            all_images.append(item[0])
                                        elif isinstance(item, str):
                                            all_images.append(item)
                    except json.JSONDecodeError:
                        # Fall back to regex extraction
                        images = re.findall(r'[\w\-]+(?:\s[\w\-]+)*\.(?:jpg|JPG|jpeg|JPEG|png|PNG|heic|HEIC)', content)
                        all_images.extend(images)
                except:
                    pass
        
        # Remove duplicates while preserving order
        seen = set()
        unique_images = []
        for img in all_images:
            if img not in seen:
                seen.add(img)
                unique_images.append(img)
        
        return {"final_results": unique_images}
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    # Build graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("extract_results", extract_final_results)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": "extract_results",
        },
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("extract_results", END)
    
    return workflow.compile()


# =============================================================================
# Main Interface
# =============================================================================

class ImageRetrievalAgent:
    """High-level interface for the image retrieval agent with concurrency support."""
    
    def __init__(
        self,
        face_info_path: str,
        metadata_path: str,
        model_name: str,
        index_dir: str,
        device: str = "cuda",
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_workers: int = 4,
        tool_mode: str = "ALL",
    ):
        # Initialize databases
        init_databases(
            face_info_path=face_info_path,
            metadata_path=metadata_path,
            model_name=model_name,
            index_dir=index_dir,
            device=device,
        )
        
        # Store config for creating multiple agents
        self._llm_config = {
            "model_name": llm_model,
            "api_key": api_key,
            "base_url": base_url,
            "tool_mode": tool_mode,
        }
        
        # Create agent
        self.agent = create_react_agent(**self._llm_config)
        
        # Thread pool for concurrent query processing
        self.max_workers = max_workers
        self.tool_mode = tool_mode
    
    def query(self, query_text: str, verbose: bool = True, target_count: int = TARGET_RESULTS) -> List[str]:
        """
        Query the agent with a natural language question.
        Always returns exactly target_count results, filling with embedding search if needed.
        
        Args:
            query_text: Natural language query (e.g., "找出2019年在北京拍的妈妈的照片")
            verbose: Whether to print intermediate steps
            target_count: Target number of results to return (default: 50)
        
        Returns:
            Ordered list of exactly target_count image filenames.
        """
        initial_state = {
            "messages": [HumanMessage(content=query_text)],
            "final_results": None,
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Query: {query_text}")
            print('='*60)
        
        # Run agent with streaming
        for step in self.agent.stream(initial_state):
            if verbose:
                for node_name, output in step.items():
                    if node_name == "agent":
                        msg = output["messages"][-1]
                        if isinstance(msg, AIMessage):
                            if msg.content:
                                print(f"\n[Agent] {msg.content[:500]}...")
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    print(f"\n[Tool Call] {tc['name']}: {json.dumps(tc['args'], ensure_ascii=False)}")
                    elif node_name == "tools":
                        for msg in output["messages"]:
                            if isinstance(msg, ToolMessage):
                                result = msg.content
                                print(f"\n[Tool Result] {result}")
        
        # Get final state with recursion limit to prevent infinite loops
        final_state = self.agent.invoke(initial_state, {"recursion_limit": 15})
        raw_results = final_state.get("final_results", [])
        
        # Fill to target_count using embedding search if needed
        final_results = fill_results_with_embedding(raw_results, query_text, target_count)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Agent Results: {len(raw_results)} | After filling: {len(final_results)}")
            print(f"Final Results: {final_results}")
            print('='*60)
        
        return final_results
    
    def query_fast(self, query_text: str, target_count: int = TARGET_RESULTS) -> List[str]:
        """
        Fast query without streaming or verbose output.
        Better for batch processing. Always returns exactly target_count results.
        
        Args:
            query_text: Natural language query
            target_count: Target number of results to return (default: 50)
        
        Returns:
            Ordered list of exactly target_count image filenames.
        """
        _, filled_results = self.query_with_raw_count(query_text, target_count)
        return filled_results
    
    def query_with_raw_count(self, query_text: str, target_count: int = TARGET_RESULTS) -> tuple:
        """
        Fast query that returns both raw count and filled results.
        
        Args:
            query_text: Natural language query
            target_count: Target number of results to return (default: 50)
        
        Returns:
            Tuple of (raw_count, filled_results):
                - raw_count: Number of results before filling
                - filled_results: List of exactly target_count image filenames
        """
        initial_state = {
            "messages": [HumanMessage(content=query_text)],
            "final_results": None,
        }
        
        # Direct invoke without streaming, with recursion limit to prevent infinite loops
        final_state = self.agent.invoke(initial_state, {"recursion_limit": 15})
        raw_results = final_state.get("final_results", [])
        raw_count = len(raw_results)
        
        # Fill to target_count using embedding search if needed
        filled_results = fill_results_with_embedding(raw_results, query_text, target_count)
        
        return raw_count, filled_results
    
    def query_batch(
        self,
        queries: List[str],
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        return_raw_counts: bool = False,
    ) -> List[List[str]]:
        """
        Process multiple queries concurrently.
        
        Args:
            queries: List of query strings
            max_workers: Number of concurrent workers (default: self.max_workers)
            progress_callback: Optional callback(completed, total, query) for progress updates
            return_raw_counts: If True, return (results, raw_counts) tuple
        
        Returns:
            If return_raw_counts is False: List of results, one per query.
            If return_raw_counts is True: Tuple of (results, raw_counts).
        """
        if max_workers is None:
            max_workers = self.max_workers
        
        results = [None] * len(queries)
        raw_counts = [0] * len(queries)
        
        def process_query(idx: int, query: str) -> tuple:
            try:
                raw_count, filled_result = self.query_with_raw_count(query)
                return idx, filled_result, raw_count, None
            except Exception as e:
                return idx, [], 0, str(e)
        
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_query, idx, query): idx
                for idx, query in enumerate(queries)
            }
            
            for future in as_completed(futures):
                idx, result, raw_count, error = future.result()
                results[idx] = result
                raw_counts[idx] = raw_count
                completed += 1
                
                if error:
                    print(f"Error processing query {idx}: {error}")
                
                if progress_callback:
                    progress_callback(completed, len(queries), queries[idx])
        
        if return_raw_counts:
            return results, raw_counts
        return results
    
    async def query_async(self, query_text: str) -> List[str]:
        """
        Async version of query for use with asyncio.
        
        Args:
            query_text: Natural language query
        
        Returns:
            Ordered list of image filenames.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.query_fast, query_text)
    
    async def query_batch_async(
        self,
        queries: List[str],
        max_concurrent: int = 4,
    ) -> List[List[str]]:
        """
        Process multiple queries concurrently using asyncio.
        
        Args:
            queries: List of query strings
            max_concurrent: Maximum concurrent queries
        
        Returns:
            List of results, one per query (in same order as input).
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(query: str) -> List[str]:
            async with semaphore:
                return await self.query_async(query)
        
        tasks = [process_with_semaphore(q) for q in queries]
        return await asyncio.gather(*tasks)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Image Retrieval Agent")
    parser.add_argument("--face_info_path", required=True, help="Path to face_info JSON file")
    parser.add_argument("--metadata_path", required=True, help="Path to metadata JSON file")
    parser.add_argument("--model_name", required=True, help="Embedding model name")
    parser.add_argument("--index_dir", required=True, help="Path to FAISS index directory")
    parser.add_argument("--device", default="cuda", help="Device for embedding model")
    parser.add_argument("--llm_model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--api_key", default=None, help="OpenAI API key (or set OPENAI_API_KEY env)")
    parser.add_argument("--base_url", default=None, help="OpenAI API base URL")
    parser.add_argument("--tool_mode", default="ALL", choices=["V", "VM", "VF", "ALL"],
                       help="Tool mode: V (embedding only), VM (embedding+metadata), VF (embedding+face), ALL (default)")
    parser.add_argument("--query", required=True, help="Query text")
    args = parser.parse_args()
    
    agent = ImageRetrievalAgent(
        face_info_path=args.face_info_path,
        metadata_path=args.metadata_path,
        model_name=args.model_name,
        index_dir=args.index_dir,
        device=args.device,
        llm_model=args.llm_model,
        api_key=args.api_key,
        base_url=args.base_url,
        tool_mode=args.tool_mode,
    )
    
    results = agent.query(args.query)
    print(f"\nFinal image list ({len(results)} images):")
    for img in results:
        print(f"  - {img}")


if __name__ == "__main__":
    main()

