#!/usr/bin/env python3
"""
Notebook Analyzer Core Module

Enhanced analysis classes that use comprehensive NVIDIA Best Practices
for evaluating Jupyter notebooks and determining GPU requirements.
"""

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, ConnectionError, Timeout, HTTPError
import json
import re
import ast
import os
import sys
import ipaddress
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from urllib.parse import urlparse, quote_plus
from concurrent.futures import ThreadPoolExecutor
import threading
import tempfile
import urllib.parse

# Performance optimization: HTTP connection pooling
_http_session = None
_session_lock = threading.Lock()

def get_http_session() -> requests.Session:
    """Get a shared HTTP session with connection pooling and retry configuration."""
    global _http_session
    
    if _http_session is None:
        with _session_lock:
            if _http_session is None:
                _http_session = requests.Session()
                # Configure connection pooling with retry logic
                from urllib3.util.retry import Retry
                retry_strategy = Retry(
                    total=3,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],  # Updated parameter name
                    backoff_factor=1
                )
                adapter = HTTPAdapter(
                    pool_connections=10,
                    pool_maxsize=20,
                    max_retries=retry_strategy
                )
                _http_session.mount('http://', adapter)
                _http_session.mount('https://', adapter)
    
    return _http_session

def make_api_request_with_retry(session, url, headers, json_data, timeout, max_retries=3, progress_callback=None):
    """
    Make API request with exponential backoff retry logic for network issues.
    
    Args:
        session: requests.Session object
        url: API endpoint URL
        headers: Request headers
        json_data: JSON payload
        timeout: Request timeout
        max_retries: Maximum number of retry attempts
        progress_callback: Optional callback for progress updates
    
    Returns:
        requests.Response object or None if all retries failed
    """
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                # Exponential backoff with jitter
                delay = (2 ** attempt) + random.uniform(0, 1)
                if progress_callback:
                    progress_callback(f"🔄 Retrying API request (attempt {attempt + 1}/{max_retries + 1}) in {delay:.1f}s...")
                time.sleep(delay)
            
            response = session.post(url, headers=headers, json=json_data, timeout=timeout)
            
            # Check for rate limiting
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After', '60')
                try:
                    retry_delay = int(retry_after)
                except ValueError:
                    retry_delay = 60
                
                if progress_callback:
                    progress_callback(f"⏳ Rate limited. Waiting {retry_delay}s before retry...")
                time.sleep(retry_delay)
                continue
            
            # Success or non-retryable error
            return response
            
        except ConnectionError as e:
            if progress_callback:
                progress_callback(f"🌐 Connection error (attempt {attempt + 1}): {str(e)[:50]}...")
        except Timeout as e:
            if progress_callback:
                progress_callback(f"⏱️ Request timeout (attempt {attempt + 1}): {str(e)[:50]}...")
        except HTTPError as e:
            if progress_callback:
                progress_callback(f"🚫 HTTP error (attempt {attempt + 1}): {str(e)[:50]}...")
        except RequestException as e:
            if progress_callback:
                progress_callback(f"📡 Request error (attempt {attempt + 1}): {str(e)[:50]}...")
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Unexpected error (attempt {attempt + 1}): {str(e)[:50]}...")
        
        # If this was the last attempt, break
        if attempt == max_retries:
            break
    
    # All retries failed
    if progress_callback:
        progress_callback("❌ All API retry attempts failed - falling back to static analysis")
    return None

# Performance optimization: Pre-compiled regex patterns
_compiled_patterns = {}
_pattern_lock = threading.Lock()

def get_compiled_pattern(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern:
    """Get a compiled regex pattern from cache or compile and cache it."""
    cache_key = (pattern, flags)
    
    if cache_key not in _compiled_patterns:
        with _pattern_lock:
            # Double-check pattern
            if cache_key not in _compiled_patterns:
                _compiled_patterns[cache_key] = re.compile(pattern, flags)
    
    return _compiled_patterns[cache_key]

def parallel_pattern_search(text: str, patterns: List[str], flags: int = re.IGNORECASE) -> List[bool]:
    """Search for multiple patterns in parallel for better performance."""
    def search_pattern(pattern):
        compiled_pattern = get_compiled_pattern(pattern, flags)
        return bool(compiled_pattern.search(text))
    
    # Use ThreadPoolExecutor for I/O bound regex operations
    with ThreadPoolExecutor(max_workers=min(8, len(patterns))) as executor:
        results = list(executor.map(search_pattern, patterns))
    
    return results

# Runtime utility functions (consolidated from LLMAnalyzer and GPUAnalyzer)
def parse_runtime_range(runtime_str: str) -> tuple:
    """Parse runtime string like '1.5-2.5' into (min, max) float tuple."""
    try:
        if '-' in runtime_str:
            parts = runtime_str.split('-')
            min_time = float(parts[0])
            max_time = float(parts[1])
            return (min_time, max_time)
        else:
            # Single value
            time_val = float(runtime_str)
            return (time_val, time_val)
    except:
        # Fallback for parsing errors
        return (1.0, 2.0)

def format_runtime(time_hours: float) -> str:
    """Format runtime value to show minutes if < 1 hour, hours if >= 1 hour."""
    if time_hours < 1.0:
        minutes = int(time_hours * 60)
        return f"{minutes} minutes"
    else:
        return f"{time_hours:.1f} hours"

def format_runtime_range(min_hours: float, max_hours: float) -> str:
    """Format runtime range to show appropriate units."""
    if min_hours == max_hours:
        return format_runtime(min_hours)
    else:
        # If both values are in the same unit range, format consistently
        if min_hours < 1.0 and max_hours < 1.0:
            # Both in minutes
            min_minutes = int(min_hours * 60)
            max_minutes = int(max_hours * 60)
            return f"{min_minutes}-{max_minutes} minutes"
        elif min_hours >= 1.0 and max_hours >= 1.0:
            # Both in hours
            return f"{min_hours:.1f}-{max_hours:.1f} hours"
        else:
            # Mixed units - format each separately
            return f"{format_runtime(min_hours)}-{format_runtime(max_hours)}"

def convert_runtime_to_new_format(runtime_str: str) -> str:
    """Convert runtime string to standardized format (unified from both implementations)."""
    if not runtime_str or runtime_str in ["N/A", "Unknown"]:
        return "1-2 hours"
    
    # Handle range formats like "1.0-2.0"
    if "-" in runtime_str and "hour" not in runtime_str and "minute" not in runtime_str:
        try:
            parts = runtime_str.split("-")
            if len(parts) == 2:
                min_val = float(parts[0])
                max_val = float(parts[1])
                return format_runtime_range(min_val, max_val)
        except ValueError:
            pass
    
    # Try to parse and reformat using the range parser
    try:
        min_time, max_time = parse_runtime_range(runtime_str)
        return format_runtime_range(min_time, max_time)
    except:
        # If all parsing fails, return the original string
        return runtime_str

# General utility functions (extracted from GPUAnalyzer)
def compare_versions(version1: str, version2: str) -> int:
    """
    Compare two version strings.
    Returns: -1 if version1 < version2, 0 if equal, 1 if version1 > version2
    """
    def normalize_version(v):
        # Convert version string to list of integers
        try:
            return [int(x) for x in v.split('.')]
        except ValueError:
            # Handle non-numeric version parts
            parts = []
            for part in v.split('.'):
                try:
                    parts.append(int(part))
                except ValueError:
                    # For non-numeric parts, use their ASCII values
                    parts.append(ord(part[0]) if part else 0)
            return parts
    
    v1_parts = normalize_version(version1)
    v2_parts = normalize_version(version2)
    
    # Pad shorter version with zeros
    max_len = max(len(v1_parts), len(v2_parts))
    v1_parts += [0] * (max_len - len(v1_parts))
    v2_parts += [0] * (max_len - len(v2_parts))
    
    for i in range(max_len):
        if v1_parts[i] < v2_parts[i]:
            return -1
        elif v1_parts[i] > v2_parts[i]:
            return 1
    
    return 0

def normalize_gpu_quantity(quantity: int) -> int:
    """Normalize GPU quantity to valid values: 1, 2, 4, 8, or multiples of 8."""
    if quantity <= 1:
        return 1
    elif quantity <= 2:
        return 2
    elif quantity <= 4:
        return 4
    elif quantity <= 8:
        return 8
    else:
        # For quantities > 8, round to nearest multiple of 8
        return ((quantity + 7) // 8) * 8

def calculate_multi_gpu_scaling(quantity: int) -> float:
    """Calculate multi-GPU scaling efficiency factor."""
    if quantity <= 1:
        return 1.0
    elif quantity == 2:
        return 0.55  # ~45% speedup per GPU (1/0.55 = 1.82x total speedup)
    elif quantity == 4:
        return 0.35  # ~65% speedup per GPU (1/0.35 = 2.86x total speedup) 
    elif quantity == 8:
        return 0.25  # ~75% speedup per GPU (1/0.25 = 4.0x total speedup)
    else:
        # For larger quantities, scaling becomes less efficient
        return max(0.15, 0.25 * (8 / quantity))

@dataclass
class GPURequirement:
    # Minimum (entry-level viable option)
    min_gpu_type: str
    min_quantity: int
    min_vram_gb: int
    min_runtime_estimate: str
    
    # Recommended (balanced price/performance option)
    recommended_gpu_type: Optional[str] = None
    recommended_quantity: Optional[int] = None
    recommended_vram_gb: Optional[int] = None
    recommended_runtime_estimate: Optional[str] = None
    recommended_viable: bool = True
    recommended_limitation: Optional[str] = None  # Why not viable if recommended_viable is False
    
    # Optimal (high performance option)
    optimal_gpu_type: str = ""
    optimal_quantity: int = 1
    optimal_vram_gb: int = 0
    optimal_runtime_estimate: str = ""
    
    # Backward compatibility - these will be populated from recommended/optimal above
    
    # Existing fields
    sxm_required: bool = False
    sxm_reasoning: List[str] = field(default_factory=list)
    arm_compatibility: str = ""
    arm_reasoning: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: List[str] = field(default_factory=list)
    llm_enhanced: bool = False
    llm_reasoning: Optional[List[str]] = None
    self_reviewed: bool = False  # Phase 2.5: Track if analysis went through self-review
    llm_model_used: Optional[str] = None  # Track which LLM model was used for analysis
    nvidia_compliance_score: float = 0.0
    structure_assessment: Optional[Dict[str, str]] = None
    content_quality_issues: Optional[List[str]] = None
    technical_recommendations: Optional[List[str]] = None
    confidence_factors: Optional[List[str]] = None
    workload_detected: bool = False  # Whether a GPU workload was detected
    
    def __post_init__(self):
        # Ensure lists are properly initialized
        if self.llm_reasoning is None:
            self.llm_reasoning = []
        if self.structure_assessment is None:
            self.structure_assessment = {}
        if self.content_quality_issues is None:
            self.content_quality_issues = []
        if self.technical_recommendations is None:
            self.technical_recommendations = []
        if self.confidence_factors is None:
            self.confidence_factors = []
    
    # Backward compatibility properties
    @property
    def consumer_gpu_type(self):
        return self.recommended_gpu_type
    
    @property
    def consumer_quantity(self):
        return self.recommended_quantity
        
    @property
    def consumer_vram_gb(self):
        return self.recommended_vram_gb
        
    @property
    def consumer_runtime_estimate(self):
        return self.recommended_runtime_estimate
        
    @property
    def consumer_viable(self):
        return self.recommended_viable
        
    @property
    def consumer_limitation(self):
        return self.recommended_limitation
        
    @property
    def enterprise_gpu_type(self):
        return self.optimal_gpu_type
        
    @property
    def enterprise_quantity(self):
        return self.optimal_quantity
        
    @property
    def enterprise_vram_gb(self):
        return self.optimal_vram_gb
        
    @property
    def enterprise_runtime_estimate(self):
        return self.optimal_runtime_estimate


class NVIDIABestPracticesLoader:
    """Load and parse NVIDIA Best Practices for Notebooks from markdown file."""
    
    def __init__(self):
        self.guidelines = None
        self.scoring_framework = None
        self.checklist = None
        self._load_guidelines()
    
    def _load_guidelines(self):
        """Load the NVIDIA Best Practices markdown file."""
        try:
            # Get the path to the guidelines file
            current_dir = Path(__file__).parent
            guidelines_path = current_dir / "nvidia_best_practices.md"
            
            if not guidelines_path.exists():
                print(f"Warning: NVIDIA Best Practices file not found at {guidelines_path}")
                self._load_default_guidelines()
                return
            
            with open(guidelines_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self._parse_guidelines(content)
            
        except Exception as e:
            print(f"Error loading NVIDIA Best Practices: {e}")
            self._load_default_guidelines()
    
    def _parse_guidelines(self, content: str):
        """Parse the markdown content into structured guidelines."""
        self.guidelines = {
            'structure_layout': self._extract_section(content, "Structure and Layout Requirements"),
            'messaging_content': self._extract_section(content, "Messaging and Content Standards"),
            'technical_standards': self._extract_section(content, "Technical Implementation Standards"),
            'compliance_scoring': self._extract_section(content, "Compliance Scoring Framework"),
            'implementation_checklist': self._extract_section(content, "Implementation Checklist"),
            'common_issues': self._extract_section(content, "Common Issues and Solutions")
        }
        
        # Extract scoring framework
        scoring_section = self._extract_section(content, "Compliance Scoring Framework")
        self.scoring_framework = self._parse_scoring_framework(scoring_section)
        
        # Extract checklist items
        checklist_section = self._extract_section(content, "Implementation Checklist")
        self.checklist = self._parse_checklist(checklist_section)
    
    def _extract_section(self, content: str, section_title: str) -> str:
        """Extract a specific section from the markdown content."""
        pattern = f"## {re.escape(section_title)}(.*?)(?=## |$)"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _parse_scoring_framework(self, content: str) -> Dict:
        """Parse the scoring framework into structured data."""
        framework = {
            'structure_layout': {'max_points': 25, 'criteria': []},
            'content_quality': {'max_points': 25, 'criteria': []},
            'technical_standards': {'max_points': 25, 'criteria': []},
            'nvidia_compliance': {'max_points': 25, 'criteria': []}
        }
        
        # Extract criteria for each category
        for category in framework.keys():
            category_pattern = f"### {category.replace('_', ' ').title()}.*?(?=### |##|$)"
            category_match = re.search(category_pattern, content, re.DOTALL | re.IGNORECASE)
            if category_match:
                criteria_text = category_match.group(0)
                # Extract individual criteria with point values
                criteria_pattern = r'- \*\*(.*?)\s*\(([\d.]+)\s*points?\)\*\*:(.*?)(?=- \*\*|$)'
                criteria = re.findall(criteria_pattern, criteria_text, re.DOTALL)
                framework[category]['criteria'] = [
                    {
                        'name': name.strip(),
                        'points': float(points),
                        'description': desc.strip()
                    }
                    for name, points, desc in criteria
                ]
        
        return framework
    
    def _parse_checklist(self, content: str) -> Dict:
        """Parse the implementation checklist."""
        checklist = {
            'pre_publication': [],
            'quality_assurance': [],
            'publication_readiness': []
        }
        
        # Extract checklist items
        checklist_pattern = r'- \[ \] (.+)'
        items = re.findall(checklist_pattern, content)
        
        # Group items by section (this is simplified - could be enhanced)
        current_section = 'pre_publication'
        for item in items:
            if 'Quality Assurance' in content[:content.find(item)]:
                current_section = 'quality_assurance'
            elif 'Publication Readiness' in content[:content.find(item)]:
                current_section = 'publication_readiness'
            
            checklist[current_section].append(item.strip())
        
        return checklist
    
    def _load_default_guidelines(self):
        """Load minimal default guidelines if the file is not available."""
        self.guidelines = {
            'structure_layout': "Use clear titles, introductions, and conclusions",
            'messaging_content': "Follow NVIDIA branding and maintain professional content",
            'technical_standards': "Use requirements.txt, environment variables, and reproducible code",
            'compliance_scoring': "Score based on structure, content, technical, and brand compliance",
            'implementation_checklist': "Verify all requirements before publication",
            'common_issues': "Address title format, dependencies, and conclusion quality"
        }
    
    def get_guidelines_for_evaluation(self) -> str:
        """Get formatted guidelines for LLM evaluation prompts."""
        if not self.guidelines:
            return "Use NVIDIA best practices for notebook evaluation."
        
        return f"""
NVIDIA NOTEBOOK BEST PRACTICES:

STRUCTURE & LAYOUT:
{self.guidelines.get('structure_layout', '')[:500]}...

CONTENT STANDARDS:
{self.guidelines.get('messaging_content', '')[:500]}...

TECHNICAL REQUIREMENTS:
{self.guidelines.get('technical_standards', '')[:500]}...

SCORING FRAMEWORK:
{self.guidelines.get('compliance_scoring', '')[:500]}...
        """.strip()


class LLMAnalyzer:
    def __init__(self, base_url: str, model: str, api_key: str, gpu_specs: Optional[Dict] = None):
        # Remove trailing slashes and /v1 suffix to avoid double /v1 in URLs
        self.base_url = base_url.rstrip('/').rstrip('/v1')
        self.model = model
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        # Load NVIDIA Best Practices
        self.best_practices = NVIDIABestPracticesLoader()
        
        # Store GPU specs for validation
        self.gpu_specs = gpu_specs or {}
        
        # Get environment-specific configuration
        self.env_config = get_environment_config()
        env_type = self.env_config.get('environment_type', 'unknown')
        print(f"LLM Analyzer initialized for {env_type} environment")
        
        # Debug environment detection
        if is_production_environment():
            vercel_env = os.getenv('VERCEL_ENV', 'not_set')
            vercel_plan = os.getenv('VERCEL_PLAN', 'not_set')
            vercel_pro_features = os.getenv('VERCEL_PRO_FEATURES', 'not_set')
            enable_self_review = os.getenv('ENABLE_SELF_REVIEW', 'not_set')
            enhanced_features = os.getenv('ENHANCED_FEATURES', 'not_set')
            full_transparency = os.getenv('FULL_TRANSPARENCY', 'not_set')
            debug_detailed_phases = os.getenv('DEBUG_DETAILED_PHASES', 'not_set')
            print(f"🔍 Environment debug: VERCEL_ENV={vercel_env}, VERCEL_PLAN={vercel_plan}")
            print(f"🔍 Pro features: VERCEL_PRO_FEATURES={vercel_pro_features}, ENABLE_SELF_REVIEW={enable_self_review}")
            print(f"🔍 Enhanced features: ENHANCED_FEATURES={enhanced_features}, FULL_TRANSPARENCY={full_transparency}")
            print(f"🔍 Debug override: DEBUG_DETAILED_PHASES={debug_detailed_phases}")
        
        # Show final configuration
        print(f"🔧 Configuration: detailed_phases={self.env_config.get('detailed_phases', False)}, self_review_enabled={self.env_config.get('self_review_enabled', False)}")
        
        if env_type == 'vercel_pro':
            print("🚀 Vercel Pro features enabled: Full transparency + Smart self-review")
            print(f"🎓 Self-review enabled: {self.env_config.get('self_review_enabled', False)}")
        elif env_type == 'vercel_free':
            print("⚡ Vercel Free plan detected: Simplified progress + Self-review disabled")
            print(f"💡 Tip: Set DEBUG_DETAILED_PHASES=true to see detailed progress for debugging")
        else:
            print(f"⚡ Self-review enabled: {self.env_config.get('self_review_enabled', False)}")
    
    # Runtime utility methods now use module-level functions
    def _parse_runtime_range(self, runtime_str: str) -> tuple:
        """Parse runtime string like '1.5-2.5' into (min, max) float tuple."""
        return parse_runtime_range(runtime_str)
    
    def _format_runtime(self, time_hours: float) -> str:
        """Format runtime value to show minutes if < 1 hour, hours if >= 1 hour."""
        return format_runtime(time_hours)
    
    def _format_runtime_range(self, min_hours: float, max_hours: float) -> str:
        """Format runtime range to show appropriate units."""
        return format_runtime_range(min_hours, max_hours)
    
    def _convert_runtime_to_new_format(self, runtime_str: str) -> str:
        """Convert existing runtime string to new format with minutes/hours."""
        return convert_runtime_to_new_format(runtime_str)

    def _clean_json_response(self, json_str: str) -> str:
        """Clean JSON response by removing comments and fixing common issues."""
        import re
        
        # Remove // comments (but preserve URLs like https://)
        # This regex looks for // that are not preceded by http: or https:
        json_str = re.sub(r'(?<!https:)(?<!http:)//.*?(?=\n|$)', '', json_str)
        
        # Remove /* */ style comments
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Remove trailing non-JSON characters
        json_str = json_str.rstrip('%').rstrip()
        
        # Fix common JSON issues
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        return json_str
    
    def _get_self_review_max_tokens(self) -> int:
        """Get appropriate max tokens for self-review based on environment."""
        env_type = self.env_config.get('environment_type', 'local_development')
        
        if env_type == 'local_development':
            # Local development: Allow generous token limit for complete self-review
            return 1500
        elif env_type == 'vercel_pro':
            # Vercel Pro: Balanced token limit
            return 1200
        elif self.env_config.get('smart_self_review', False):
            # Smart self-review: Reduced tokens for efficiency
            return 800
        else:
            # Default/Vercel Free: Standard limit
            return 1000
    
    def analyze_notebook_context(self, code_cells: List[str], markdown_cells: List[str], progress_callback=None) -> Optional[Dict]:
        """Send notebook to LLM for contextual analysis."""
        try:
            if progress_callback:
                if self.env_config['detailed_phases']:
                    progress_callback("🤖 Starting AI analysis...")
                else:
                    progress_callback("🤖 Analyzing workload with AI...")
            
            # Phase 1: Content Extraction and Filtering
            relevant_code = []
            relevant_markdown = []
            
            if progress_callback and self.env_config['detailed_phases']:
                progress_callback("🔍 Phase 1: Scanning notebook for ML/GPU patterns...")
            
            # Extract only GPU-relevant code cells (more efficient)
            gpu_keywords = ['cuda', 'gpu', 'torch', 'tensorflow', 'train', 'model', 'batch', 'device']
            frameworks_found = set()
            models_mentioned = set()
            
            for cell in code_cells[:15]:  # Check more cells but filter
                if any(keyword in cell.lower() for keyword in gpu_keywords):
                    relevant_code.append(cell)
                    
                    # Detect specific frameworks and models during scanning
                    if 'torch' in cell.lower() or 'pytorch' in cell.lower():
                        frameworks_found.add('PyTorch')
                    if 'tensorflow' in cell.lower() or 'keras' in cell.lower():
                        frameworks_found.add('TensorFlow')
                    if 'transformers' in cell.lower():
                        frameworks_found.add('Hugging Face')
                    if 'cuda' in cell.lower() or '@cuda.jit' in cell.lower():
                        frameworks_found.add('CUDA')
                    if 'numba' in cell.lower():
                        frameworks_found.add('Numba')
                    
                    # Look for specific model mentions
                    model_patterns = ['bert', 'gpt', 'llama', 'mistral', 'claude', 'resnet', 'vgg', 'yolo']
                    for pattern in model_patterns:
                        if pattern in cell.lower():
                            models_mentioned.add(pattern.upper())
                    
                if len(relevant_code) >= 8:  # Limit to most relevant
                    break
            
            # Report findings from Phase 1 (only in detailed mode)
            if progress_callback and self.env_config['detailed_phases']:
                if frameworks_found:
                    progress_callback(f"🔍 Found frameworks: {', '.join(frameworks_found)}")
                if models_mentioned:
                    progress_callback(f"🤖 Detected models: {', '.join(list(models_mentioned)[:3])}")
                if not frameworks_found and not models_mentioned:
                    progress_callback("🔍 No obvious ML/GPU patterns found in code scan")
            
            # If no GPU-relevant code found, take first few cells
            if not relevant_code:
                relevant_code = code_cells[:5]
                if progress_callback and self.env_config['detailed_phases']:
                    progress_callback("📄 No GPU-specific code found, analyzing general content...")
            
            # Phase 2: Markdown Analysis (simplified in production)
            if progress_callback and self.env_config['detailed_phases']:
                progress_callback("📝 Phase 2: Analyzing documentation and requirements...")
            
            # Extract key markdown (titles, requirements, etc.)
            markdown_insights = []
            for cell in markdown_cells[:3]:
                if any(keyword in cell.lower() for keyword in ['requirement', 'gpu', 'hardware', 'setup']):
                    relevant_markdown.append(cell)
                    # Look for specific requirements mentioned (only track in detailed mode)
                    if self.env_config['detailed_phases']:
                        if 'gpu' in cell.lower():
                            markdown_insights.append("GPU requirements mentioned")
                        if 'cuda' in cell.lower():
                            markdown_insights.append("CUDA requirements noted")
                        if 'memory' in cell.lower() or 'ram' in cell.lower():
                            markdown_insights.append("Memory requirements discussed")
            
            if progress_callback and self.env_config['detailed_phases'] and markdown_insights:
                progress_callback(f"📝 Documentation insights: {', '.join(markdown_insights)}")
            
            # Phase 3: Content Preparation (simplified in production)
            if progress_callback and self.env_config['detailed_phases']:
                progress_callback("📊 Phase 3: Preparing content for AI analysis...")
            
            # Combine with smart truncation
            notebook_content = "\n".join([
                "=== RELEVANT CODE ===",
                *relevant_code,
                "\n=== KEY DOCUMENTATION ===" if relevant_markdown else "",
                *relevant_markdown
            ])
            
            # Environment-specific content limits
            max_content = self.env_config['max_content_length']
            original_length = len(notebook_content)
            if len(notebook_content) > max_content:
                notebook_content = notebook_content[:max_content] + "\n... [content truncated for efficiency]"
                if progress_callback and self.env_config['detailed_phases']:
                    progress_callback(f"📊 Content prepared: {len(notebook_content)} chars (truncated from {original_length})")
            else:
                if progress_callback and self.env_config['detailed_phases']:
                    progress_callback(f"📊 Content prepared: {len(notebook_content)} characters")
            
            # Phase 4: AI Request Preparation (simplified in production)
            if progress_callback and self.env_config['detailed_phases']:
                progress_callback("🧠 Phase 4: Formulating analysis request...")
            elif progress_callback:
                progress_callback("🧠 Preparing AI analysis request...")
            
            prompt = f"""Analyze this Jupyter notebook for GPU requirements using the 3-tier recommendation system (minimum/recommended/optimal). 

**CRITICAL FIRST STEP**: Determine if this workload actually needs a GPU at all.

**CPU-Only Workloads** (no GPU needed):
- Basic Python operations (print, variables, loops, functions)
- Simple data manipulation (small lists, dictionaries, basic math)
- File operations (reading files, environment variables)
- Pure data analysis with pandas/numpy on small datasets
- Basic visualization with matplotlib/seaborn
- Simple string processing
- Tutorial/educational content with no computational workload

**GPU-Beneficial Workloads**:
1. **Workload Type**: Is this inference, fine-tuning, training from scratch, GPU computing, or other?
2. **Model Details**: What models are being used? What are their memory requirements?
3. **Batch Sizes**: What batch sizes are used and are they for experimentation or production?
4. **Memory Optimization**: Are there techniques like gradient checkpointing, LoRA, quantization?
5. **Multi-GPU Patterns**: Is distributed training or model parallelism used?
6. **Dataset Scale**: How large is the dataset being processed?
7. **GPU Computing**: Look for CUDA kernels, Numba CUDA (@cuda.jit), PyCUDA, or other GPU acceleration
8. **Runtime Estimation**: Based on workload complexity, model size, optimizations, and typical convergence
9. **Performance Considerations**: Consider GPU performance differences when recommending hardware
10. **Tutorial Detection**: Identify if this is a tutorial/demo with small datasets vs production workload

**IMPORTANT**: Pay special attention to:
- **CPU-only indicators**: Basic Python, simple math, small data, no ML/AI libraries
- Numba CUDA patterns (@cuda.jit, cuda.grid, cuda.device_array)
- Direct CUDA programming (PyCUDA, CuPy)
- GPU-accelerated computing (not just ML/AI)
- Scientific computing that uses GPU acceleration
- Small dataset indicators (sample sizes, tutorial mentions, example data)
- Context clues that suggest demo/tutorial vs production use

**3-Tier System Context**:
- **Minimum**: Entry-level viable option (lowest cost that works)
- **Recommended**: Balanced price/performance (best value for most users)
- **Optimal**: High performance option (best performance regardless of cost)

**GPU Performance Context** (relative performance factors):
- **Consumer GPUs**: RTX 4060 (0.35×), RTX 4070 (0.50×), RTX 4080 (0.70×), RTX 4090 (1.0× baseline), RTX 5080 (1.15×), RTX 5090 (1.4×)
        - **Enterprise GPUs**: L4 (0.25×), L40 (0.60×), L40S (0.75×), A100 PCIe (1.0×), A100 SXM (1.05×), H100 PCIe (2.2×), H100 SXM (2.4×), H100 NVL (2.3×), H200 SXM (2.8×), H200 NVL (2.8×), B200 SXM (4.0×)
- Performance factors relative to RTX 4090. Higher = faster execution.
- Enterprise GPUs provide better reliability, ECC memory, and professional support.

**Valid GPU Options** (use ONLY these):
- Consumer: RTX 4060, RTX 4070, RTX 4080, RTX 4090, RTX 5080, RTX 5090
- Enterprise: L4, L40, L40S, A100 PCIe 40G, A100 PCIe 80G, A100 SXM 40G, A100 SXM 80G, H100 PCIe, H100 SXM, H100 NVL, H200 SXM, H200 NVL, B200 SXM

Notebook Content:
{notebook_content}

Respond in JSON format with:
{{
    "workload_type": "cpu-only|inference|fine-tuning|training|gpu-computing|other",
    "complexity": "simple|moderate|complex|extreme",
    "models_detected": ["model1", "model2"],
    "estimated_vram_gb": number,
    "multi_gpu_required": boolean,
    "memory_optimizations": ["technique1", "technique2"],
    "batch_size_analysis": "description",
    "runtime_factors": ["factor1", "factor2"],
    "baseline_runtime_hours": "1.5-2.5",
    "baseline_reference_gpu": "RTX 4090",
    "optimization_speedup_factor": 0.7,
    "performance_considerations": ["Consider enterprise GPU performance vs consumer options"],
    "tutorial_indicators": ["indicator1", "indicator2"],
    "dataset_scale": "small|medium|large|enterprise",
    "confidence": 0.0-1.0,
    "reasoning": ["reason1", "reason2"],
    "runtime_reasoning": ["runtime factor1", "runtime factor2"]
}}

**Critical Analysis Guidelines**:
- **CPU-Only First**: If the notebook contains only basic Python operations, simple data manipulation, or educational content with no computational workload, classify as "cpu-only" with estimated_vram_gb: 0
- For tutorials/demos with small datasets: Recommend consumer GPUs even if enterprise tools are mentioned
- For production workloads: Consider enterprise GPUs for reliability and support
- For GPU computing: Assess actual computational complexity, not just tool mentions
- For VRAM estimation: Consider batch sizes, model parameters, and optimization techniques (set to 0 for CPU-only workloads)
- For runtime estimation: Account for GPU performance differences and optimization speedups
- Always provide realistic, evidence-based recommendations

For runtime estimation:
- baseline_runtime_hours: Estimated time on baseline_reference_gpu (single GPU)
- baseline_reference_gpu: Reference GPU for the estimate (typically RTX 4090 or A100)  
- optimization_speedup_factor: Combined speedup from optimizations (0.5 = 50% faster, 1.0 = no change)
- performance_considerations: Notes about GPU performance trade-offs and recommendations
- Consider: model parameters, dataset size, epochs, optimizations like LoRA/quantization, and GPU performance factors"""

            # Phase 5: AI Analysis Request (simplified in production)
            if progress_callback and self.env_config['detailed_phases']:
                progress_callback("🚀 Phase 5: Sending analysis to AI model...")
            elif progress_callback:
                progress_callback("🚀 Sending analysis to AI...")
            
            # Use robust retry mechanism for API requests
            session = get_http_session()
            response = make_api_request_with_retry(
                session=session,
                url=f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json_data={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert in GPU computing and machine learning workloads. Analyze notebooks for accurate GPU requirements."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1000
                },
                timeout=self.env_config['llm_timeout'],
                max_retries=3,
                progress_callback=progress_callback
            )
            
            # Handle response or fallback
            if response is None:
                # All retries failed - return fallback analysis
                if progress_callback:
                    progress_callback("🔄 Using fallback analysis due to API connectivity issues...")
                return {
                    "workload_type": "unknown",
                    "complexity": "moderate",
                    "models_detected": [],
                    "estimated_vram_gb": 16,
                    "multi_gpu_required": False,
                    "memory_optimizations": [],
                    "batch_size_analysis": "Could not analyze",
                    "runtime_factors": [],
                    "confidence": 0.3,
                    "reasoning": ["API connectivity issues - using fallback analysis"]
                }
            
            # Phase 6: Response Processing (simplified in production)
            if progress_callback and self.env_config['detailed_phases']:
                progress_callback("📥 Phase 6: Processing AI response...")
            elif progress_callback:
                progress_callback("📥 Processing AI response...")
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                if progress_callback and self.env_config['detailed_phases']:
                    progress_callback("🔍 Phase 7: Parsing analysis results...")
                elif progress_callback:
                    progress_callback("🔍 Parsing analysis results...")
                
                # Try to extract JSON from the response
                try:
                    # Look for JSON block in the response
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        analysis_result = json.loads(json_str)
                        
                                                # Phase 8: Results Analysis and Insights (detailed only in development)
                        if progress_callback:
                            progress_callback("🎯 Phase 8: Extracting key insights...")
                            
                            if self.env_config['detailed_phases']:
                                # Show detailed insights from the analysis
                                workload_type = analysis_result.get('workload_type', 'unknown')
                                complexity = analysis_result.get('complexity', 'unknown')
                                models = analysis_result.get('models_detected', [])
                                vram = analysis_result.get('estimated_vram_gb', 0)
                                multi_gpu = analysis_result.get('multi_gpu_required', False)
                                optimizations = analysis_result.get('memory_optimizations', [])
                                confidence = analysis_result.get('confidence', 0)
                                
                                # Detailed workload classification
                                progress_callback(f"🎯 Workload classified: {workload_type}")
                                progress_callback(f"⚡ Complexity level: {complexity}")
                                
                                # Model detection results
                                if models:
                                    progress_callback(f"🤖 Models identified: {', '.join(models[:3])}")
                                else:
                                    progress_callback("🤖 No specific models detected")
                                
                                # VRAM analysis
                                if vram > 0:
                                    progress_callback(f"💾 VRAM requirement: {vram}GB")
                                    if multi_gpu:
                                        progress_callback("🔗 Multi-GPU setup recommended")
                                    else:
                                        progress_callback("🖥️ Single GPU sufficient")
                                
                                # Optimization insights
                                if optimizations:
                                    progress_callback(f"⚡ Optimizations detected: {', '.join(optimizations)}")
                                else:
                                    progress_callback("⚡ No memory optimizations detected")
                                
                                # Confidence assessment
                                confidence_pct = confidence * 100
                                if confidence_pct >= 80:
                                    progress_callback(f"✅ High confidence analysis: {confidence_pct:.0f}%")
                                elif confidence_pct >= 60:
                                    progress_callback(f"🟡 Moderate confidence: {confidence_pct:.0f}%")
                                else:
                                    progress_callback(f"🟠 Lower confidence: {confidence_pct:.0f}% (limited evidence)")
                                
                                # Reasoning insights
                                reasoning = analysis_result.get('reasoning', [])
                                if reasoning:
                                    # Show the most important reasoning point
                                    key_reason = reasoning[0] if reasoning else "Analysis completed"
                                    progress_callback(f"💡 Key insight: {key_reason}")
                            else:
                                # Simplified production feedback
                                workload_type = analysis_result.get('workload_type', 'unknown')
                                confidence = analysis_result.get('confidence', 0)
                                vram = analysis_result.get('estimated_vram_gb', 0)
                                progress_callback(f"🎯 Workload: {workload_type} ({confidence*100:.0f}% confidence)")
                                if vram > 0:
                                    progress_callback(f"💾 VRAM needed: {vram}GB")
                        
                        return analysis_result
                except Exception as parse_error:
                    if progress_callback:
                        progress_callback("⚠️ JSON parsing failed, using fallback analysis")
                    pass
                
                # Fallback: return basic analysis if JSON parsing fails
                if progress_callback:
                    progress_callback("🔄 Applying fallback analysis...")
                return {
                    "workload_type": "unknown",
                    "complexity": "moderate",
                    "models_detected": [],
                    "estimated_vram_gb": 16,
                    "multi_gpu_required": False,
                    "memory_optimizations": [],
                    "batch_size_analysis": "Could not analyze",
                    "runtime_factors": [],
                    "confidence": 0.3,
                    "reasoning": ["LLM response parsing failed"]
                }
            else:
                if progress_callback:
                    progress_callback(f"❌ AI analysis failed (HTTP {response.status_code})")
                print(f"LLM API error: {response.status_code}")
                return None
                
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ AI analysis error: {str(e)[:50]}...")
            print(f"LLM analysis failed: {e}")
            return None
    
    def enhance_gpu_recommendation(self, static_analysis: Dict, llm_context: Dict) -> Tuple[Dict, List[str]]:
        """Enhance static analysis with LLM insights."""
        enhanced_analysis = static_analysis.copy()
        llm_reasoning = []
        
        # Check if LLM detected no workload that requires GPU
        llm_complexity = llm_context.get('complexity', '').lower()
        llm_workload_type = llm_context.get('workload_type', '').lower()
        
        # CRITICAL FIX: Handle CPU-only workloads detected by LLM
        if llm_workload_type == 'cpu-only':
            # LLM explicitly detected CPU-only workload - override with CPU-only
            enhanced_analysis.update({
                'min_gpu_type': 'CPU-only',
                'min_quantity': 0,
                'min_vram_gb': 0,
                'optimal_gpu_type': 'CPU-only',
                'optimal_quantity': 0,
                'optimal_vram_gb': 0,
                'min_runtime_estimate': 'CPU execution',
                'optimal_runtime_estimate': 'CPU execution',
                'workload_detected': False,
                'workload_type': 'cpu-only'
            })
            llm_reasoning.append("LLM analysis confirms CPU-only workload - no GPU needed")
            return enhanced_analysis, llm_reasoning
        elif llm_workload_type in ['gpu-computing', 'gpu_computing']:
            # This is a GPU computing workload (like Numba CUDA) - don't override
            llm_reasoning.append("LLM detected GPU computing workload - maintaining GPU recommendation")
        elif llm_complexity in ['simple', 'none', 'basic']:
            # Check if LLM reasoning indicates no GPU workload
            llm_reasoning_text = ' '.join(llm_context.get('reasoning', [])).lower()
            no_workload_indicators = [
                'no explicit models', 'no training', 'no inference', 'no gpu',
                'triggering compatibility warnings', 'no workload', 'no ml workload',
                'no evidence of training', 'no evidence of inference'
            ]
            
            # ENHANCED CHECK: Also look for positive GPU indicators
            gpu_computing_indicators = [
                'numba cuda', 'cuda kernel', 'gpu acceleration', 'cuda programming',
                'gpu computing', 'parallel processing', 'cuda.jit', 'cuda.grid'
            ]
            
            has_no_workload = any(indicator in llm_reasoning_text for indicator in no_workload_indicators)
            has_gpu_computing = any(indicator in llm_reasoning_text for indicator in gpu_computing_indicators)
            
            if has_no_workload and not has_gpu_computing:
                # LLM detected no GPU workload - override with CPU-only
                enhanced_analysis.update({
                    'min_gpu_type': 'CPU-only',
                    'min_quantity': 0,
                    'min_vram_gb': 0,
                    'optimal_gpu_type': 'CPU-only',
                    'optimal_quantity': 0,
                    'optimal_vram_gb': 0,
                    'min_runtime_estimate': 'N/A',
                    'optimal_runtime_estimate': 'N/A',
                    'workload_detected': False,
                    'workload_type': 'none'
                })
                llm_reasoning.append("LLM analysis confirms no GPU workload detected - CPU-only recommended")
                return enhanced_analysis, llm_reasoning
        
        # Respect the CPU-first analysis from static analysis
        if static_analysis.get('min_gpu_type') == 'CPU-only':
            llm_reasoning.append("Static analysis determined CPU-only workload - maintaining recommendation")
            return enhanced_analysis, llm_reasoning
        
        # Enhance VRAM estimate with LLM insights
        original_vram = static_analysis.get('min_vram_gb', 8)
        updated_vram = original_vram
        
        if 'estimated_vram_gb' in llm_context and llm_context['estimated_vram_gb']:
            llm_vram = llm_context['estimated_vram_gb']
            static_vram = static_analysis.get('min_vram_gb', 8)
            
            # Convert to per-GPU VRAM if multi-GPU setup
            quantity = static_analysis.get('optimal_quantity', 1)
            if quantity > 1:
                llm_vram = llm_vram // quantity
                static_vram = static_vram // quantity
            
            # Use higher estimate but cap at reasonable limits
            enhanced_vram = max(llm_vram, static_vram)
            updated_vram = min(enhanced_vram, 200)  # Cap at 200GB per GPU
            enhanced_analysis['min_vram_gb'] = updated_vram * quantity  # Store total VRAM
            
            if abs(llm_vram - static_vram) > 4:  # Significant difference
                llm_reasoning.append(f"LLM estimated {llm_vram}GB per GPU vs static analysis {static_vram}GB per GPU")
        
        # Enhance complexity analysis
        if 'complexity' in llm_context:
            enhanced_analysis['workload_complexity'] = llm_context['complexity']
            llm_reasoning.append(f"LLM identified workload complexity: {llm_context['complexity']}")
        
        # Add memory optimization insights
        if 'memory_optimizations' in llm_context and llm_context['memory_optimizations']:
            optimizations = llm_context['memory_optimizations']
            if optimizations:
                # More conservative reduction when LLM detected significantly higher VRAM requirement
                llm_vram = llm_context.get('estimated_vram_gb', 0)
                static_vram = static_analysis.get('min_vram_gb', 8)
                has_significant_llm_increase = llm_vram > static_vram + 8  # LLM found 8GB+ more than static
                
                if has_significant_llm_increase:
                    # Be more conservative with memory optimizations for high-VRAM workloads
                    # Fine-tuning large models still needs substantial memory even with optimizations
                    reduction_factor = 0.85 if len(optimizations) > 1 else 0.9
                    llm_reasoning.append(f"Conservative memory optimization applied due to high VRAM requirement ({llm_vram}GB)")
                else:
                    # Standard reduction for smaller workloads
                    reduction_factor = 0.7 if len(optimizations) > 1 else 0.85
                
                current_vram = enhanced_analysis.get('min_vram_gb', 8)
                updated_vram = max(8, int(current_vram * reduction_factor))  # Min 8GB for GPU workloads
                enhanced_analysis['min_vram_gb'] = updated_vram
                llm_reasoning.append(f"Memory optimizations detected: {', '.join(optimizations)}")
        
        # CRITICAL FIX: Re-evaluate GPU types if VRAM requirement changed significantly
        if abs(updated_vram - original_vram) > 4:  # Significant VRAM change
            llm_reasoning.append(f"Re-evaluating GPU recommendations based on updated VRAM requirement: {updated_vram}GB")
            
            # Check if SXM is required - if so, only recommend SXM GPUs
            sxm_required = enhanced_analysis.get('sxm_required', False)
            
            if sxm_required:
                # Get total system VRAM requirement
                quantity = enhanced_analysis.get('optimal_quantity', 1)
                total_vram_needed = updated_vram * quantity
                per_gpu_vram = updated_vram
                
                # Check if we already have a suitable SXM GPU selected
                current_min_gpu = enhanced_analysis.get('min_gpu_type', '')
                if current_min_gpu == 'A100 SXM 80G' and total_vram_needed <= 640:  # 8 × 80GB
                    # Keep A100 SXM if it's sufficient - no need to upgrade
                    enhanced_analysis['optimal_gpu_type'] = 'H100 SXM'
                    enhanced_analysis['optimal_vram_gb'] = max(updated_vram, 80)
                    llm_reasoning.append(f"Keeping A100 SXM 80G as minimum since total VRAM requirement ({total_vram_needed}GB) is within its capacity (640GB)")
                else:
                    # Select minimum GPU based on total system VRAM needs
                    if total_vram_needed <= 640:  # 8 × 80GB = 640GB total system VRAM
                        enhanced_analysis['min_gpu_type'] = 'A100 SXM 80G'
                        enhanced_analysis['min_vram_gb'] = max(per_gpu_vram, 80)
                        enhanced_analysis['optimal_gpu_type'] = 'H100 SXM'
                        enhanced_analysis['optimal_vram_gb'] = max(per_gpu_vram, 80)
                    elif total_vram_needed <= 1128:  # 8 × 141GB = 1128GB total system VRAM
                        enhanced_analysis['min_gpu_type'] = 'H100 SXM'
                        enhanced_analysis['min_vram_gb'] = max(per_gpu_vram, 80)
                        enhanced_analysis['optimal_gpu_type'] = 'H200 SXM'
                        enhanced_analysis['optimal_vram_gb'] = max(per_gpu_vram, 141)
                    else:  # Need more than 1128GB total system VRAM
                        enhanced_analysis['min_gpu_type'] = 'H200 SXM'
                        enhanced_analysis['min_vram_gb'] = max(per_gpu_vram, 141)
                        enhanced_analysis['optimal_gpu_type'] = 'B200 SXM'
                        enhanced_analysis['optimal_vram_gb'] = max(per_gpu_vram, 192)
                    
                    llm_reasoning.append(f"SXM GPUs selected due to multi-GPU requirement: {enhanced_analysis['min_gpu_type']} -> {enhanced_analysis['optimal_gpu_type']}")
            else:
                # Apply standard GPU selection logic for single-GPU workloads
                if updated_vram <= 8:
                    # Entry-level workload
                    enhanced_analysis['min_gpu_type'] = 'RTX 4060'
                    enhanced_analysis['min_vram_gb'] = 8  # RTX 4060 actual VRAM
                    enhanced_analysis['optimal_gpu_type'] = 'RTX 4070'
                    enhanced_analysis['optimal_vram_gb'] = 12  # RTX 4070 actual VRAM
                    enhanced_analysis['min_runtime_estimate'] = self._convert_runtime_to_new_format('1.0-2.0')
                    enhanced_analysis['optimal_runtime_estimate'] = self._convert_runtime_to_new_format('1.0-2.0')
                elif updated_vram <= 16:
                    # Mid-tier workload  
                    enhanced_analysis['min_gpu_type'] = 'RTX 4070'
                    enhanced_analysis['min_vram_gb'] = 12  # RTX 4070 actual VRAM
                    enhanced_analysis['optimal_gpu_type'] = 'RTX 4080'
                    enhanced_analysis['optimal_vram_gb'] = 16  # RTX 4080 actual VRAM
                    enhanced_analysis['min_runtime_estimate'] = self._convert_runtime_to_new_format('1.0-2.0')
                    enhanced_analysis['optimal_runtime_estimate'] = self._convert_runtime_to_new_format('1.0-2.0')
                elif updated_vram <= 24:
                    # High-end workload
                    enhanced_analysis['min_gpu_type'] = 'RTX 4090'
                    enhanced_analysis['min_vram_gb'] = 24  # RTX 4090 actual VRAM
                    enhanced_analysis['optimal_gpu_type'] = 'L4'
                    enhanced_analysis['optimal_vram_gb'] = 24  # L4 actual VRAM
                    enhanced_analysis['min_runtime_estimate'] = self._convert_runtime_to_new_format('1.0-2.0')
                    enhanced_analysis['optimal_runtime_estimate'] = self._convert_runtime_to_new_format('1.0-2.0')
                else:
                    # Enterprise workload - use PCIe GPUs for single-GPU high-VRAM needs
                    enhanced_analysis['min_gpu_type'] = 'L40S'
                    enhanced_analysis['min_vram_gb'] = 48  # L40S actual VRAM
                    enhanced_analysis['optimal_gpu_type'] = 'A100 PCIe 80G'
                    enhanced_analysis['optimal_vram_gb'] = 80  # A100 PCIe 80G actual VRAM
                    enhanced_analysis['min_runtime_estimate'] = self._convert_runtime_to_new_format('1.0-2.0')
                    enhanced_analysis['optimal_runtime_estimate'] = self._convert_runtime_to_new_format('1.0-2.0')
                
                llm_reasoning.append(f"Updated GPU recommendations: {enhanced_analysis['min_gpu_type']} (min) -> {enhanced_analysis['optimal_gpu_type']} (optimal)")
        
        # Multi-GPU insights
        if 'multi_gpu_required' in llm_context and llm_context['multi_gpu_required']:
            if not static_analysis.get('optimal_quantity', 1) > 1:
                enhanced_analysis['optimal_quantity'] = 2
                enhanced_analysis['sxm_required'] = True
                llm_reasoning.append("LLM detected multi-GPU requirements not caught by static analysis")
        
        # Extract and store LLM runtime data for use in recommendations
        if any(key in llm_context for key in ['baseline_runtime_hours', 'baseline_reference_gpu', 'optimization_speedup_factor']):
            enhanced_analysis['llm_runtime_data'] = {
                'baseline_runtime_hours': llm_context.get('baseline_runtime_hours', '2-3'),
                'baseline_reference_gpu': llm_context.get('baseline_reference_gpu', 'RTX 4090'),
                'optimization_speedup_factor': llm_context.get('optimization_speedup_factor', 1.0)
            }
            if 'runtime_reasoning' in llm_context:
                llm_reasoning.extend(llm_context['runtime_reasoning'])
        
        # Add LLM reasoning
        if 'reasoning' in llm_context:
            llm_reasoning.extend(llm_context['reasoning'])
        
        # Add LLM performance considerations
        if 'performance_considerations' in llm_context:
            llm_reasoning.extend(llm_context['performance_considerations'])
        
        return enhanced_analysis, llm_reasoning
    
    def self_review_analysis(self, code_cells: List[str], preliminary_analysis: Dict, static_reasoning: List[str], llm_reasoning: List[str], progress_callback=None) -> Optional[Dict]:
        """
        Phase 2.5: LLM self-review - the 'teacher grading' approach.
        Reviews the complete analysis for consistency, accuracy, and logical coherence.
        Optimized for Vercel Pro with smart complexity detection.
        """
        try:
            # Smart self-review: Skip if analysis is already high-confidence and simple
            if self.env_config.get('smart_self_review', False):
                confidence = preliminary_analysis.get('confidence', 0)
                complexity = preliminary_analysis.get('complexity', 'unknown')
                
                # Skip self-review for high-confidence, simple analyses to save time
                if confidence >= 0.85 and complexity in ['simple', 'basic']:
                    if progress_callback:
                        progress_callback("🎓 Self-review skipped: High confidence simple analysis")
                    return {
                        "review_passed": True,
                        "consistency_issues": [],
                        "recommended_corrections": {},
                        "unified_reasoning": ["Self-review skipped due to high confidence and simple complexity"],
                        "confidence_explanation": f"Analysis confidence {confidence*100:.0f}% deemed sufficient",
                        "overall_assessment": "Self-review bypassed for efficiency"
                    }
            
            if progress_callback:
                if self.env_config['detailed_phases']:
                    progress_callback("🎓 Phase 1: Preparing analysis for self-review...")
                else:
                    progress_callback("🎓 Performing accuracy self-review...")
            
            # Combine first few code cells for context (optimized for Vercel Pro)
            max_cells = 7 if self.env_config.get('environment_type') == 'vercel_pro' else 5
            notebook_sample = "\n".join([
                "=== NOTEBOOK SAMPLE ===",
                *code_cells[:max_cells]  # More cells for Pro plan
            ])
            
            max_sample_length = self.env_config.get('max_content_length', 8000)
            if len(notebook_sample) > max_sample_length:
                notebook_sample = notebook_sample[:max_sample_length] + "\n... [truncated]"
            
            if progress_callback:
                progress_callback("🔍 Phase 2: Analyzing current recommendations for consistency...")
            
            # Create comprehensive analysis summary for review
            analysis_summary = f"""
WORKLOAD ANALYSIS:
- Type: {preliminary_analysis.get('workload_type', 'unknown')}
- Detected: {preliminary_analysis.get('workload_detected', False)}

GPU RECOMMENDATIONS:
- Minimum: {preliminary_analysis.get('min_gpu_type', 'N/A')} ({preliminary_analysis.get('min_quantity', 0)}x, {preliminary_analysis.get('min_vram_gb', 0)}GB)
        - Recommended: {preliminary_analysis.get('consumer_gpu_type', 'N/A')} ({preliminary_analysis.get('consumer_quantity', 0)}x, {preliminary_analysis.get('consumer_vram_gb', 0)}GB)
        - Optimal: {preliminary_analysis.get('enterprise_gpu_type', 'N/A')} ({preliminary_analysis.get('enterprise_quantity', 0)}x, {preliminary_analysis.get('enterprise_vram_gb', 0)}GB)
        - Recommended Viable: {preliminary_analysis.get('consumer_viable', True)}

TECHNICAL DETAILS:
- SXM Required: {preliminary_analysis.get('sxm_required', False)}
- ARM Compatibility: {preliminary_analysis.get('arm_compatibility', 'Unknown')}
- Confidence: {preliminary_analysis.get('confidence', 0)*100:.0f}%

STATIC ANALYSIS REASONING:
{chr(10).join(f"• {reason}" for reason in static_reasoning[:5])}

LLM REASONING:
{chr(10).join(f"• {reason}" for reason in llm_reasoning[:5])}
"""

            prompt = f"""You are an expert reviewer evaluating GPU analysis for the 3-tier recommendation system. Act as a "teacher grading your work."

NOTEBOOK CODE SAMPLE:
{notebook_sample}

YOUR ANALYSIS RESULTS:
{analysis_summary}

**3-TIER SYSTEM REQUIREMENTS**:
- **CPU-Only Workloads**: If the workload is truly CPU-only (basic Python, simple data analysis, no ML/AI), all GPU fields should be "CPU-only" or null - this is CORRECT and should NOT be changed
- **GPU Workloads**: When GPU workload is detected, all three tiers must show actual GPU recommendations
- **Minimum**: Entry-level viable option (lowest cost that works)
- **Recommended**: Balanced price/performance (best value for most users)  
- **Optimal**: High performance option (best performance regardless of cost)
- **CRITICAL RULE**: For GPU workloads, all three tiers must show actual GPU recommendations
- **CRITICAL RULE**: For GPU workloads, never set recommended_viable=false - instead populate recommended tier with appropriate enterprise GPU

**CRITICAL REVIEW QUESTIONS**:
1. **CPU-Only vs GPU Workload**: Is this truly a CPU-only workload (basic Python, simple data analysis) or does it need GPU acceleration?
2. **3-Tier Logic**: For GPU workloads, are all three tiers properly populated with actual GPUs? For CPU-only workloads, should all tiers show "CPU-only"?
3. **Tier Progression**: Do the tiers follow logical progression (minimum ≤ recommended ≤ optimal in capability)?
4. **Workload Alignment**: Do the GPU recommendations match the detected workload complexity and requirements?
5. **Reasoning Consistency**: Are static analysis and LLM insights properly unified without contradictions?
6. **Confidence Calibration**: Does the confidence percentage match the certainty of evidence?
7. **Tutorial vs Production**: Is the workload correctly classified as tutorial/demo vs production use?

**SPECIFIC VALIDATION CHECKS**:
- **Tutorial Detection**: Small datasets (10 samples, example data) should get consumer GPU recommendations
- **Production Workloads**: Large-scale training/inference should get enterprise GPU recommendations
- **GPU Computing**: CUDA/scientific computing should be assessed for actual computational complexity
- **VRAM Logic**: Do VRAM estimates make sense for detected models/operations/batch sizes?
- **Runtime Realism**: Are runtime estimates realistic for the recommended hardware?
- **Performance Scaling**: Do higher tiers provide meaningful performance improvements?

**FORBIDDEN SCENARIOS FOR GPU WORKLOADS**:
- ❌ Recommended tier showing "N/A" or "Not Recommended" when GPU is needed
- ❌ Setting recommended_viable=false instead of showing enterprise GPU
- ❌ Identical GPUs across all three tiers
- ❌ Tier regression (optimal worse than recommended)
- ❌ Overestimating simple tutorials as enterprise workloads
- ❌ Underestimating production workloads as consumer-only

**ALLOWED FOR CPU-ONLY WORKLOADS**:
- ✅ All GPU tiers showing "CPU-only" or null values
- ✅ No GPU recommendations for basic Python operations
- ✅ Simple data analysis without ML/AI libraries

**VALID GPU OPTIONS** (use ONLY these):
- Consumer: RTX 4060, RTX 4070, RTX 4080, RTX 4090, RTX 5080, RTX 5090
- Enterprise: L4, L40, L40S, A100 PCIe 40G, A100 PCIe 80G, A100 SXM 40G, A100 SXM 80G, H100 PCIe, H100 SXM, H100 NVL, H200 SXM, H200 NVL, B200 SXM

**EXAMPLE CORRECT 3-TIER PATTERNS**:
- CPU-Only Workload: CPU-only → CPU-only → CPU-only
- Tutorial/Demo: RTX 4060 → RTX 4070 → RTX 4080
- Mid-scale Training: RTX 4080 → RTX 4090 → L40S
- Large-scale Training: RTX 4090 → L40S → A100 PCIe 80G
- Enterprise Production: L40S → A100 PCIe 80G → H100 PCIe

Respond in JSON format:
{{
    "review_passed": true/false,
    "consistency_issues": ["issue1", "issue2"],
    "recommended_corrections": {{
        "workload_type": "corrected_type_if_needed",
        "confidence": integer_between_0_and_100,
        "min_gpu_type": "valid_gpu_from_list_above_if_needed",
        "recommended_gpu_type": "valid_gpu_for_recommended_tier",
        "optimal_gpu_type": "valid_gpu_for_optimal_tier",
        "recommended_viable": true,
        "reasoning_updates": ["updated reasoning point 1", "updated reasoning point 2"]
    }},
    "unified_reasoning": ["clear unified reasoning point 1", "clear unified reasoning point 2"],
    "confidence_explanation": "why this confidence level is appropriate",
    "tier_validation": "assessment of 3-tier system compliance",
    "overall_assessment": "brief summary of analysis quality"
}}

**PRIORITY**: 
1. **First determine if this is truly CPU-only** (basic Python, simple data analysis, no ML/AI libraries)
2. **For CPU-only workloads**: Keep all GPU recommendations as "CPU-only" - DO NOT force GPU recommendations
3. **For GPU workloads**: Ensure all three tiers show actual GPU recommendations
4. **Fix any "Not Recommended" scenarios** by selecting appropriate enterprise GPUs for the recommended tier (only for GPU workloads)"""

            if progress_callback:
                progress_callback("🧠 Phase 3: Sending self-review request to AI...")
            
            # Use robust retry mechanism for API requests
            session = get_http_session()
            response = make_api_request_with_retry(
                session=session,
                url=f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json_data={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert reviewer specializing in GPU computing and machine learning workloads. Your job is to critically evaluate analysis results for accuracy and consistency."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": self._get_self_review_max_tokens()
                },
                timeout=self.env_config['llm_timeout'],
                max_retries=3,
                progress_callback=progress_callback
            )
            
            # Handle response or fallback
            if response is None:
                # All retries failed - return fallback self-review
                if progress_callback:
                    progress_callback("🔄 Self-review unavailable due to API connectivity issues")
                return {
                    "review_passed": True,
                    "consistency_issues": [],
                    "recommended_corrections": {},
                    "unified_reasoning": ["Self-review unavailable due to connectivity issues - using original analysis"],
                    "confidence_explanation": "API connectivity issues prevented self-review",
                    "overall_assessment": "Self-review skipped due to network issues"
                }
            
            if progress_callback:
                progress_callback("📥 Phase 4: Processing self-review response...")
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Enhanced debug logging for parsing issues
                print(f"🔍 DEBUG: Self-review response length: {len(content)}")
                # Save the problematic response to a file for analysis
                try:
                    with open('/tmp/self_review_response.txt', 'w') as f:
                        f.write(content)
                    print("🔍 DEBUG: Saved response to /tmp/self_review_response.txt")
                except:
                    pass
                
                if progress_callback:
                    progress_callback("🔍 Phase 5: Parsing self-review results...")
                
                try:
                    # Extract JSON from response
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        
                        # Clean JSON by removing comments and trailing characters
                        json_str = self._clean_json_response(json_str)
                        
                        review_result = json.loads(json_str)
                        
                        if progress_callback:
                            progress_callback("🎯 Phase 6: Analyzing self-review findings...")
                            
                            # Show what the review found
                            review_passed = review_result.get('review_passed', True)
                            consistency_issues = review_result.get('consistency_issues', [])
                            corrections = review_result.get('recommended_corrections', {})
                            
                            # Debug: Log what we got from self-review
                            print(f"🔍 DEBUG: Self-review results - passed: {review_passed}, issues: {len(consistency_issues)}, corrections: {len(corrections)}")
                            if consistency_issues:
                                print(f"🔍 DEBUG: Issues found: {consistency_issues}")
                            
                            if review_passed:
                                progress_callback("✅ Self-review passed - analysis is consistent")
                            else:
                                progress_callback("🔧 Self-review found issues requiring correction")
                                
                                # Show all specific issues found
                                print(f"🔍 DEBUG: About to display issues - consistency_issues type: {type(consistency_issues)}, length: {len(consistency_issues)}")
                                if consistency_issues:
                                    print(f"🔍 DEBUG: Displaying {len(consistency_issues)} issues via progress callback")
                                    progress_callback(f"📋 Found {len(consistency_issues)} specific issues:")
                                    for i, issue in enumerate(consistency_issues, 1):
                                        print(f"🔍 DEBUG: Displaying issue {i}: {issue}")
                                        progress_callback(f"⚠️ Issue {i}: {issue}")
                                else:
                                    print(f"🔍 DEBUG: No issues to display - consistency_issues is empty or falsy")
                                    progress_callback("⚠️ Self-review indicated issues but no specific issues were listed")
                            
                            # Show corrections being applied
                            if corrections:
                                progress_callback(f"🔧 Applying {len(corrections)} corrections:")
                                if 'workload_type' in corrections:
                                    progress_callback(f"🔧 Correcting workload type: {corrections['workload_type']}")
                                if 'confidence' in corrections:
                                    progress_callback(f"🔧 Adjusting confidence: {corrections['confidence']}%")
                                if 'min_gpu_type' in corrections:
                                    progress_callback(f"🔧 Updating GPU recommendation: {corrections['min_gpu_type']}")
                                if 'recommended_viable' in corrections:
                                    viable_status = "viable" if corrections['recommended_viable'] else "not viable"
                                    progress_callback(f"🔧 Correcting recommended GPU viability: {viable_status}")
                                if 'recommended_limitation' in corrections:
                                    progress_callback(f"🔧 Recommended GPU limitation: {corrections['recommended_limitation']}")
                                # Backward compatibility
                                if 'consumer_viable' in corrections:
                                    viable_status = "viable" if corrections['consumer_viable'] else "not viable"
                                    progress_callback(f"🔧 Correcting recommended GPU viability: {viable_status}")
                                if 'consumer_limitation' in corrections:
                                    progress_callback(f"🔧 Recommended GPU limitation: {corrections['consumer_limitation']}")
                                if corrections.get('reasoning_updates'):
                                    progress_callback(f"🔧 Updating reasoning with {len(corrections['reasoning_updates'])} points")
                            
                            # Show overall assessment
                            overall_assessment = review_result.get('overall_assessment', '')
                            if overall_assessment:
                                progress_callback(f"📊 Assessment: {overall_assessment}")
                        
                        return review_result
                except Exception as parse_error:
                    if progress_callback:
                        progress_callback("⚠️ Self-review parsing failed")
                    
                    # Enhanced error logging for debugging
                    print(f"🔍 Self-review parsing error: {parse_error}")
                    print(f"🔍 Raw response content (first 500 chars): {content[:500]}")
                    print(f"🔍 JSON extraction attempt - start: {json_start}, end: {json_end}")
                    if json_start != -1 and json_end > json_start:
                        extracted_json = content[json_start:json_end]
                        print(f"🔍 Extracted JSON (first 300 chars): {extracted_json[:300]}")
                    
                    # Try alternative JSON extraction methods
                    try:
                        # Method 1: Look for JSON blocks with backticks
                        if '```json' in content.lower():
                            start_marker = content.lower().find('```json')
                            if start_marker != -1:
                                json_block_start = start_marker + 7
                                json_block_end = content.find('```', json_block_start)
                                if json_block_end != -1:
                                    json_str = content[json_block_start:json_block_end].strip()
                                    review_result = json.loads(json_str)
                                    print("🔍 Successfully parsed JSON from code block")
                                    return review_result
                        
                        # Method 2: Try to find complete JSON anywhere in the response
                        import re
                        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                        json_matches = re.findall(json_pattern, content, re.DOTALL)
                        for match in json_matches:
                            try:
                                review_result = json.loads(match)
                                # Validate it has expected fields
                                if 'review_passed' in review_result or 'unified_reasoning' in review_result:
                                    print("🔍 Successfully parsed JSON using regex fallback")
                                    return review_result
                            except:
                                continue
                        
                        print("🔍 All JSON parsing methods failed")
                    except Exception as fallback_error:
                        print(f"🔍 Fallback parsing also failed: {fallback_error}")
                
                # Fallback if JSON parsing fails
                return {
                    "review_passed": True,
                    "consistency_issues": [],
                    "recommended_corrections": {},
                    "unified_reasoning": ["Self-review parsing failed - using original analysis"],
                    "confidence_explanation": "Could not parse self-review response",
                    "overall_assessment": "Self-review inconclusive"
                }
            else:
                print(f"Self-review API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Self-review failed: {e}")
            return None
    
    def apply_self_review_corrections(self, preliminary_analysis: Dict, static_reasoning: List[str], 
                                    llm_reasoning: List[str], self_review: Dict) -> Tuple[Dict, List[str]]:
        """
        Apply corrections from self-review to create the final unified analysis.
        """
        corrected_analysis = preliminary_analysis.copy()
        final_reasoning = []
        
        # Apply recommended corrections regardless of review status
        # Self-review can provide corrections even when the review "passes"
        corrections = self_review.get('recommended_corrections', {})
        if corrections:
            
            # Apply workload type correction
            if 'workload_type' in corrections:
                corrected_analysis['workload_type'] = corrections['workload_type']
            
            # Apply recommended viability corrections (with backward compatibility)
            recommended_viable_key = 'recommended_viable' if 'recommended_viable' in corrections else 'consumer_viable'
            if recommended_viable_key in corrections:
                corrected_analysis['consumer_viable'] = corrections[recommended_viable_key]
                
                # If recommended GPUs are now not viable, ensure minimum isn't a consumer card
                if not corrections[recommended_viable_key]:
                    current_min_gpu = corrected_analysis.get('min_gpu_type', '')
                    consumer_cards = ['RTX 4060', 'RTX 4070', 'RTX 4080', 'RTX 4090', 'RTX 5080', 'RTX 5090']
                    
                    if current_min_gpu in consumer_cards:
                        # Upgrade minimum to enterprise card
                        vram_needed = corrected_analysis.get('min_vram_gb', 16)
                        if vram_needed <= 24:
                            corrected_analysis['min_gpu_type'] = 'L4'
                            corrected_analysis['min_vram_gb'] = 24
                        elif vram_needed <= 48:
                            corrected_analysis['min_gpu_type'] = 'L40S'
                            corrected_analysis['min_vram_gb'] = 48
                        else:
                            corrected_analysis['min_gpu_type'] = 'A100 PCIe 80G'
                            corrected_analysis['min_vram_gb'] = 80
                        
                        final_reasoning.append(f"Corrected minimum GPU from {current_min_gpu} to {corrected_analysis['min_gpu_type']} since recommended GPUs are not viable")
            
            # Apply recommended limitation corrections (with backward compatibility)
            recommended_limitation_key = 'recommended_limitation' if 'recommended_limitation' in corrections else 'consumer_limitation'
            if recommended_limitation_key in corrections:
                corrected_analysis['consumer_limitation'] = corrections[recommended_limitation_key]
            
            # Apply confidence adjustment with validation
            if 'confidence' in corrections:
                confidence_value = corrections['confidence']
                # Ensure confidence is within valid range (0-100 as percentage, convert to 0.0-1.0)
                if isinstance(confidence_value, (int, float)):
                    if confidence_value > 1.0:  # Assume it's a percentage
                        confidence_value = min(max(confidence_value, 0), 100) / 100.0
                    else:  # Already a decimal
                        confidence_value = min(max(confidence_value, 0.0), 1.0)
                    corrected_analysis['confidence'] = confidence_value
                    final_reasoning.append(f"Self-review adjusted confidence to {confidence_value*100:.0f}%")
            
            # Apply GPU type correction with validation
            if 'min_gpu_type' in corrections:
                gpu_type = corrections['min_gpu_type']
                # Validate GPU type is in our specifications
                if gpu_type in self.gpu_specs:
                    corrected_analysis['min_gpu_type'] = gpu_type
                    # Regenerate consumer/enterprise recommendations if minimum changed
                    self._update_gpu_recommendations_after_correction(corrected_analysis)
                    final_reasoning.append(f"Self-review corrected minimum GPU to {gpu_type}")
                else:
                    final_reasoning.append(f"Self-review suggested invalid GPU '{gpu_type}' - keeping original recommendation")
        
        # Use unified reasoning from self-review if available
        unified_reasoning = self_review.get('unified_reasoning', [])
        if unified_reasoning:
            final_reasoning.extend(unified_reasoning)
        else:
            # Fallback to original reasoning if no unified reasoning provided
            final_reasoning.extend(llm_reasoning[:3])  # Prioritize LLM reasoning
            final_reasoning.extend(static_reasoning[:2])  # Add some static reasoning
        
        # Add self-review insights
        if self_review.get('consistency_issues'):
            final_reasoning.append(f"Self-review identified and corrected {len(self_review['consistency_issues'])} consistency issues")
        
        if self_review.get('confidence_explanation'):
            final_reasoning.append(f"Confidence assessment: {self_review['confidence_explanation']}")
        
        # Add review quality indicator
        review_quality = "passed" if self_review.get('review_passed', True) else "required corrections"
        final_reasoning.append(f"Analysis self-review {review_quality} - enhanced accuracy and consistency")
        
        return corrected_analysis, final_reasoning
    
    def _update_gpu_recommendations_after_correction(self, analysis: Dict):
        """
        Update consumer and enterprise recommendations after minimum GPU correction.
        This ensures logical hierarchy is maintained and VRAM/quantity/runtime match actual GPU specs.
        """
        min_gpu = analysis.get('min_gpu_type', '')
        
        # Get actual GPU specs for the corrected minimum GPU
        if min_gpu in self.gpu_specs:
            min_gpu_specs = self.gpu_specs[min_gpu]
            # Update minimum tier with actual GPU specs
            analysis['min_vram_gb'] = min_gpu_specs['vram']
            analysis['min_quantity'] = 1  # Default to 1 unless multi-GPU detected
            
            # Simple runtime estimation based on performance factor
            baseline_runtime = analysis.get('min_runtime_estimate', '30-60 minutes')
            performance_factor = min_gpu_specs.get('performance_factor', 1.0)
            if performance_factor < 0.5:
                analysis['min_runtime_estimate'] = '60-120 minutes'
            elif performance_factor < 1.0:
                analysis['min_runtime_estimate'] = '45-90 minutes'
            else:
                analysis['min_runtime_estimate'] = baseline_runtime
        
        # Update recommended and optimal tiers to maintain logical hierarchy
        if 'RTX 4060' in min_gpu:
            # Recommended tier: RTX 4070
            if 'RTX 4070' in self.gpu_specs:
                recommended_specs = self.gpu_specs['RTX 4070']
                analysis['consumer_gpu_type'] = 'RTX 4070'
                analysis['consumer_vram_gb'] = recommended_specs['vram']
                analysis['consumer_quantity'] = 1
                # Calculate runtime for RTX 4070
                perf_factor = recommended_specs.get('performance_factor', 1.0)
                if perf_factor < 0.5:
                    analysis['consumer_runtime_estimate'] = '45-90 minutes'
                elif perf_factor < 1.0:
                    analysis['consumer_runtime_estimate'] = '30-60 minutes'
                else:
                    analysis['consumer_runtime_estimate'] = '15-30 minutes'
            
            # Optimal tier: L40S
            if 'L40S' in self.gpu_specs:
                optimal_specs = self.gpu_specs['L40S']
                analysis['enterprise_gpu_type'] = 'L40S'
                analysis['enterprise_vram_gb'] = optimal_specs['vram']
                analysis['enterprise_quantity'] = 1
                analysis['enterprise_runtime_estimate'] = '5-15 minutes'
                
        elif 'RTX 4070' in min_gpu:
            # Recommended tier: RTX 4080
            if 'RTX 4080' in self.gpu_specs:
                recommended_specs = self.gpu_specs['RTX 4080']
                analysis['consumer_gpu_type'] = 'RTX 4080'
                analysis['consumer_vram_gb'] = recommended_specs['vram']
                analysis['consumer_quantity'] = 1
                analysis['consumer_runtime_estimate'] = '20-40 minutes'
            
            # Optimal tier: L40S
            if 'L40S' in self.gpu_specs:
                optimal_specs = self.gpu_specs['L40S']
                analysis['enterprise_gpu_type'] = 'L40S'
                analysis['enterprise_vram_gb'] = optimal_specs['vram']
                analysis['enterprise_quantity'] = 1
                analysis['enterprise_runtime_estimate'] = '5-15 minutes'
                
        elif 'RTX 4090' in min_gpu:
            # Recommended tier: RTX 4090 (same as minimum)
            if 'RTX 4090' in self.gpu_specs:
                recommended_specs = self.gpu_specs['RTX 4090']
                analysis['consumer_gpu_type'] = 'RTX 4090'
                analysis['consumer_vram_gb'] = recommended_specs['vram']
                analysis['consumer_quantity'] = 1
                analysis['consumer_runtime_estimate'] = '15-30 minutes'
            
            # Optimal tier: A100 PCIe
            if 'A100 PCIe 80G' in self.gpu_specs:
                optimal_specs = self.gpu_specs['A100 PCIe 80G']
                analysis['enterprise_gpu_type'] = 'A100 PCIe 80G'
                analysis['enterprise_vram_gb'] = optimal_specs['vram']
                analysis['enterprise_quantity'] = 1
                analysis['enterprise_runtime_estimate'] = '5-10 minutes'
        else:
            # Enterprise minimum - consumer might not be viable
            analysis['consumer_viable'] = False
            analysis['consumer_limitation'] = "Workload requires enterprise-grade hardware"
    
    def evaluate_notebook_compliance(self, code_cells: List[str], markdown_cells: List[str]) -> Optional[Dict]:
        """Evaluate notebook compliance with NVIDIA best practices using LLM and comprehensive guidelines."""
        try:
            # Combine first few cells for title/intro analysis
            intro_content = "\n".join(markdown_cells[:3]) if markdown_cells else ""
            structure_sample = "\n".join([
                "=== FIRST 3 MARKDOWN CELLS ===",
                intro_content,
                "\n=== NOTEBOOK STRUCTURE ===",
                f"Total markdown cells: {len(markdown_cells)}",
                f"Total code cells: {len(code_cells)}",
                "\n=== SAMPLE CODE CELLS ===",
                *code_cells[:2]  # First 2 code cells
            ])
            
            if len(structure_sample) > 8000:
                structure_sample = structure_sample[:8000] + "\n... [truncated]"
            
            # Get comprehensive guidelines for the prompt
            guidelines_text = self.best_practices.get_guidelines_for_evaluation()
            
            prompt = f"""Evaluate this Jupyter notebook against NVIDIA's comprehensive best practices for notebooks.

{guidelines_text}

Use the detailed scoring framework to evaluate:

**STRUCTURE & LAYOUT (25 points):**
- Title Quality (6.25 points): "Accomplish X with NVIDIA Product" format, clear, accessible
- Introduction Completeness (6.25 points): Target audience, overview, time estimate, tools, requirements
- Navigation (6.25 points): Proper markdown headers, logical flow, progress indicators
- Conclusion Quality (6.25 points): Summary, call-to-action, links to resources

**CONTENT QUALITY (25 points):**
- Documentation Ratio (8.33 points): Balanced markdown to code, adequate explanatory text
- Code Explanations (8.33 points): Code cells explained, clear I/O descriptions
- Educational Value (8.34 points): Clear objectives, practical content, professional writing

**TECHNICAL STANDARDS (25 points):**
- Requirements Management (6.25 points): requirements.txt implemented, version pinning
- Environment Variables (6.25 points): Proper handling, no hardcoded credentials
- Reproducibility (6.25 points): Seeds set, deterministic operations, debugging-friendly
- File Structure (6.25 points): Minimal complexity, documented structure

**NVIDIA COMPLIANCE (25 points):**
- Product Messaging (6.25 points): Proper NVIDIA references, consistent branding
- Brand Consistency (6.25 points): Professional presentation, unified voice
- Developer Focus (6.25 points): Clear value proposition, developer-centric
- Maintenance Quality (6.25 points): Well-structured, complete, clear guidelines

Notebook Content:
{structure_sample}

Respond in JSON format:
{{
    "structure_score": 0-25,
    "content_score": 0-25, 
    "technical_score": 0-25,
    "nvidia_score": 0-25,
    "total_score": 0-100,
    "structure_issues": ["specific issue 1", "specific issue 2"],
    "content_issues": ["specific issue 1", "specific issue 2"],
    "technical_issues": ["specific issue 1", "specific issue 2"],
    "nvidia_issues": ["specific issue 1", "specific issue 2"],
    "strengths": ["strength 1", "strength 2"],
    "recommendations": ["actionable rec 1", "actionable rec 2"],
    "confidence": 0.0-1.0,
    "detailed_analysis": {{
        "title_analysis": "detailed feedback on title",
        "introduction_analysis": "detailed feedback on introduction",
        "structure_analysis": "detailed feedback on structure",
        "technical_analysis": "detailed feedback on technical aspects"
    }}
}}"""

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert in technical documentation and NVIDIA's comprehensive content standards. Evaluate notebooks thoroughly against official best practices."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                try:
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        return json.loads(json_str)
                except:
                    pass
                    
            return None
            
        except Exception as e:
            print(f"LLM compliance evaluation failed: {e}")
            return None 


class GPUAnalyzer:
    def __init__(self, quiet_mode=False):
        self.quiet_mode = quiet_mode
        self.llm_analyzer = None
        
        # Load NVIDIA Best Practices
        self.best_practices = NVIDIABestPracticesLoader()
        
        # Enhanced GPU specifications with detailed metadata and categorization
        self.gpu_specs = {
            # RTX 50 Series (Consumer)
            'RTX 5090': {
                'vram': 32, 'compute_capability': 9.0, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2025, 'tier': 'flagship', 'tensor_cores': True,
                'category': 'consumer', 'max_reasonable_quantity': 2, 'performance_factor': 1.7,
                'cost_factor': 2.0  # ~$2000 (2x RTX 4060 baseline)
            },
            'RTX 5080': {
                'vram': 16, 'compute_capability': 9.0, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2025, 'tier': 'high', 'tensor_cores': True,
                'category': 'consumer', 'max_reasonable_quantity': 2, 'performance_factor': 0.85,
                'cost_factor': 1.2  # ~$1200 (1.2x RTX 4060 baseline)
            },
            
            # RTX 40 Series (Consumer)
            'RTX 4090': {
                'vram': 24, 'compute_capability': 8.9, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2022, 'tier': 'flagship', 'tensor_cores': True,
                'category': 'consumer', 'max_reasonable_quantity': 2, 'performance_factor': 1.0,
                'cost_factor': 1.6  # ~$1600 (baseline reference)
            },
            'RTX 4080': {
                'vram': 16, 'compute_capability': 8.9, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2022, 'tier': 'high', 'tensor_cores': True,
                'category': 'consumer', 'max_reasonable_quantity': 2, 'performance_factor': 0.70,
                'cost_factor': 1.2  # ~$1200
            },
            'RTX 4070': {
                'vram': 12, 'compute_capability': 8.9, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2023, 'tier': 'mid', 'tensor_cores': True,
                'category': 'consumer', 'max_reasonable_quantity': 2, 'performance_factor': 0.50,
                'cost_factor': 0.6  # ~$600
            },
            'RTX 4060': {
                'vram': 8, 'compute_capability': 8.9, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2023, 'tier': 'entry', 'tensor_cores': True,
                'category': 'consumer', 'max_reasonable_quantity': 2, 'performance_factor': 0.35,
                'cost_factor': 0.3  # ~$300 (cheapest baseline)
            },
            
            # Professional GPUs (Enterprise)
            'L4': {
                'vram': 24, 'compute_capability': 8.9, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2023, 'tier': 'mid', 'tensor_cores': True,
                'category': 'enterprise', 'max_reasonable_quantity': 8, 'performance_factor': 0.25,
                'cost_factor': 2.5  # ~$2500
            },
            'L40': {
                'vram': 48, 'compute_capability': 8.9, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2023, 'tier': 'high', 'tensor_cores': True,
                'category': 'enterprise', 'max_reasonable_quantity': 8, 'performance_factor': 0.65,
                'cost_factor': 7.0  # ~$7000
            },
            'L40S': {
                'vram': 48, 'compute_capability': 8.9, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2023, 'tier': 'high', 'tensor_cores': True,
                'category': 'enterprise', 'max_reasonable_quantity': 8, 'performance_factor': 0.75,
                'cost_factor': 8.0  # ~$8000
            },
            
            # Data Center GPUs (Enterprise)
            'A100 SXM 40G': {
                'vram': 40, 'compute_capability': 8.0, 'form_factor': 'SXM', 'nvlink': True,
                'release_year': 2020, 'tier': 'enterprise', 'tensor_cores': True,
                'category': 'enterprise', 'max_reasonable_quantity': 32, 'performance_factor': 1.2,
                'cost_factor': 10.0  # ~$10000
            },
            'A100 SXM 80G': {
                'vram': 80, 'compute_capability': 8.0, 'form_factor': 'SXM', 'nvlink': True,
                'release_year': 2020, 'tier': 'enterprise', 'tensor_cores': True,
                'category': 'enterprise', 'max_reasonable_quantity': 32, 'performance_factor': 1.2,
                'cost_factor': 15.0  # ~$15000
            },
            'A100 PCIe 40G': {
                'vram': 40, 'compute_capability': 8.0, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2020, 'tier': 'enterprise', 'tensor_cores': True,
                'category': 'enterprise', 'max_reasonable_quantity': 8, 'performance_factor': 1.0,
                'cost_factor': 8.0  # ~$8000
            },
            'A100 PCIe 80G': {
                'vram': 80, 'compute_capability': 8.0, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2020, 'tier': 'enterprise', 'tensor_cores': True,
                'category': 'enterprise', 'max_reasonable_quantity': 8, 'performance_factor': 1.0,
                'cost_factor': 12.0  # ~$12000
            },
            'H100 SXM': {
                'vram': 80, 'compute_capability': 9.0, 'form_factor': 'SXM', 'nvlink': True,
                'release_year': 2022, 'tier': 'cutting_edge', 'tensor_cores': True,
                'category': 'enterprise', 'max_reasonable_quantity': 32, 'performance_factor': 2.5,
                'cost_factor': 25.0  # ~$25000
            },
            'H100 PCIe': {
                'vram': 80, 'compute_capability': 9.0, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2022, 'tier': 'cutting_edge', 'tensor_cores': True,
                'category': 'enterprise', 'max_reasonable_quantity': 8, 'performance_factor': 2.2,
                'cost_factor': 20.0  # ~$20000
            },
            'H200 SXM': {
                'vram': 141, 'compute_capability': 9.0, 'form_factor': 'SXM', 'nvlink': True,
                'release_year': 2023, 'tier': 'cutting_edge', 'tensor_cores': True,
                'category': 'enterprise', 'max_reasonable_quantity': 32, 'performance_factor': 3.0,
                'cost_factor': 35.0  # ~$35000
            },
            'H100 NVL': {
                'vram': 94, 'compute_capability': 9.0, 'form_factor': 'NVL', 'nvlink': True,
                'release_year': 2022, 'tier': 'cutting_edge', 'tensor_cores': True,
                'category': 'enterprise', 'max_reasonable_quantity': 8, 'performance_factor': 2.3,
                'cost_factor': 22.0  # ~$22000
            },
            'H200 NVL': {
                'vram': 141, 'compute_capability': 9.0, 'form_factor': 'NVL', 'nvlink': True,
                'release_year': 2023, 'tier': 'cutting_edge', 'tensor_cores': True,
                'category': 'enterprise', 'max_reasonable_quantity': 8, 'performance_factor': 2.8,
                'cost_factor': 32.0  # ~$32000
            },
            'B200 SXM': {
                'vram': 192, 'compute_capability': 10.0, 'form_factor': 'SXM', 'nvlink': True,
                'release_year': 2024, 'tier': 'cutting_edge', 'tensor_cores': True,
                'category': 'enterprise', 'max_reasonable_quantity': 32, 'performance_factor': 4.0,
                'cost_factor': 50.0  # ~$50000+ (estimated)
            }
        }
        
        # Initialize LLM if API key is available
        openai_api_key = os.getenv('OPENAI_API_KEY')
        openai_base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com')
        openai_model = os.getenv('OPENAI_MODEL', 'gpt-4')
        
        if openai_api_key:
            self.llm_analyzer = LLMAnalyzer(
                base_url=openai_base_url,
                model=openai_model,
                api_key=openai_api_key,
                gpu_specs=self.gpu_specs
            )
            if not self.quiet_mode:
                if 'nvidia.com' in openai_base_url:
                    print(f"✅ LLM enhancement enabled using NVIDIA API with {openai_model}")
                else:
                    print(f"✅ LLM enhancement enabled using {openai_model}")
        else:
            if not self.quiet_mode:
                print("⚠️ No LLM API key found. Analysis will use static patterns only.")
        
        # Enhanced model specifications with comprehensive VRAM estimates
        self.model_specs = {
            'bert': {'base_vram': 2, 'param_multiplier': 0.1},
            'gpt': {'base_vram': 4, 'param_multiplier': 0.2},
            'llama': {'base_vram': 8, 'param_multiplier': 0.25},
            'mistral': {'base_vram': 6, 'param_multiplier': 0.2},
            'gemma': {'base_vram': 4, 'param_multiplier': 0.15},
            'phi': {'base_vram': 3, 'param_multiplier': 0.1},
            'resnet': {'base_vram': 4, 'param_multiplier': 0.05},
            'efficientnet': {'base_vram': 3, 'param_multiplier': 0.03},
            'vit': {'base_vram': 6, 'param_multiplier': 0.08},
            'clip': {'base_vram': 8, 'param_multiplier': 0.1},
            'stable.diffusion': {'base_vram': 12, 'param_multiplier': 0.15},
            'whisper': {'base_vram': 4, 'param_multiplier': 0.1}
        }
        
        # Enhanced framework analysis patterns
        self.framework_patterns = {
            'pytorch': [r'torch\.', r'import torch', r'from torch'],
            'tensorflow': [r'tensorflow', r'import tf', r'tf\.'],
            'jax': [r'import jax', r'jax\.'],
            'transformers': [r'from transformers', r'transformers\.'],
            'diffusers': [r'from diffusers', r'diffusers\.'],
            'accelerate': [r'from accelerate', r'accelerate\.'],
            'peft': [r'from peft', r'peft\.', r'lora'],
            'quantization': [r'quantiz', r'int8', r'int4', r'bnb', r'bitsandbytes'],
            'cudf': [r'cudf\.', r'import cudf', r'from cudf', r'backend=["\']cudf["\']', r'engine=["\']cudf["\']'],
            'rapids': [r'cupy\.', r'import cupy', r'from cupy', r'cuml\.', r'import cuml', r'from cuml', r'cugraph\.', r'import cugraph'],
            'cuda_computing': [r'numba\.cuda', r'from numba import cuda', r'@cuda\.jit', r'cuda\.grid', r'cuda\.', r'pycuda\.', r'import pycuda'],
            'nvidia_frameworks': [r'nemo\.', r'import nemo', r'triton\.', r'import triton', r'tensorrt\.', r'import tensorrt']
        }
        
        # Multi-GPU patterns (can work with PCIe + NVLink)
        self.multi_gpu_patterns = [
            # Standard PyTorch multi-GPU (works fine with PCIe)
            r'\btorch\.nn\.DataParallel\b', r'\btorch\.nn\.parallel\.DistributedDataParallel\b',
            r'\bmodel\s*=\s*nn\.DataParallel\b', r'\bmodel\s*=\s*DDP\b',
            r'\baccelerator\s*=\s*["\']ddp["\']', r'\bstrategy\s*=\s*["\']ddp["\']',
            
            # GPU availability checks (not necessarily SXM)
            r'\btorch\.cuda\.device_count\(\)\s*>\s*1', r'\bavailable_gpus\s*>\s*1',
            r'\bif.*torch\.cuda\.device_count', r'\bgpu_count\s*>\s*1',
            
            # Small-scale distributed training (PCIe friendly)
            r'\bworld_size\s*[=<]\s*[2-4]', r'\bnum_gpus\s*[=<]\s*[2-4]',
            r'\bdist\.init_process_group\b', r'\ball_reduce\b', r'\ball_gather\b',
            
            # Framework multi-GPU (usually works with PCIe)
            r'\bhorovod\b', r'\baccelerate.*multi_gpu',
            r'\btransformers.*DataParallel', r'\blightning.*gpus\s*=\s*[2-4]'
        ]
        
        # SXM requirement patterns (enhanced with more precise detection for real SXM needs)
        self.sxm_patterns = [
            # Large-scale distributed training (real SXM needs)
            r'\bnum_nodes\s*[>=]\s*[2-9]', r'\bworld_size\s*[>=]\s*8',
            r'\bgpus_per_node\s*[>=]\s*8', r'\btotal_gpus\s*[>=]\s*8',
            
            # Multi-node distributed training
            r'\btorch\.distributed\.launch.*--nnodes\s+[2-9]',
            r'\btorchrun.*--nnodes\s+[2-9]', r'\bmpirun.*-np\s+[8-9]',
            r'\bdist\.init_process_group.*nccl.*rank',
            
            # Explicit SXM/NVLink requirements
            r'\bnvlink.*required', r'\bsxm.*required', r'\bnvswitch',
            r'\bdgx.*system', r'\bsuperpod', r'\bbasepod',
            
            # Large model parallelism requiring SXM bandwidth
            r'\btensor_model_parallel_size\s*[>=]\s*4',
            r'\bpipeline_model_parallel_size\s*[>=]\s*4',
            r'\bmodel_parallel.*degree\s*[>=]\s*4',
            
            # High-power training workloads (likely need SXM power limits)
            r'\bfull.*precision.*training', r'\bbf16.*training.*large',
            r'\bmixed.*precision.*false', r'\bfp32.*training',
            
            # Enterprise/data center specific patterns
            r'\bslurm.*--gres=gpu:[8-9]', r'\bsbatch.*--gres=gpu:[8-9]',
            r'\bpbs.*-l.*gpus=[8-9]', r'\btorque.*gpus=[8-9]',
            
            # Framework-specific large-scale patterns
            r'\bdeepspeed.*zero_stage.*3.*offload_param',
            r'\bfairscale.*fully_sharded_data_parallel',
            r'\btransformers.*trainer.*dataparallel_process_group_size\s*[>=]\s*8'
        ]
        
        # ARM/Grace compatibility patterns - Enhanced for comprehensive analysis
        self.arm_compatible_frameworks = [
            'tensorflow', 'pytorch', 'torch', 'jax', 'cupy', 'rapids',
            'transformers', 'diffusers', 'accelerate', 'peft', 'numpy',
            'scipy', 'scikit-learn', 'pandas', 'matplotlib', 'seaborn',
            'opencv-python', 'pillow', 'numba', 'dask',
            'cudf', 'cuml', 'cugraph', 'nemo', 'triton'
        ]
        
        # Enhanced ARM incompatible patterns with more comprehensive detection
        self.arm_incompatible_patterns = [
            # Legacy CUDA/cuDNN versions
            r'cudnn.*version.*<.*8', r'tensorrt.*<.*8', r'triton.*kernel',
            
            # Intel-specific optimizations
            r'intel.*mkl', r'mkl.*intel', r'oneapi', r'intel.*optimization',
            r'mkldnn', r'onednn.*intel', r'intel.*threading',
            
            # Legacy/problematic libraries
            r'apex\.', r'flash.attention.*<.*2', r'xformers.*<.*0\.2',
            r'deepspeed.*<.*0\.7', r'bitsandbytes.*<.*0\.3',
            
            # Architecture-specific optimizations
            r'x86.*specific', r'sse[0-9]*', r'avx[0-9]*', r'fma.*instruction',
            r'intel.*compiler', r'icc.*compiler',
            
            # Legacy TensorFlow/PyTorch patterns
            r'tensorflow.*<.*2\.8', r'torch.*<.*1\.12', r'tf\.compat\.v1',
            
            # Proprietary/closed-source with limited ARM support
            r'tensorrt.*<.*8\.4', r'cuda.*<.*11\.4', r'nccl.*<.*2\.10',
            
            # Specific problematic operations
            r'torch\.jit\.script.*@.*torch\.jit\.ignore', r'tf\.function.*experimental_relax_shapes.*False',
            
            # Legacy quantization libraries
            r'fbgemm', r'qnnpack.*x86', r'pytorch_quantization.*<.*2\.0'
        ]
        
        # ARM optimization indicators (positive signals)
        self.arm_optimization_patterns = [
            r'aarch64', r'arm64', r'neon.*optimization', r'grace.*optimization',
            r'nvidia.*grace', r'arm.*specific', r'cross.*platform',
            r'architecture.*agnostic', r'multi.*arch', r'universal.*binary'
        ]
        
        # Version-specific ARM compatibility database
        self.arm_compatibility_versions = {
            'tensorflow': {'min_compatible': '2.8.0', 'optimal': '2.12.0'},
            'torch': {'min_compatible': '1.12.0', 'optimal': '2.0.0'},
            'torchvision': {'min_compatible': '0.13.0', 'optimal': '0.15.0'},
            'transformers': {'min_compatible': '4.18.0', 'optimal': '4.25.0'},
            'diffusers': {'min_compatible': '0.14.0', 'optimal': '0.20.0'},
            'accelerate': {'min_compatible': '0.16.0', 'optimal': '0.21.0'},
            'bitsandbytes': {'min_compatible': '0.37.0', 'optimal': '0.41.0'},
            'flash-attn': {'min_compatible': '2.0.0', 'optimal': '2.3.0'},
            'xformers': {'min_compatible': '0.0.20', 'optimal': '0.0.22'},
            'deepspeed': {'min_compatible': '0.9.0', 'optimal': '0.10.0'}
        }
    
    def evaluate_notebook_structure(self, code_cells: List[str], markdown_cells: List[str]) -> Dict[str, str]:
        """Enhanced notebook structure evaluation using comprehensive NVIDIA guidelines."""
        assessment = {}
        
        # Enhanced title analysis using NVIDIA Best Practices
        if markdown_cells:
            first_cell = markdown_cells[0].lower()
            
            # Check for proper title format based on guidelines
            if re.search(r'(modeling|building|training|deploying|analyzing|processing|optimizing).*with.*nvidia', first_cell):
                assessment['title'] = "✅ Excellent title format (follows 'doing X with NVIDIA Product')"
            elif 'with nvidia' in first_cell or 'using nvidia' in first_cell:
                assessment['title'] = "✅ Good title format"
            elif 'nvidia' in first_cell:
                assessment['title'] = "⚠️ Title mentions NVIDIA but could follow 'doing X with NVIDIA Product' format"
            else:
                assessment['title'] = "❌ Title should mention NVIDIA products and follow best practice format"
        else:
            assessment['title'] = "❌ No title found"
        
        # Enhanced introduction analysis
        intro_content = ' '.join(markdown_cells[:3]).lower() if len(markdown_cells) >= 3 else ''
        intro_score = 0
        intro_elements = []
        
        # Check for comprehensive introduction elements
        intro_checks = [
            (['audience', 'for', 'target', 'developer', 'persona'], "target audience"),
            (['overview', 'learn', 'will', 'tutorial', 'guide'], "overview/learning outcomes"),
            (['minutes', 'hours', 'time', 'complete', 'duration'], "time estimate"),
            (['requirements', 'prerequisites', 'hardware', 'software'], "requirements"),
            (['nvidia', 'gpu', 'cuda', 'tensor'], "NVIDIA tools"),
            (['data', 'dataset', 'input', 'model'], "data/model requirements")
        ]
        
        for keywords, element in intro_checks:
            if any(word in intro_content for word in keywords):
                intro_score += 1
                intro_elements.append(element)
        
        if intro_score >= 5:
            assessment['introduction'] = f"✅ Comprehensive introduction ({', '.join(intro_elements)})"
        elif intro_score >= 3:
            assessment['introduction'] = f"✅ Good introduction ({', '.join(intro_elements)})"
        elif intro_score >= 2:
            assessment['introduction'] = f"⚠️ Basic introduction present - could be enhanced"
        else:
            assessment['introduction'] = "❌ Missing key introduction elements per NVIDIA guidelines"
        
        # Enhanced header/navigation analysis
        header_count = sum(1 for cell in markdown_cells if re.search(r'^#+\s', cell, re.MULTILINE))
        total_cells = len(code_cells) + len(markdown_cells)
        header_ratio = header_count / max(total_cells, 1)
        
        if header_count >= 5 and header_ratio > 0.15:
            assessment['navigation'] = "✅ Excellent navigation structure with comprehensive headers"
        elif header_count >= 3:
            assessment['navigation'] = "✅ Good use of headers for navigation"
        elif header_count >= 1:
            assessment['navigation'] = "⚠️ Some headers present, could use more for better navigation"
        else:
            assessment['navigation'] = "❌ Missing headers for navigation - critical for user experience"
        
        # Enhanced conclusion analysis
        if markdown_cells:
            last_cells = ' '.join(markdown_cells[-3:]).lower()
            conclusion_elements = []
            
            if any(word in last_cells for word in ['summary', 'conclusion', 'learned', 'key takeaways']):
                conclusion_elements.append("summary")
            if any(word in last_cells for word in ['next steps', 'further reading', 'additional resources']):
                conclusion_elements.append("next steps")
            if re.search(r'http[s]?://', ' '.join(markdown_cells[-3:])):
                conclusion_elements.append("resource links")
            if any(word in last_cells for word in ['notebook', 'tutorial', 'guide', 'documentation']):
                conclusion_elements.append("related content")
            
            if len(conclusion_elements) >= 3:
                assessment['conclusion'] = f"✅ Comprehensive conclusion ({', '.join(conclusion_elements)})"
            elif len(conclusion_elements) >= 2:
                assessment['conclusion'] = f"✅ Good conclusion ({', '.join(conclusion_elements)})"
            elif len(conclusion_elements) >= 1:
                assessment['conclusion'] = "⚠️ Basic conclusion present - could be enhanced"
            else:
                assessment['conclusion'] = "❌ Missing proper conclusion with summary and next steps"
        else:
            assessment['conclusion'] = "❌ No conclusion found"
        
        return assessment
    
    def assess_content_quality(self, code_cells: List[str], markdown_cells: List[str]) -> List[str]:
        """Enhanced content quality assessment using NVIDIA Best Practices."""
        issues = []
        
        # Enhanced documentation ratio analysis
        total_cells = len(code_cells) + len(markdown_cells)
        if total_cells > 0:
            markdown_ratio = len(markdown_cells) / total_cells
            
            # Calculate content depth
            avg_markdown_length = sum(len(cell) for cell in markdown_cells) / max(len(markdown_cells), 1)
            avg_code_length = sum(len(cell) for cell in code_cells) / max(len(code_cells), 1)
            
            if markdown_ratio < 0.25:
                issues.append("Low documentation ratio - NVIDIA guidelines recommend substantial explanatory text")
            elif markdown_ratio > 0.75 and avg_code_length < 100:
                issues.append("High documentation ratio with minimal code - ensure sufficient practical examples")
            
            if avg_markdown_length < 50:
                issues.append("Markdown cells are very brief - consider more detailed explanations")
        
        # Enhanced code explanation analysis
        explained_code_blocks = 0
        for i, code_cell in enumerate(code_cells):
            has_explanation = False
            
            # Check if code cell is explained by surrounding markdown
            if i > 0 and i-1 < len(markdown_cells):
                prev_markdown = markdown_cells[i-1].lower()
                if any(word in prev_markdown for word in ['code', 'cell', 'execute', 'run', 'implement']):
                    has_explanation = True
            
            if i < len(markdown_cells):
                next_markdown = markdown_cells[i].lower()
                if any(word in next_markdown for word in ['output', 'result', 'shows', 'demonstrates']):
                    has_explanation = True
            
            if has_explanation:
                explained_code_blocks += 1
        
        if code_cells:
            explanation_ratio = explained_code_blocks / len(code_cells)
            if explanation_ratio < 0.4:
                issues.append("Many code cells lack explanatory text - NVIDIA guidelines require clear explanations")
            elif explanation_ratio < 0.6:
                issues.append("Some code cells could benefit from more detailed explanations")
        
        # Enhanced link and resource analysis
        all_markdown = ' '.join(markdown_cells)
        
        # Count different types of links
        external_links = len(re.findall(r'\[.*?\]\(https?://.*?\)', all_markdown))
        nvidia_links = len(re.findall(r'nvidia\.com|developer\.nvidia\.com', all_markdown, re.IGNORECASE))
        doc_links = len(re.findall(r'docs?\.|documentation|api.*reference', all_markdown, re.IGNORECASE))
        
        if external_links == 0:
            issues.append("No external resource links found - NVIDIA guidelines recommend linking to relevant documentation")
        elif external_links < 2:
            issues.append("Very few external links - consider adding more references to NVIDIA docs and related resources")
        
        if nvidia_links == 0:
            issues.append("No links to NVIDIA resources found - should reference relevant NVIDIA documentation")
        
        # Enhanced educational value assessment
        educational_indicators = 0
        educational_keywords = [
            'learn', 'understand', 'demonstrate', 'example', 'tutorial',
            'guide', 'step', 'process', 'workflow', 'best practice'
        ]
        
        for keyword in educational_keywords:
            if keyword in all_markdown.lower():
                educational_indicators += 1
        
        if educational_indicators < 3:
            issues.append("Limited educational indicators - enhance learning-focused language and explanations")
        
        # Professional writing assessment
        if markdown_cells:
            incomplete_sentences = 0
            short_explanations = 0
            
            for cell in markdown_cells:
                sentences = re.split(r'[.!?]+', cell)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        if len(sentence) > 10 and not sentence[0].isupper():
                            incomplete_sentences += 1
                        if len(sentence) < 20:
                            short_explanations += 1
            
            if incomplete_sentences > len(markdown_cells) * 0.2:
                issues.append("Some text may not follow proper sentence structure - review capitalization and grammar")
            
            if short_explanations > len(markdown_cells) * 0.4:
                issues.append("Many explanations are very brief - consider more detailed descriptions for better understanding")
        
        return issues
    
    def check_technical_standards(self, code_cells: List[str]) -> List[str]:
        """Enhanced technical standards checking using NVIDIA Best Practices."""
        recommendations = []
        all_code = '\n'.join(code_cells)
        
        # Enhanced requirements.txt analysis
        requirements_mentioned = bool(re.search(r'requirements\.txt', all_code, re.IGNORECASE))
        pip_install_found = bool(re.search(r'pip install|!pip', all_code, re.IGNORECASE))
        
        if not requirements_mentioned and pip_install_found:
            recommendations.append("Create and reference requirements.txt file - NVIDIA standard requires centralized dependency management")
        elif not requirements_mentioned:
            recommendations.append("Add requirements.txt file installation - critical for reproducibility")
        
        # Enhanced version pinning analysis
        pip_install_lines = re.findall(r'(?:pip install|!pip install).*', all_code, re.IGNORECASE)
        unpinned_packages = []
        
        for line in pip_install_lines:
            if 'requirements.txt' not in line:
                # Extract package names and check for version pinning
                packages = re.findall(r'\b([a-zA-Z0-9_-]+)(?![=<>])', line.replace('pip install', '').replace('!pip install', ''))
                for pkg in packages[:2]:  # Limit to avoid spam
                    if pkg not in ['pip', 'install', 'upgrade'] and len(pkg) > 2:
                        unpinned_packages.append(pkg)
        
        if unpinned_packages:
            recommendations.append(f"Pin package versions for reproducibility (e.g., {unpinned_packages[0]}==1.2.3) - NVIDIA requirement")
        
        # Enhanced environment variable analysis
        hardcoded_credentials = []
        env_var_usage = []
        
        credential_patterns = [
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'API keys'),
            (r'token\s*=\s*["\'][^"\']+["\']', 'tokens'),
            (r'password\s*=\s*["\'][^"\']+["\']', 'passwords'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'secrets')
        ]
        
        for pattern, cred_type in credential_patterns:
            if re.search(pattern, all_code, re.IGNORECASE):
                hardcoded_credentials.append(cred_type)
        
        env_patterns = [r'os\.environ\[', r'os\.getenv\(', r'getenv\(']
        for pattern in env_patterns:
            if re.search(pattern, all_code):
                env_var_usage.append(pattern)
        
        if hardcoded_credentials:
            recommendations.append(f"Remove hardcoded {', '.join(hardcoded_credentials)} - use environment variables per NVIDIA security guidelines")
        elif not env_var_usage and any(word in all_code.lower() for word in ['api', 'key', 'token', 'auth']):
            recommendations.append("Consider using environment variables for configuration - NVIDIA best practice for security")
        
        # Enhanced reproducibility analysis
        reproducibility_elements = []
        
        # Check for various types of seeds
        seed_patterns = [
            (r'(?:torch\.manual_seed|random\.seed|np\.random\.seed)', 'random seeds'),
            (r'random_state\s*=\s*\d+', 'sklearn random_state'),
            (r'seed\s*=\s*\d+', 'explicit seed parameters'),
            (r'deterministic\s*=\s*True', 'deterministic settings')
        ]
        
        for pattern, element in seed_patterns:
            if re.search(pattern, all_code, re.IGNORECASE):
                reproducibility_elements.append(element)
        
        ml_frameworks = any(lib in all_code.lower() for lib in ['torch', 'tensorflow', 'sklearn', 'numpy', 'random'])
        
        if ml_frameworks and not reproducibility_elements:
            recommendations.append("Set random seeds for reproducibility - critical for NVIDIA notebooks")
        elif ml_frameworks and len(reproducibility_elements) < 2:
            recommendations.append("Consider setting additional random seeds (torch, numpy, python random) for full reproducibility")
        
        # Enhanced file complexity analysis
        file_references = []
        complex_patterns = [
            (r'\.py["\']', 'Python files'),
            (r'\.json["\']', 'JSON files'),
            (r'\.yaml["\']|\.yml["\']', 'YAML files'),
            (r'\.csv["\']', 'CSV files'),
            (r'\.pkl["\']|\.pickle["\']', 'Pickle files'),
            (r'\.h5["\']|\.hdf5["\']', 'HDF5 files')
        ]
        
        for pattern, file_type in complex_patterns:
            matches = len(re.findall(pattern, all_code))
            if matches > 0:
                file_references.append(f"{matches} {file_type}")
        
        total_external_files = sum(int(ref.split()[0]) for ref in file_references)
        
        if total_external_files > 8:
            recommendations.append("High number of external file dependencies - consider simplifying or documenting file structure")
        elif total_external_files > 4:
            recommendations.append("Multiple external files detected - ensure file structure is documented per NVIDIA guidelines")
        
        # GPU-specific technical standards
        gpu_code_patterns = [
            (r'\.cuda\(\)|\.to\(["\']cuda["\']', 'CUDA device usage'),
            (r'torch\.cuda\.|cuda\.', 'CUDA operations'),
            (r'mixed.precision|autocast', 'mixed precision'),
            (r'gradient.*checkpoint', 'gradient checkpointing')
        ]
        
        gpu_optimizations = []
        for pattern, optimization in gpu_code_patterns:
            if re.search(pattern, all_code, re.IGNORECASE):
                gpu_optimizations.append(optimization)
        
        if 'cuda' in all_code.lower() and 'mixed precision' not in [opt.lower() for opt in gpu_optimizations]:
            recommendations.append("Consider mixed precision training for better GPU utilization - NVIDIA best practice")
        
        return recommendations
    
    def calculate_nvidia_compliance_score(self, structure_assessment: Dict, content_issues: List, 
                                        technical_recommendations: List, llm_evaluation: Optional[Dict] = None) -> float:
        """Enhanced NVIDIA compliance scoring using comprehensive best practices."""
        
        # If LLM evaluation is available, use it as primary score with enhancements
        if llm_evaluation and 'total_score' in llm_evaluation:
            base_score = llm_evaluation['total_score']
            
            # Apply additional checks based on loaded best practices
            if self.best_practices.scoring_framework:
                # Enhance score based on detailed criteria
                bonus_points = 0
                
                # Structure bonuses
                if 'Excellent title format' in str(structure_assessment.get('title', '')):
                    bonus_points += 2
                if 'Comprehensive introduction' in str(structure_assessment.get('introduction', '')):
                    bonus_points += 2
                if 'Excellent navigation' in str(structure_assessment.get('navigation', '')):
                    bonus_points += 1
                
                # Content quality bonuses
                if len(content_issues) == 0:
                    bonus_points += 3
                elif len(content_issues) <= 2:
                    bonus_points += 1
                
                # Technical standards bonuses
                if len(technical_recommendations) == 0:
                    bonus_points += 3
                elif len(technical_recommendations) <= 2:
                    bonus_points += 1
                
                # Apply bonus but cap at 100
                enhanced_score = min(base_score + bonus_points, 100)
                return enhanced_score
            
            return min(base_score, 100)
        
        # Enhanced static scoring using comprehensive criteria
        scores = {
            'structure': 0.0,
            'content': 0.0,
            'technical': 0.0,
            'nvidia': 0.0
        }
        
        # Structure & Layout scoring (25 points max)
        structure_points = {
            '✅ Excellent title format': 6.25,
            '✅ Good title format': 5.0,
            '⚠️ Title mentions NVIDIA': 3.0,
            '❌ Title should mention': 0,
            '❌ No title found': 0
        }
        
        title_status = structure_assessment.get('title', '')
        for status_text, points in structure_points.items():
            if status_text in title_status:
                scores['structure'] += points
                break
        
        # Introduction scoring
        intro_status = structure_assessment.get('introduction', '')
        if '✅ Comprehensive introduction' in intro_status:
            scores['structure'] += 6.25
        elif '✅ Good introduction' in intro_status:
            scores['structure'] += 5.0
        elif '⚠️' in intro_status:
            scores['structure'] += 2.5
        
        # Navigation scoring
        nav_status = structure_assessment.get('navigation', '')
        if '✅ Excellent navigation' in nav_status:
            scores['structure'] += 6.25
        elif '✅ Good use of headers' in nav_status:
            scores['structure'] += 5.0
        elif '⚠️' in nav_status:
            scores['structure'] += 2.5
        
        # Conclusion scoring
        conclusion_status = structure_assessment.get('conclusion', '')
        if '✅ Comprehensive conclusion' in conclusion_status:
            scores['structure'] += 6.25
        elif '✅ Good conclusion' in conclusion_status:
            scores['structure'] += 5.0
        elif '⚠️' in conclusion_status:
            scores['structure'] += 2.5
        
        # Content Quality scoring (25 points max)
        # Deduct points for issues, but use nuanced scoring
        nvidia_specific_issues = sum(1 for issue in content_issues if 'nvidia' in issue.lower())
        general_issues = len(content_issues) - nvidia_specific_issues
        
        content_base = 25.0
        content_base -= nvidia_specific_issues * 4.0  # More severe penalty for NVIDIA-specific issues
        content_base -= general_issues * 2.5  # Standard penalty for general issues
        scores['content'] = max(content_base, 0.0)
        
        # Technical Standards scoring (25 points max)
        # Categorize technical recommendations by severity
        critical_tech_issues = sum(1 for rec in technical_recommendations 
                                 if any(word in rec.lower() for word in ['critical', 'security', 'hardcoded', 'nvidia requirement']))
        standard_tech_issues = len(technical_recommendations) - critical_tech_issues
        
        tech_base = 25.0
        tech_base -= critical_tech_issues * 5.0  # Higher penalty for critical issues
        tech_base -= standard_tech_issues * 3.0  # Standard penalty for other issues
        scores['technical'] = max(tech_base, 0.0)
        
        # NVIDIA Compliance scoring (25 points max)
        # This is harder to assess statically, so use heuristics
        nvidia_base = 20.0  # Start with moderate base score
        
        # Check for NVIDIA-specific elements
        all_text = ' '.join([str(v) for v in structure_assessment.values()])
        
        if 'nvidia' in all_text.lower():
            nvidia_base += 3.0  # Bonus for mentioning NVIDIA
        
        if len([issue for issue in content_issues if 'nvidia' in issue.lower()]) == 0:
            nvidia_base += 2.0  # Bonus for no NVIDIA-specific issues
        
        scores['nvidia'] = min(nvidia_base, 25.0)
        
        # Calculate total score
        total_score = sum(scores.values())
        
        # Apply best practices enhancement if available
        if self.best_practices.guidelines:
            # Small bonus for having comprehensive guidelines loaded
            total_score = min(total_score + 2.0, 100.0)
        
        return round(total_score, 1)
    
    def get_best_practices_summary(self) -> Dict[str, str]:
        """Get a summary of the loaded NVIDIA Best Practices for display."""
        if not self.best_practices.guidelines:
            return {"status": "No guidelines loaded"}
        
        return {
            "status": "NVIDIA Best Practices loaded",
            "structure_requirements": "Clear titles, comprehensive introductions, proper navigation, strong conclusions",
            "content_standards": "Balanced documentation ratio, explained code, educational value, professional writing",
            "technical_standards": "Requirements.txt, environment variables, reproducibility, minimal file complexity",
            "scoring_framework": "100-point system: 25 points each for structure, content, technical, and NVIDIA compliance",
            "guidelines_source": "analyzer/nvidia_best_practices.md"
        }
    
    def analyze_notebook(self, url_or_path: str) -> GPURequirement:
        """
        Main entry point for notebook analysis.
        Coordinates all analysis components and returns comprehensive results.
        """
        if not self.quiet_mode:
            print(f"📁 Analyzing notebook: {url_or_path}")
        
        # Extract and parse notebook content
        try:
            code_cells, markdown_cells = self._extract_notebook_content(url_or_path)
        except Exception as e:
            if not self.quiet_mode:
                print(f"❌ Failed to load notebook: {e}")
            raise
        
        if not self.quiet_mode:
            print(f"✅ Successfully loaded notebook ({len(code_cells)} code cells, {len(markdown_cells)} markdown cells)")
        
        # Perform static analysis
        static_analysis = self._perform_static_analysis(code_cells, markdown_cells)
        
        # Enhance with LLM if available
        llm_context = None
        llm_reasoning = []
        self_review = None  # Phase 2.5: Initialize self-review
        if self.llm_analyzer:
            if not self.quiet_mode:
                print("🤖 Enhancing analysis with LLM...")
            llm_context = self.llm_analyzer.analyze_notebook_context(code_cells, markdown_cells)
            if llm_context:
                enhanced_analysis, llm_reasoning = self.llm_analyzer.enhance_gpu_recommendation(static_analysis, llm_context)
                static_analysis.update(enhanced_analysis)
                
                # CRITICAL FIX: Regenerate comprehensive recommendations after LLM enhancement
                # The LLM may have significantly changed VRAM requirements, so we need to update
                # the consumer/enterprise recommendations based on the new estimates
                self._generate_comprehensive_recommendations(static_analysis)
                
                # CONSISTENCY FIX: Ensure minimum VRAM shows total available VRAM when quantity > 1
                # This must happen AFTER LLM enhancement to get the correct final VRAM value
                if static_analysis.get('min_quantity', 1) > 1:
                    # Convert per-GPU VRAM to total VRAM for multi-GPU minimum setups
                    min_gpu_type = static_analysis.get('min_gpu_type', '')
                    if min_gpu_type in self.gpu_specs:
                        per_gpu_vram = self.gpu_specs[min_gpu_type]['vram']
                        min_quantity = static_analysis.get('min_quantity', 1)
                        static_analysis['min_vram_gb'] = per_gpu_vram * min_quantity
                
                # RUNTIME CONSISTENCY FIX: Update minimum runtime to match consumer/enterprise when same hardware
                # If consumer recommendation exists and uses same GPU, use consumer runtime for consistency
                consumer_gpu_type = static_analysis.get('consumer_gpu_type')
                consumer_quantity = static_analysis.get('consumer_quantity')
                min_gpu_type = static_analysis.get('min_gpu_type', '')
                min_quantity = static_analysis.get('min_quantity', 1)
                
                if (consumer_gpu_type == min_gpu_type and consumer_quantity == min_quantity and
                    static_analysis.get('consumer_runtime_estimate')):
                    # Same hardware - use consumer runtime for consistency
                    static_analysis['min_runtime_estimate'] = static_analysis['consumer_runtime_estimate']
                
                # Let LLM confidence take precedence over static analysis for CPU-only workloads
                enhanced_confidence = self._calculate_dynamic_confidence(static_analysis, llm_context)
                static_analysis['confidence'] = enhanced_confidence
                
                if not self.quiet_mode:
                    print(f"✅ LLM analysis complete (confidence: {enhanced_confidence*100:.0f}%)")
                
                # PHASE 2.5: Self-review analysis for consistency and accuracy
                if not self.quiet_mode:
                    print("🎓 Performing self-review for accuracy and consistency...")
                
                self_review = self.llm_analyzer.self_review_analysis(
                    code_cells, static_analysis, static_analysis.get('reasoning', []), llm_reasoning
                )
                
                if self_review:
                    # Apply self-review corrections
                    corrected_analysis, final_reasoning = self.llm_analyzer.apply_self_review_corrections(
                        static_analysis, static_analysis.get('reasoning', []), llm_reasoning, self_review
                    )
                    
                    # Update analysis with corrected values
                    static_analysis.update(corrected_analysis)
                    
                    # Keep original static reasoning, but replace LLM reasoning with unified reasoning from self-review
                    llm_reasoning = final_reasoning
                    
                    if not self.quiet_mode:
                        review_status = "passed" if self_review.get('review_passed', True) else "corrected issues"
                        print(f"✅ Self-review {review_status} - enhanced accuracy and consistency")
                else:
                    # If no self-review, recalculate confidence one more time with final LLM context
                    final_confidence = self._calculate_dynamic_confidence(static_analysis, llm_context)
                    static_analysis['confidence'] = final_confidence
                    self_review = None
        
        # Evaluate NVIDIA Best Practices compliance
        if not self.quiet_mode:
            print("📋 Evaluating NVIDIA compliance...")
        
        structure_assessment = self.evaluate_notebook_structure(code_cells, markdown_cells)
        content_issues = self.assess_content_quality(code_cells, markdown_cells)
        technical_recommendations = self.check_technical_standards(code_cells)
        
        # Get LLM compliance evaluation if available
        llm_compliance = None
        if self.llm_analyzer:
            llm_compliance = self.llm_analyzer.evaluate_notebook_compliance(code_cells, markdown_cells)
        
        # Calculate comprehensive compliance score
        compliance_score = self.calculate_nvidia_compliance_score(
            structure_assessment, content_issues, technical_recommendations, llm_compliance
        )
        
        if not self.quiet_mode:
            print(f"✅ Compliance evaluation complete (score: {compliance_score:.0f}/100)")
        
        # Build final result
        return GPURequirement(
            min_gpu_type=static_analysis['min_gpu_type'],
            min_quantity=static_analysis['min_quantity'],
            min_vram_gb=static_analysis['min_vram_gb'],
            min_runtime_estimate=static_analysis['min_runtime_estimate'],
            recommended_gpu_type=static_analysis.get('consumer_gpu_type'),
            recommended_quantity=static_analysis.get('consumer_quantity'),
            recommended_vram_gb=static_analysis.get('consumer_vram_gb'),
            recommended_runtime_estimate=static_analysis.get('consumer_runtime_estimate'),
            recommended_viable=static_analysis.get('consumer_viable', True),
            recommended_limitation=static_analysis.get('consumer_limitation'),
            optimal_gpu_type=static_analysis.get('enterprise_gpu_type', ""),
            optimal_quantity=static_analysis.get('enterprise_quantity', 1),
            optimal_vram_gb=static_analysis.get('enterprise_vram_gb', 0),
            optimal_runtime_estimate=static_analysis.get('enterprise_runtime_estimate', ""),
            sxm_required=static_analysis['sxm_required'],
            sxm_reasoning=static_analysis['sxm_reasoning'],
            arm_compatibility=static_analysis['arm_compatibility'],
            arm_reasoning=static_analysis['arm_reasoning'],
            confidence=static_analysis['confidence'],
            reasoning=static_analysis['reasoning'],
            llm_enhanced=llm_context is not None,
            llm_reasoning=llm_reasoning,
            self_reviewed=llm_context is not None and self_review is not None,
            llm_model_used=self.llm_analyzer.model if self.llm_analyzer and llm_context is not None else None,
            nvidia_compliance_score=compliance_score,
            structure_assessment=structure_assessment,
            content_quality_issues=content_issues,
            technical_recommendations=technical_recommendations,
            confidence_factors=static_analysis.get('confidence_factors', []),
            workload_detected=static_analysis['workload_detected']
        )
        
    def _extract_notebook_content(self, url_or_path: str) -> Tuple[List[str], List[str]]:
        """Extract code and markdown cells from notebook with optimized I/O."""
        import urllib.parse
        import json
        import requests
        
        code_cells = []
        markdown_cells = []
        
        # Determine if it's a URL or local path
        parsed = urllib.parse.urlparse(url_or_path)
        is_url = parsed.scheme in ('http', 'https')
        
        if is_url:
            # Handle various URL formats (GitHub, GitLab, etc.)
            if 'github.com' in url_or_path and '/blob/' in url_or_path:
                # Convert GitHub blob URL to raw URL
                url_or_path = url_or_path.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
            elif 'gitlab.com' in url_or_path and '/-/blob/' in url_or_path:
                # Convert GitLab blob URL to raw URL
                url_or_path = url_or_path.replace('/-/blob/', '/-/raw/')
            
            # Download notebook content directly to memory (no temp file)
            response = requests.get(url_or_path, timeout=30)
            response.raise_for_status()
            content = response.text
            
            # Process content directly in memory
            if url_or_path.endswith('.py'):
                # marimo notebook - process directly
                code_cells = [content]
                # Extract docstrings as markdown using optimized AST parsing
                import ast
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if (isinstance(node, ast.Expr) and 
                            isinstance(node.value, ast.Constant) and 
                            isinstance(node.value.value, str) and
                            len(node.value.value) > 50):  # Likely a docstring
                            markdown_cells.append(node.value.value)
                except:
                    pass
            else:
                # Jupyter notebook - optimized JSON processing
                notebook_data = json.loads(content)
                
                # Process cells in parallel for large notebooks
                cells = notebook_data.get('cells', [])
                if len(cells) > 20:  # Use parallel processing for large notebooks
                    def process_cell(cell):
                        cell_type = cell.get('cell_type')
                        source = cell.get('source', [])
                        content = ''.join(source) if isinstance(source, list) else source
                        return (cell_type, content)
                    
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        processed_cells = list(executor.map(process_cell, cells))
                    
                    for cell_type, content in processed_cells:
                        if cell_type == 'code':
                            code_cells.append(content)
                        elif cell_type == 'markdown':
                            markdown_cells.append(content)
                else:
                    # Sequential processing for smaller notebooks
                    for cell in cells:
                        if cell.get('cell_type') == 'code':
                            source = cell.get('source', [])
                            if isinstance(source, list):
                                code_cells.append(''.join(source))
                            else:
                                code_cells.append(source)
                        elif cell.get('cell_type') == 'markdown':
                            source = cell.get('source', [])
                            if isinstance(source, list):
                                markdown_cells.append(''.join(source))
                            else:
                                markdown_cells.append(source)
        else:
            # Local file
            if not os.path.exists(url_or_path):
                raise FileNotFoundError(f"File not found: {url_or_path}")
            
            if url_or_path.endswith('.py'):
                # marimo notebook
                with open(url_or_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    code_cells = [content]
                    # Extract docstrings as markdown
                    import ast
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                if len(node.value.value) > 50:  # Likely a docstring
                                    markdown_cells.append(node.value.value)
                    except:
                        pass
            else:
                # Jupyter notebook
                with open(url_or_path, 'r', encoding='utf-8') as f:
                    notebook_data = json.load(f)
                for cell in notebook_data.get('cells', []):
                    if cell.get('cell_type') == 'code':
                        source = cell.get('source', [])
                        if isinstance(source, list):
                            code_cells.append(''.join(source))
                        else:
                            code_cells.append(source)
                    elif cell.get('cell_type') == 'markdown':
                        source = cell.get('source', [])
                        if isinstance(source, list):
                            markdown_cells.append(''.join(source))
                        else:
                            markdown_cells.append(source)
        
        return code_cells, markdown_cells
        
    def _perform_static_analysis(self, code_cells: List[str], markdown_cells: List[str]) -> Dict:
        """Perform static analysis of notebook content using CPU-first approach."""
        # Start with CPU-only defaults 
        analysis = {
            'min_gpu_type': 'CPU-only',
            'min_quantity': 0,
            'min_vram_gb': 0,
            'min_runtime_estimate': 'CPU execution',
            'consumer_gpu_type': None,
            'consumer_quantity': None,
            'consumer_vram_gb': None,
            'consumer_runtime_estimate': None,
            'consumer_viable': False,
            'consumer_limitation': "CPU-optimized workload",
            'enterprise_gpu_type': "",
            'enterprise_quantity': 0,
            'enterprise_vram_gb': 0,
            'enterprise_runtime_estimate': "",
            'sxm_required': False,
            'sxm_reasoning': [],
            'arm_compatibility': 'Likely Compatible',
            'arm_reasoning': [],
            'confidence': 0.7,  # Will be recalculated dynamically
            'reasoning': [],
            'workload_detected': False,  # Default to no GPU workload
            'workload_type': 'data-analysis'  # Default workload type
        }
        
        all_code = '\n'.join(code_cells)
        all_content = all_code + '\n'.join(markdown_cells)
        
        # Use the new GPU benefit detection system
        gpu_benefit_analysis = self.detect_gpu_benefit_level(all_content)
        
        # Store GPU benefit analysis for later use
        analysis['_gpu_benefit_analysis'] = gpu_benefit_analysis
        
        # Update analysis based on GPU benefit level
        benefit_level = gpu_benefit_analysis['benefit_level']
        
        # CRITICAL FIX: Use confidence from GPU benefit analysis as the baseline
        analysis['confidence'] = gpu_benefit_analysis['confidence']
        
        if benefit_level == 'none':
            # CPU-only workload - keep defaults
            analysis['workload_type'] = gpu_benefit_analysis['workload_type']
            analysis['reasoning'].extend(gpu_benefit_analysis['reasoning'])
            return analysis
        
        # GPU workload detected - update analysis
        analysis['workload_detected'] = True
        analysis['workload_type'] = gpu_benefit_analysis['workload_type']
        analysis['reasoning'].extend(gpu_benefit_analysis['reasoning'])
        
        # Set initial GPU recommendations based on benefit level
        vram_needed = gpu_benefit_analysis['vram_estimate']
        
        if benefit_level == 'required':
            # High-end workload
            analysis['min_gpu_type'] = 'L40S' if vram_needed > 24 else 'RTX 4090'
            analysis['min_vram_gb'] = max(vram_needed, 24)
            analysis['consumer_viable'] = vram_needed <= 24
            analysis['consumer_limitation'] = "Workload requires enterprise-grade hardware" if vram_needed > 24 else None
        elif benefit_level == 'recommended':
            # Training/significant ML workload
            analysis['min_gpu_type'] = 'RTX 4070' if vram_needed <= 12 else 'RTX 4080'
            analysis['min_vram_gb'] = max(vram_needed, 8)
            analysis['consumer_viable'] = True
            analysis['consumer_limitation'] = None
        else:  # beneficial
            # Light ML workload
            analysis['min_gpu_type'] = 'RTX 4060'
            analysis['min_vram_gb'] = max(vram_needed, 8)
            analysis['consumer_viable'] = True
            analysis['consumer_limitation'] = None
        
        # Update quantities and runtime estimates
        analysis['min_quantity'] = 1
        analysis['min_runtime_estimate'] = self._convert_runtime_to_new_format('1.0-2.0')
        analysis['consumer_gpu_type'] = analysis['min_gpu_type']
        analysis['consumer_quantity'] = 1
        analysis['consumer_vram_gb'] = analysis['min_vram_gb']
        analysis['consumer_runtime_estimate'] = analysis['min_runtime_estimate']
        
        # Detect frameworks for additional context
        detected_frameworks = []
        for framework, patterns in self.framework_patterns.items():
            # Use parallel search for framework patterns
            framework_matches = parallel_pattern_search(all_code, patterns)
            if any(framework_matches):
                detected_frameworks.append(framework)
        
        analysis['detected_frameworks'] = detected_frameworks
        
        # Estimate VRAM requirements based on detected models
        for model, specs in self.model_specs.items():
            if model in all_code.lower():
                vram_needed = max(vram_needed, specs['base_vram'])
                analysis['reasoning'].append(f"Detected {model} model requiring {specs['base_vram']}GB+ VRAM")
                
                # CRITICAL FIX: Only update GPU recommendation if model actually requires more VRAM
                # Don't automatically escalate to enterprise hardware for small models
                if specs['base_vram'] > analysis['min_vram_gb']:
                    analysis['min_vram_gb'] = specs['base_vram']
                    
                    # Only escalate to enterprise if model is genuinely large (>24GB)
                    if specs['base_vram'] > 24:
                        # Large model requires enterprise GPU
                        analysis['consumer_viable'] = False
                        analysis['consumer_limitation'] = "Large model requires enterprise GPU"
                        # CRITICAL FIX: If consumer is not viable, minimum should be enterprise too
                        # This ensures logical consistency between minimum and recommended tiers
                        if specs['base_vram'] <= 48:
                            analysis['min_gpu_type'] = 'L40S'
                            analysis['min_vram_gb'] = 48
                        else:
                            analysis['min_gpu_type'] = 'A100 PCIe 80G'
                            analysis['min_vram_gb'] = 80
                    elif specs['base_vram'] > 16:
                        # Medium model - update to RTX 4090 but keep consumer viable
                        analysis['min_gpu_type'] = 'RTX 4090'
                        analysis['min_vram_gb'] = 24  # RTX 4090 VRAM
                    elif specs['base_vram'] > 12:
                        # Small-medium model - update to RTX 4080 but keep consumer viable  
                        analysis['min_gpu_type'] = 'RTX 4080'
                        analysis['min_vram_gb'] = 16  # RTX 4080 VRAM
                    # For models ≤12GB, keep existing GPU recommendation from benefit level
        
        # Check for multi-GPU patterns using parallel search
        multi_gpu_matches = parallel_pattern_search(all_code, self.multi_gpu_patterns)
        multi_gpu_detected = any(multi_gpu_matches)
        if multi_gpu_detected:
            # Find which pattern matched for reasoning
            for i, matched in enumerate(multi_gpu_matches):
                if matched:
                    analysis['reasoning'].append(f"Multi-GPU capability detected: {self.multi_gpu_patterns[i]}")
                    break
        
        # Check for SXM-specific patterns using parallel search
        sxm_matches = parallel_pattern_search(all_code, self.sxm_patterns)
        if any(sxm_matches):
            analysis['sxm_required'] = True
            # Find which pattern matched for reasoning
            for i, matched in enumerate(sxm_matches):
                if matched:
                    analysis['sxm_reasoning'].append(f"Large-scale training pattern detected: {self.sxm_patterns[i]}")
                    break
        
        # GPU recommendations are already set based on GPU benefit level above

        # Set quantities based on multi-GPU detection and workload requirements
        if multi_gpu_detected or analysis.get('sxm_required', False):
            # Multi-GPU workload - minimum should be at least 2
            if analysis.get('sxm_required', False):
                # Large-scale training typically needs 4+ GPUs minimum
                analysis['min_quantity'] = 4
                analysis['optimal_quantity'] = 8
                analysis['reasoning'].append("Large-scale training requires minimum 4 GPUs, optimal 8")
            else:
                # Standard multi-GPU setup
                analysis['min_quantity'] = 2
                analysis['optimal_quantity'] = 2
                analysis['reasoning'].append("Multi-GPU setup recommended - PCIe GPUs with NVLink sufficient")
        else:
            # Single GPU workload
            analysis['min_quantity'] = 1
            analysis['optimal_quantity'] = 1
        
        # Validate and normalize quantities to allowed values (1, 2, 4, 8, multiples of 8)
        analysis['min_quantity'] = normalize_gpu_quantity(analysis['min_quantity'])
        analysis['optimal_quantity'] = normalize_gpu_quantity(analysis['optimal_quantity'])
        
        # CRITICAL FIX: Validate SXM requirements against selected GPUs
        analysis = self._validate_sxm_requirements(analysis)

        # Generate comprehensive recommendations: minimum, consumer, enterprise
        self._generate_comprehensive_recommendations(analysis)

        # Enhanced ARM compatibility assessment
        arm_compatibility_score = 0
        arm_issues = []
        arm_positives = []
        
        # Check for ARM-compatible frameworks
        compatible_frameworks_found = []
        for framework in detected_frameworks:
            if framework in self.arm_compatible_frameworks:
                compatible_frameworks_found.append(framework)
                arm_compatibility_score += 10
        
        if compatible_frameworks_found:
            arm_positives.append(f"Uses {len(compatible_frameworks_found)} ARM-compatible frameworks: {', '.join(compatible_frameworks_found[:3])}")
        
        # Check for ARM optimization indicators
        arm_optimizations_found = []
        for pattern in self.arm_optimization_patterns:
            if re.search(pattern, all_code, re.IGNORECASE):
                matches = re.findall(pattern, all_code, re.IGNORECASE)
                arm_optimizations_found.extend(matches[:2])  # Limit to avoid spam
        
        if arm_optimizations_found:
            arm_compatibility_score += 15
            arm_positives.append(f"Contains ARM-specific optimizations: {', '.join(set(arm_optimizations_found))}")
        
        # Check for version-specific compatibility
        version_warnings = []
        import_patterns = re.findall(r'(?:import|from)\s+([a-zA-Z0-9_-]+)', all_code)
        pip_patterns = re.findall(r'pip install\s+([a-zA-Z0-9_-]+)(?:==([0-9.]+))?', all_code, re.IGNORECASE)
        
        for package, version in pip_patterns:
            if package in self.arm_compatibility_versions and version:
                min_version = self.arm_compatibility_versions[package]['min_compatible']
                if compare_versions(version, min_version) < 0:
                    arm_issues.append(f"{package} v{version} may have ARM issues (min recommended: v{min_version})")
                    arm_compatibility_score -= 20
                else:
                    optimal_version = self.arm_compatibility_versions[package]['optimal']
                    if compare_versions(version, optimal_version) >= 0:
                        arm_compatibility_score += 5
                        arm_positives.append(f"{package} v{version} has good ARM support")
        
        # Check for ARM incompatible patterns
        for pattern in self.arm_incompatible_patterns:
            matches = re.findall(pattern, all_code, re.IGNORECASE)
            if matches:
                arm_compatibility_score -= 25
                # Group similar issues to avoid spam with more specific pattern matching
                if pattern in [r'intel.*mkl', r'mkl.*intel', r'oneapi', r'intel.*optimization', r'mkldnn', r'onednn.*intel', r'intel.*threading']:
                    arm_issues.append("Uses Intel-specific optimizations that may not work on ARM")
                elif pattern in [r'x86.*specific', r'sse[0-9]*', r'avx[0-9]*', r'fma.*instruction', r'intel.*compiler', r'icc.*compiler']:
                    arm_issues.append("Contains x86-specific instructions/optimizations")
                elif pattern == r'cudnn.*version.*<.*8':
                    arm_issues.append("Uses cuDNN version < 8.0 (limited ARM support)")
                elif pattern == r'tensorrt.*<.*8':
                    arm_issues.append("Uses TensorRT version < 8.0 (limited ARM support)")
                elif pattern == r'tensorflow.*<.*2\.8':
                    arm_issues.append("Uses TensorFlow < 2.8 (limited ARM support)")
                elif pattern == r'torch.*<.*1\.12':
                    arm_issues.append("Uses PyTorch < 1.12 (limited ARM support)")
                elif pattern == r'fbgemm':
                    arm_issues.append("Uses FBGEMM quantization (x86-only)")
                elif pattern in [r'apex\.', r'flash.attention.*<.*2', r'xformers.*<.*0\.2', r'deepspeed.*<.*0\.7', r'bitsandbytes.*<.*0\.3']:
                    arm_issues.append("Uses legacy library versions with limited ARM support")
                elif 'triton.*kernel' in pattern:
                    arm_issues.append("Uses Triton kernels that may have limited ARM support")
                else:
                    arm_issues.append("Potential ARM incompatibility detected")
                break  # Avoid duplicate similar warnings
        
        # Workload-specific ARM compatibility considerations
        if analysis['workload_type'] == 'training':
            if 'distributed' in all_code.lower():
                arm_compatibility_score -= 10
                arm_issues.append("Distributed training on ARM may have performance limitations")
            else:
                arm_compatibility_score += 5
                arm_positives.append("Single-node training generally works well on ARM")
        else:
            arm_compatibility_score += 10
            arm_positives.append("Inference workloads typically have good ARM compatibility")
        
        # Model-specific ARM considerations
        arm_heavy_models = ['stable.diffusion', 'llama', 'gpt']
        for model in arm_heavy_models:
            if model in all_code.lower():
                if analysis['min_vram_gb'] > 40:
                    arm_issues.append(f"Large {model} models may have reduced performance on ARM")
                    arm_compatibility_score -= 5
                else:
                    arm_positives.append(f"Smaller {model} models generally work well on ARM")
                    arm_compatibility_score += 3
        
        # Final ARM compatibility determination
        if arm_compatibility_score >= 30:
            analysis['arm_compatibility'] = 'Highly Compatible'
        elif arm_compatibility_score >= 10:
            analysis['arm_compatibility'] = 'Compatible'
        elif arm_compatibility_score >= -10:
            analysis['arm_compatibility'] = 'Likely Compatible'
        elif arm_compatibility_score >= -25:
            analysis['arm_compatibility'] = 'Possibly Incompatible'
        else:
            analysis['arm_compatibility'] = 'Likely Incompatible'
        
        # Build comprehensive ARM reasoning
        analysis['arm_reasoning'] = []
        if arm_positives:
            analysis['arm_reasoning'].extend(arm_positives[:3])  # Top 3 positive indicators
        if arm_issues:
            analysis['arm_reasoning'].extend(arm_issues[:3])     # Top 3 issues
        
        # Add summary score for transparency
        analysis['arm_reasoning'].append(f"ARM compatibility score: {arm_compatibility_score} (higher is better)")
        
        # Calculate dynamic confidence based on analysis quality
        dynamic_confidence = self._calculate_dynamic_confidence(analysis)
        analysis['confidence'] = dynamic_confidence
        
        return analysis
    
    def _validate_sxm_requirements(self, analysis: Dict) -> Dict:
        """
        Validate SXM requirements against selected GPUs and fix inconsistencies.
        If SXM is required but selected GPUs don't support it, either clear the requirement
        or upgrade to SXM-capable GPUs.
        """
        if not analysis.get('sxm_required', False):
            return analysis
        
        # Check if selected GPUs support SXM
        min_gpu = analysis.get('min_gpu_type', '')
        optimal_gpu = analysis.get('optimal_gpu_type', '')
        
        min_supports_sxm = min_gpu in self.gpu_specs and self.gpu_specs[min_gpu].get('form_factor') == 'SXM'
        optimal_supports_sxm = optimal_gpu in self.gpu_specs and self.gpu_specs[optimal_gpu].get('form_factor') == 'SXM'
        
        # If neither GPU supports SXM, we have two options:
        # 1. Clear the SXM requirement (likely false positive)
        # 2. Upgrade to SXM GPUs (if workload truly needs it)
        
        if not min_supports_sxm and not optimal_supports_sxm:
            # Use intelligent assessment to determine if SXM is truly needed
            truly_needs_sxm = self._assess_sxm_necessity(analysis)
            
            if not truly_needs_sxm:
                # Clear false positive SXM requirement
                analysis['sxm_required'] = False
                analysis['sxm_reasoning'] = ["SXM requirement cleared - workload can run efficiently on PCIe GPUs"]
                analysis['reasoning'].append("Cleared SXM requirement - PCIe GPUs with NVLink sufficient for this workload")
            else:
                # Real multi-GPU requirement - upgrade to SXM GPUs
                vram_requirement = analysis.get('min_vram_gb', 24)
                
                if vram_requirement <= 80:
                    # Use A100 SXM for moderate requirements
                    analysis['min_gpu_type'] = 'A100 SXM 80G'
                    analysis['optimal_gpu_type'] = 'H100 SXM'
                    analysis['min_vram_gb'] = 80
                    analysis['optimal_vram_gb'] = 80
                elif vram_requirement <= 141:
                    # Use H200 SXM for high requirements
                    analysis['min_gpu_type'] = 'H100 SXM'
                    analysis['optimal_gpu_type'] = 'H200 SXM'
                    analysis['min_vram_gb'] = 80
                    analysis['optimal_vram_gb'] = 141
                else:
                    # Use B200 SXM for extreme requirements
                    analysis['min_gpu_type'] = 'H200 SXM'
                    analysis['optimal_gpu_type'] = 'B200 SXM'
                    analysis['min_vram_gb'] = 141
                    analysis['optimal_vram_gb'] = 192
                
                analysis['reasoning'].append(f"Upgraded to SXM GPUs due to large-scale training requirement: {analysis['min_gpu_type']} -> {analysis['optimal_gpu_type']}")
        
        return analysis
    
    def _assess_sxm_necessity(self, analysis: Dict) -> bool:
        """
        Intelligently assess if SXM form factor is truly necessary based on workload characteristics.
        Returns True if SXM is needed, False if PCIe GPUs are sufficient.
        """
        # Factor 1: Scale assessment
        scale_score = 0
        optimal_quantity = analysis.get('optimal_quantity', 1)
        vram_requirement = analysis.get('min_vram_gb', 8)
        
        # Large GPU count suggests SXM need
        if optimal_quantity >= 8:
            scale_score += 40
        elif optimal_quantity >= 4:
            scale_score += 20
        elif optimal_quantity >= 2:
            scale_score += 5  # 2 GPUs can easily be PCIe + NVLink
        
        # Factor 2: Model size and complexity
        model_score = 0
        workload_type = analysis.get('workload_type', 'inference')
        workload_complexity = analysis.get('workload_complexity', 'moderate')
        
        # Very large models may benefit from SXM power/bandwidth
        if vram_requirement >= 100:
            model_score += 30
        elif vram_requirement >= 80:
            model_score += 15
        elif vram_requirement >= 40:
            model_score += 5
        
        # Training complexity
        if workload_type == 'training' and workload_complexity in ['complex', 'extreme']:
            model_score += 20
        elif workload_type == 'training':
            model_score += 10
        
        # Factor 3: Pattern analysis
        pattern_score = 0
        sxm_reasoning = analysis.get('sxm_reasoning', [])
        reasoning_text = ' '.join(sxm_reasoning).lower()
        
        # High-confidence SXM patterns
        high_confidence_patterns = [
            'num_nodes', 'world_size', 'multi_node', 'dgx_system', 
            'superpod', 'slurm', 'tensor_model_parallel_size'
        ]
        for pattern in high_confidence_patterns:
            if pattern in reasoning_text:
                pattern_score += 25
                break
        
        # Medium-confidence patterns
        medium_confidence_patterns = [
            'zero_stage.*3', 'fully_sharded', 'pipeline_parallel'
        ]
        for pattern in medium_confidence_patterns:
            if pattern in reasoning_text:
                pattern_score += 15
                break
        
        # Low-confidence patterns (likely false positives)
        low_confidence_patterns = [
            'device_count', 'dataparallel', 'multi.*gpu.*setup'
        ]
        for pattern in low_confidence_patterns:
            if pattern in reasoning_text:
                pattern_score -= 10  # Negative score for likely false positives
        
        # Calculate total score
        total_score = scale_score + model_score + pattern_score
        
        # Decision threshold: SXM needed if score >= 50
        return total_score >= 50
    
    def _assess_consumer_viability(self, analysis: Dict, per_gpu_vram: Optional[int] = None, quantity: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """
        Unified consumer viability assessment supporting both parameter styles.
        
        Args:
            analysis: Analysis dictionary containing workload information
            per_gpu_vram: Optional explicit per-GPU VRAM requirement (if None, calculated from analysis)
            quantity: Optional explicit GPU quantity (if None, extracted from analysis)
            
        Returns:
            (is_viable, limitation_reason)
        """
        # Determine VRAM and quantity parameters
        if per_gpu_vram is None or quantity is None:
            # Legacy mode: extract from analysis dictionary
            total_vram = analysis.get('min_vram_gb', 8)
            extracted_quantity = analysis.get('optimal_quantity', 1)
            calculated_per_gpu_vram = total_vram // extracted_quantity if extracted_quantity > 0 else total_vram
            
            # Use provided parameters or calculated values
            final_per_gpu_vram = per_gpu_vram if per_gpu_vram is not None else calculated_per_gpu_vram
            final_quantity = quantity if quantity is not None else extracted_quantity
        else:
            # Direct mode: use provided parameters
            final_per_gpu_vram = per_gpu_vram
            final_quantity = quantity
        
        max_consumer_vram = 24  # RTX 4090
        
        # Check VRAM requirements
        if final_per_gpu_vram > max_consumer_vram:
            return False, f"VRAM requirement ({final_per_gpu_vram}GB per GPU) exceeds consumer GPU capacity (max {max_consumer_vram}GB)"
        
        # Check scale requirements
        if final_quantity > 2:
            return False, f"Multi-GPU setup ({final_quantity} GPUs) beyond consumer capabilities (max 2)"
        
        # Check SXM requirements
        if analysis.get('sxm_required', False):
            return False, "Workload requires enterprise-grade interconnect (SXM)"
        
        # Check workload complexity
        workload_complexity = analysis.get('workload_complexity', 'moderate')
        if workload_complexity == 'extreme':
            return False, "Workload complexity requires enterprise-grade features"
        
        # Check for enterprise-specific patterns
        reasoning_text = ' '.join(analysis.get('reasoning', [])).lower()
        enterprise_indicators = ['large-scale', 'multi-node', 'enterprise', 'data center', 'production scale']
        for indicator in enterprise_indicators:
            if indicator in reasoning_text:
                return False, f"Workload type requires enterprise infrastructure ({indicator})"
        
        return True, None
    
    def _generate_consumer_recommendation(self, vram_needed: int, quantity: int, workload_type: str, 
                                        llm_runtime_data: Optional[Dict] = None) -> Dict:
        """Generate consumer GPU recommendation based on requirements."""
        
        # Select best consumer GPU based on VRAM needs
        if vram_needed <= 8:
            gpu_type = 'RTX 4060'
            vram = 8
        elif vram_needed <= 12:
            gpu_type = 'RTX 4070'
            vram = 12
        elif vram_needed <= 16:
            gpu_type = 'RTX 4080'
            vram = 16
        else:  # Up to 24GB
            gpu_type = 'RTX 4090'
            vram = 24
        
        final_quantity = min(quantity, 2)  # Cap at 2 for consumer
        total_vram = vram * final_quantity  # Calculate total available VRAM
        
        # Calculate runtime using LLM data if available
        if llm_runtime_data:
            baseline_runtime = llm_runtime_data.get('baseline_runtime_hours', '2-3')
            baseline_gpu = llm_runtime_data.get('baseline_reference_gpu', 'RTX 4090')
            optimization_factor = llm_runtime_data.get('optimization_speedup_factor', 1.0)
            runtime = self._calculate_runtime_for_gpu(baseline_runtime, baseline_gpu, gpu_type, 
                                                    final_quantity, optimization_factor)
        else:
            # Fallback runtime estimates
            runtime = self._convert_runtime_to_new_format('1.0-2.0')
        
        return {
            'type': gpu_type,
            'quantity': final_quantity,
            'vram': total_vram,
            'runtime': runtime
        }
    
    def _generate_enterprise_recommendation(self, vram_needed: int, quantity: int, workload_type: str, 
                                          sxm_required: bool, llm_runtime_data: Optional[Dict] = None) -> Dict:
        """Generate enterprise GPU recommendation based on requirements."""
        
        # Calculate per-GPU VRAM requirement
        per_gpu_vram = vram_needed // quantity if quantity > 0 else vram_needed
        
        # Select based on VRAM needs and SXM requirements
        if sxm_required:
            # Use SXM GPUs for large-scale workloads
            if per_gpu_vram <= 80:  # Each GPU needs ≤80GB
                gpu_type = 'A100 SXM 80G'
                vram = 80
            elif per_gpu_vram <= 141:  # Each GPU needs ≤141GB
                gpu_type = 'H200 SXM'
                vram = 141
            else:  # Each GPU needs >141GB
                gpu_type = 'B200 SXM'
                vram = 192
        else:
            # Use PCIe enterprise GPUs for more moderate scale
            if per_gpu_vram <= 24:
                gpu_type = 'L40S'
                vram = 48
            elif per_gpu_vram <= 40:
                gpu_type = 'A100 PCIe 40G'
                vram = 40
            elif per_gpu_vram <= 48:
                gpu_type = 'L40S'
                vram = 48
            elif per_gpu_vram <= 80:
                gpu_type = 'A100 PCIe 80G'
                vram = 80
            else:
                gpu_type = 'H100 PCIe'
                vram = 80
        
        total_vram = vram * quantity
        
        # Calculate runtime using LLM data if available
        if llm_runtime_data:
            baseline_runtime = llm_runtime_data.get('baseline_runtime_hours', '2-3')
            baseline_gpu = llm_runtime_data.get('baseline_reference_gpu', 'RTX 4090')
            optimization_factor = llm_runtime_data.get('optimization_speedup_factor', 1.0)
            runtime = self._calculate_runtime_for_gpu(baseline_runtime, baseline_gpu, gpu_type, 
                                                    quantity, optimization_factor)
        else:
            runtime = self._convert_runtime_to_new_format('1.0-2.0')  # Fallback runtime estimate
        
        return {
            'type': gpu_type,
            'quantity': quantity,
            'vram': total_vram,
            'runtime': runtime
        }

    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare two version strings. Delegates to module-level function."""
        return compare_versions(version1, version2)
    
    def sanitize_file_content(self, file_content: bytes, filename: str) -> tuple:
        """
        Sanitize uploaded file content for security using comprehensive sandbox validation.
        Returns (is_safe: bool, error_msg: str, sanitized_content: dict/str)
        """
        try:
            # Import the security sandbox
            from .security_sandbox import SecuritySandbox
            
            # Create sandbox with strict limits
            sandbox = SecuritySandbox(max_memory_mb=256, max_time_seconds=10)
            
            # Decode file content
            try:
                content_str = file_content.decode('utf-8')
            except UnicodeDecodeError:
                return False, "Invalid file encoding - please use UTF-8", None
            
            # Validate based on file type
            if filename.lower().endswith('.ipynb'):
                # Use sandbox validation for Jupyter notebooks
                is_safe, error_msg, sanitized_content = sandbox.validate_notebook_structure(content_str)
                return is_safe, error_msg, sanitized_content
                
            elif filename.lower().endswith('.py'):
                # Use sandbox validation for Python files
                is_safe, error_msg, sanitized_content = sandbox.validate_python_file(content_str)
                return is_safe, error_msg, sanitized_content
                
            else:
                return False, "Unsupported file type - only .ipynb and .py files are allowed", None
                
        except ImportError:
            # Fallback to basic validation if sandbox not available
            return self._basic_sanitize_file_content(file_content, filename)
        except Exception as e:
            return False, f"Security validation error: {str(e)}", None
    
    def _basic_sanitize_file_content(self, file_content: bytes, filename: str) -> tuple:
        """
        Basic fallback sanitization (original method) - should not be used in production.
        """
        try:
            content_str = file_content.decode('utf-8')
            
            if filename.lower().endswith('.ipynb'):
                import json
                try:
                    notebook_data = json.loads(content_str)
                    
                    if not isinstance(notebook_data, dict):
                        return False, "Invalid notebook format - not a JSON object", None
                    if 'cells' not in notebook_data:
                        return False, "Invalid notebook format - missing cells", None
                    if not isinstance(notebook_data['cells'], list):
                        return False, "Invalid notebook format - cells must be an array", None
                    
                    # Enhanced dangerous pattern detection
                    content_lower = content_str.lower()
                    dangerous_patterns = [
                        'subprocess.', 'os.system', 'eval(', 'exec(', '__import__', 
                        'open(', 'file(', 'input(', 'getattr(', 'setattr(',
                        'globals(', 'locals(', 'vars(', 'dir(',
                        'compile(', 'reload(', '__builtins__', 'rm -rf',
                        'sudo ', 'chmod ', 'wget ', 'curl ', 'nc -l'
                    ]
                    
                    for pattern in dangerous_patterns:
                        if pattern in content_lower:
                            return False, f"SECURITY: Blocked dangerous pattern: {pattern}", None
                    
                    return True, "", notebook_data
                    
                except json.JSONDecodeError:
                    return False, "Invalid JSON format", None
                    
            elif filename.lower().endswith('.py'):
                # Enhanced Python validation
                import ast
                try:
                    ast.parse(content_str)
                except SyntaxError:
                    return False, "Invalid Python syntax", None
                
                content_lower = content_str.lower()
                dangerous_patterns = [
                    'subprocess.', 'os.system', 'eval(', 'exec(', '__import__', 
                    'open(', 'file(', 'input(', 'getattr(', 'setattr(',
                    'globals(', 'locals(', 'vars(', 'dir(',
                    'compile(', 'reload(', '__builtins__', 'rm -rf',
                    'sudo ', 'chmod ', 'wget ', 'curl ', 'nc -l'
                ]
                
                for pattern in dangerous_patterns:
                    if pattern in content_lower:
                        return False, f"SECURITY: Blocked dangerous pattern: {pattern}", None
                
                return True, "", {"content": content_str}
            else:
                return False, "Unsupported file type", None
                
        except UnicodeDecodeError:
            return False, "Invalid file encoding - please use UTF-8", None
        except Exception as e:
            return False, f"File validation error: {str(e)}", None
    
    def sanitize_url_args(self, args: List[str]) -> str:
        """
        Reconstruct URL from shell arguments that may have been split.
        Handles cases where shell splits URLs with special characters.
        """
        # Join all arguments and clean up
        full_url = ' '.join(args).strip()
        
        # Remove any quotes that might have been added by shell
        if full_url.startswith('"') and full_url.endswith('"'):
            full_url = full_url[1:-1]
        if full_url.startswith("'") and full_url.endswith("'"):
            full_url = full_url[1:-1]
        
        return full_url 

    def _normalize_gpu_quantity(self, quantity: int) -> int:
        """Normalize GPU quantity to valid values. Delegates to module-level function."""
        return normalize_gpu_quantity(quantity)

    def _parse_runtime_range(self, runtime_str: str) -> tuple:
        """Parse runtime string like '1.5-2.5' into (min, max) float tuple."""
        return parse_runtime_range(runtime_str)

    def _calculate_multi_gpu_scaling(self, quantity: int) -> float:
        """Calculate multi-GPU scaling efficiency factor. Delegates to module-level function."""
        return calculate_multi_gpu_scaling(quantity)

    def _calculate_runtime_for_gpu(self, baseline_runtime: str, baseline_gpu: str, target_gpu: str, 
                                 quantity: int, optimization_factor: float = 1.0) -> str:
        """Calculate runtime for target GPU based on baseline estimate."""
        try:
            # Parse baseline runtime
            min_time, max_time = parse_runtime_range(baseline_runtime)
            
            # Get performance factors
            baseline_perf = self.gpu_specs.get(baseline_gpu, {}).get('performance_factor', 1.0)
            target_perf = self.gpu_specs.get(target_gpu, {}).get('performance_factor', 1.0)
            
            # Calculate GPU performance scaling
            gpu_speedup = target_perf / baseline_perf
            
            # Calculate multi-GPU scaling
            multi_gpu_factor = calculate_multi_gpu_scaling(quantity)
            
            # Apply all factors
            total_speedup = gpu_speedup * optimization_factor / multi_gpu_factor
            
            # Calculate new runtime range
            new_min = max(0.1, min_time / total_speedup)
            new_max = max(0.1, max_time / total_speedup)
            
            # Format output with appropriate units
            if new_min == new_max:
                if new_min < 1.0:
                    return f"{int(new_min * 60)} minutes"
                else:
                    return f"{new_min:.1f} hours"
            else:
                if new_min < 1.0 and new_max < 1.0:
                    return f"{int(new_min * 60)}-{int(new_max * 60)} minutes"
                elif new_min >= 1.0 and new_max >= 1.0:
                    return f"{new_min:.1f}-{new_max:.1f} hours"
                else:
                    min_str = f"{int(new_min * 60)} minutes" if new_min < 1.0 else f"{new_min:.1f} hours"
                    max_str = f"{int(new_max * 60)} minutes" if new_max < 1.0 else f"{new_max:.1f} hours"
                    return f"{min_str}-{max_str}"
                
        except Exception as e:
            # Fallback runtime
            return convert_runtime_to_new_format("1.0-2.0")

    def _generate_comprehensive_recommendations(self, analysis: Dict):
        """Generate minimum, consumer, and enterprise recommendations using new GPU specs-based system."""
        
        # Step 1: Use existing GPU benefit analysis from static analysis
        # (Don't re-run detection, use the results from _perform_static_analysis)
        gpu_benefit_analysis = analysis.get('_gpu_benefit_analysis')
        
        # If not available, run detection (fallback)
        if not gpu_benefit_analysis:
            all_code = '\n'.join(analysis.get('code_cells', []))
            all_markdown = '\n'.join(analysis.get('markdown_cells', []))
            
            gpu_benefit_analysis = self.detect_gpu_benefit_level(
                all_code + all_markdown, 
                analysis.get('detected_frameworks', []), 
                analysis.get('code_patterns', {})
            )
        
        # Step 2: Calculate enhanced VRAM requirements
        vram_estimate = self.calculate_vram_requirements_with_gpu_context(analysis)
        
        # Step 3: Apply CPU-first logic based on GPU benefit analysis
        if gpu_benefit_analysis['benefit_level'] == 'none':
            # CPU-only recommendation
            analysis['min_gpu_type'] = 'CPU-only'
            analysis['min_quantity'] = 0
            analysis['min_vram_gb'] = 0
            analysis['min_runtime_estimate'] = 'CPU execution'
            analysis['consumer_gpu_type'] = None
            analysis['consumer_quantity'] = None
            analysis['consumer_vram_gb'] = None
            analysis['consumer_runtime_estimate'] = None
            analysis['consumer_viable'] = False
            analysis['consumer_limitation'] = "CPU-optimized workload - GPU not needed"
            analysis['enterprise_gpu_type'] = ""
            analysis['enterprise_quantity'] = 0
            analysis['enterprise_vram_gb'] = 0
            analysis['enterprise_runtime_estimate'] = ""
            analysis['optimal_gpu_type'] = 'CPU-only'
            analysis['optimal_quantity'] = 0
            analysis['optimal_vram_gb'] = 0
            analysis['optimal_runtime_estimate'] = 'CPU execution'
            
            # Update workload type (reasoning already added in _perform_static_analysis)
            analysis['workload_type'] = gpu_benefit_analysis['workload_type']
            # Don't add reasoning here to avoid duplicates - it's already added in _perform_static_analysis
            
            return
        
        # Step 4: Generate tiered GPU recommendations based on benefit level
        benefit_level = gpu_benefit_analysis['benefit_level']
        workload_type = gpu_benefit_analysis['workload_type']
        vram_needed = gpu_benefit_analysis['vram_estimate']
        
        if benefit_level in ['beneficial', 'recommended', 'required']:
            use_case_context = analysis.get('use_case_context', 'professional')
            consumer_viable = analysis.get('consumer_viable', True)
            recommendations = self.generate_tiered_recommendations(
                vram_needed,
                workload_type,
                use_case_context,
                consumer_viable
            )
        else:
            # For CPU-only workloads, no GPU recommendations needed
            recommendations = {}
        
        # Step 5: Provide honest assessment
        honest_assessment = self.provide_honest_assessment(analysis, vram_estimate, gpu_benefit_analysis)
        
        # Step 6: Map new tiered recommendations to existing structure
        if recommendations:
            # Get quantities from static analysis (accounts for multi-GPU)
            base_quantity = analysis.get('min_quantity', 1)
            
            # MINIMUM tier -> min_gpu fields
            if 'budget_minimum' in recommendations:
                min_rec = recommendations['budget_minimum']
                analysis['min_gpu_type'] = min_rec['gpu_name']
                analysis['min_quantity'] = base_quantity
                analysis['min_vram_gb'] = min_rec['vram_gb'] * base_quantity  # Total VRAM
                analysis['min_runtime_estimate'] = self._format_runtime_estimate(min_rec['estimated_runtime_multiplier'])
            
            # RECOMMENDED tier -> always map to consumer fields (new 3-tier approach)
            if 'recommended' in recommendations:
                rec_rec = recommendations['recommended']
                
                # Always populate consumer/recommended fields regardless of category
                analysis['consumer_gpu_type'] = rec_rec['gpu_name']
                analysis['consumer_quantity'] = base_quantity
                analysis['consumer_vram_gb'] = rec_rec['vram_gb']
                analysis['consumer_runtime_estimate'] = self._format_runtime_estimate(rec_rec['estimated_runtime_multiplier'])
                
                # CRITICAL FIX: Don't override consumer viability based on recommended tier category
                # The consumer viability should have been determined earlier based on workload requirements
                # If consumer is not viable, then the recommended tier correctly shows an enterprise GPU
                # and we should preserve the original consumer_viable determination
                if not analysis.get('consumer_viable', True):
                    # Consumer not viable - recommended tier is showing enterprise GPU as expected
                    # Keep the existing consumer_limitation or set a default
                    if not analysis.get('consumer_limitation'):
                        analysis['consumer_limitation'] = f"Workload requires enterprise-grade hardware - {rec_rec['gpu_name']} recommended"
                else:
                    # Consumer viable - recommended tier might be consumer or enterprise
                    if rec_rec['category'] == 'consumer':
                        analysis['consumer_viable'] = True
                        analysis['consumer_limitation'] = None
                    else:
                        # Enterprise GPU in recommended tier even though consumer is viable
                        # This means enterprise is better but consumer would work
                        analysis['consumer_viable'] = True
                        analysis['consumer_limitation'] = f"Workload optimized for {rec_rec['gpu_name']} - consumer alternatives may be slower"
                
                # Always set enterprise/optimal as upgrade path
                if 'optimal' in recommendations:
                    opt_rec = recommendations['optimal']
                    analysis['enterprise_gpu_type'] = opt_rec['gpu_name']
                    analysis['enterprise_quantity'] = base_quantity
                    analysis['enterprise_vram_gb'] = opt_rec['vram_gb']
                    analysis['enterprise_runtime_estimate'] = self._format_runtime_estimate(opt_rec['estimated_runtime_multiplier'])
                else:
                    # If no optimal tier, use recommended as enterprise too
                    analysis['enterprise_gpu_type'] = rec_rec['gpu_name']
                    analysis['enterprise_quantity'] = base_quantity
                    analysis['enterprise_vram_gb'] = rec_rec['vram_gb']
                    analysis['enterprise_runtime_estimate'] = self._format_runtime_estimate(rec_rec['estimated_runtime_multiplier'])
            
            # OPTIMAL tier -> optimal_gpu fields (for backward compatibility)
            if 'optimal' in recommendations:
                opt_rec = recommendations['optimal']
                analysis['optimal_gpu_type'] = opt_rec['gpu_name']
                analysis['optimal_quantity'] = base_quantity
                analysis['optimal_vram_gb'] = opt_rec['vram_gb'] * base_quantity
                analysis['optimal_runtime_estimate'] = self._format_runtime_estimate(opt_rec['estimated_runtime_multiplier'])
            elif 'recommended' in recommendations:
                # Fall back to recommended if no optimal
                rec_rec = recommendations['recommended']
                analysis['optimal_gpu_type'] = rec_rec['gpu_name']
                analysis['optimal_quantity'] = base_quantity
                analysis['optimal_vram_gb'] = rec_rec['vram_gb'] * base_quantity
                analysis['optimal_runtime_estimate'] = self._format_runtime_estimate(rec_rec['estimated_runtime_multiplier'])
        else:
            # No GPU recommendations - already set to CPU-only above
            pass
        
        # Step 7: Add assessment insights to reasoning
        analysis['reasoning'].extend(honest_assessment['honesty_factors'])
        if honest_assessment['caveats']:
            analysis['reasoning'].extend([f"Caveat: {caveat}" for caveat in honest_assessment['caveats']])
        
        # Step 8: Store additional analysis data for output formatting
        analysis['_vram_estimate'] = vram_estimate
        analysis['_honest_assessment'] = honest_assessment
        analysis['_gpu_benefit_analysis'] = gpu_benefit_analysis
        analysis['_tiered_recommendations'] = recommendations

        # Step 9: Fix VRAM calculations to ensure they match actual GPU specs
        self._fix_vram_calculations(analysis)
        
        # Step 10: Ensure minimum GPU consistency with consumer viability
        self._ensure_minimum_gpu_consistency(analysis)
        
        # Step 11: Ensure consumer fields are never null
        self._ensure_consumer_fields_populated(analysis)

    def _format_runtime_estimate(self, runtime_multiplier):
        """Convert runtime multiplier to human-readable estimate"""
        if runtime_multiplier <= 0.5:
            return "5-15 minutes"
        elif runtime_multiplier <= 1.0:
            return "15-30 minutes"
        elif runtime_multiplier <= 2.0:
            return "30-60 minutes"
        else:
            return "1-3 hours"
    
    def _convert_runtime_to_new_format(self, runtime_str: str) -> str:
        """Convert runtime string to new format."""
        return convert_runtime_to_new_format(runtime_str)

    def _fix_vram_calculations(self, analysis: Dict):
        """Fix vRAM calculations to ensure they are based on actual GPU specs * quantity."""
        
        # Fix minimum vRAM (per-GPU VRAM * quantity = total VRAM)
        min_gpu_type = analysis.get('min_gpu_type', '')
        min_quantity = analysis.get('min_quantity', 1)
        if min_gpu_type and min_gpu_type in self.gpu_specs:
            per_gpu_vram = self.gpu_specs[min_gpu_type]['vram']
            analysis['min_vram_gb'] = per_gpu_vram * min_quantity
        
        # Fix consumer vRAM (per-GPU VRAM * quantity = total VRAM)
        consumer_gpu_type = analysis.get('consumer_gpu_type')
        consumer_quantity = analysis.get('consumer_quantity', 1)
        if consumer_gpu_type and consumer_gpu_type in self.gpu_specs:
            per_gpu_vram = self.gpu_specs[consumer_gpu_type]['vram']
            analysis['consumer_vram_gb'] = per_gpu_vram * consumer_quantity
        
        # Fix enterprise vRAM (per-GPU VRAM * quantity = total VRAM)
        enterprise_gpu_type = analysis.get('enterprise_gpu_type', '')
        enterprise_quantity = analysis.get('enterprise_quantity', 1)
        if enterprise_gpu_type and enterprise_gpu_type in self.gpu_specs:
            per_gpu_vram = self.gpu_specs[enterprise_gpu_type]['vram']
            analysis['enterprise_vram_gb'] = per_gpu_vram * enterprise_quantity
        
        # Fix optimal vRAM (per-GPU VRAM * quantity = total VRAM)
        optimal_gpu_type = analysis.get('optimal_gpu_type', '')
        optimal_quantity = analysis.get('optimal_quantity', 1)
        if optimal_gpu_type and optimal_gpu_type in self.gpu_specs:
            per_gpu_vram = self.gpu_specs[optimal_gpu_type]['vram']
            analysis['optimal_vram_gb'] = per_gpu_vram * optimal_quantity

    # _assess_consumer_viability_with_vram removed - functionality merged into _assess_consumer_viability

    def _calculate_dynamic_confidence(self, analysis: Dict, llm_context: Optional[Dict] = None) -> float:
        """
        Calculate dynamic confidence based on analysis quality factors.
        Returns a confidence score between 0.0 and 1.0.
        """
        base_confidence = 0.3  # Start with low baseline
        confidence_factors = []
        
        # Factor 1: Workload Detection Quality (0.0 - 0.25)
        workload_detected = analysis.get('workload_detected', False)
        workload_type = analysis.get('workload_type', 'none')
        
        if workload_type == 'none':
            workload_confidence = 0.0
            confidence_factors.append("No GPU workload detected")
        elif workload_type == 'demonstration':
            workload_confidence = 0.1
            confidence_factors.append("ML libraries present but no active workload")
        elif workload_type == 'basic':
            workload_confidence = 0.15
            confidence_factors.append("Basic GPU workload patterns detected")
        elif workload_type in ['training', 'inference']:
            workload_confidence = 0.25
            confidence_factors.append(f"Clear {workload_type} workload identified")
        else:
            workload_confidence = 0.1
            confidence_factors.append("Ambiguous workload type")
        
        base_confidence += workload_confidence
        
        # Factor 2: Framework Detection Quality (0.0 - 0.2)
        reasoning = analysis.get('reasoning', [])
        framework_mentions = sum(1 for reason in reasoning if any(fw in reason.lower() for fw in 
                               ['pytorch', 'tensorflow', 'transformers', 'keras', 'jax', 'cudf', 'rapids']))
        
        if framework_mentions >= 3:
            framework_confidence = 0.2
            confidence_factors.append("Multiple ML frameworks clearly identified")
        elif framework_mentions >= 2:
            framework_confidence = 0.15
            confidence_factors.append("Multiple frameworks detected")
        elif framework_mentions >= 1:
            framework_confidence = 0.1
            confidence_factors.append("Framework detected")
        else:
            framework_confidence = 0.0
            confidence_factors.append("No clear framework identification")
        
        base_confidence += framework_confidence
        
        # Factor 3: Model Identification Quality (0.0 - 0.2)
        model_mentions = sum(1 for reason in reasoning if any(model in reason.lower() for model in 
                           ['bert', 'gpt', 'llama', 'stable diffusion', 'resnet', 'transformer', 'model']))
        
        if model_mentions >= 3:
            model_confidence = 0.2
            confidence_factors.append("Multiple specific models identified")
        elif model_mentions >= 2:
            model_confidence = 0.15
            confidence_factors.append("Multiple models detected")
        elif model_mentions >= 1:
            model_confidence = 0.1
            confidence_factors.append("Model architecture identified")
        else:
            model_confidence = 0.0
            confidence_factors.append("No specific models identified")
        
        base_confidence += model_confidence
        
        # Factor 4: VRAM Estimation Confidence (0.0 - 0.15)
        min_vram = analysis.get('min_vram_gb', 0)
        if min_vram > 0:
            if min_vram >= 80:
                vram_confidence = 0.15
                confidence_factors.append("High VRAM requirement clearly identified")
            elif min_vram >= 24:
                vram_confidence = 0.12
                confidence_factors.append("Substantial VRAM requirement identified")
            elif min_vram >= 8:
                vram_confidence = 0.1
                confidence_factors.append("VRAM requirement estimated")
            else:
                vram_confidence = 0.05
                confidence_factors.append("Basic VRAM requirement")
        else:
            vram_confidence = 0.0
            confidence_factors.append("No VRAM requirement identified")
        
        base_confidence += vram_confidence
        
        # Factor 5: Multi-GPU Detection Confidence (0.0 - 0.1)
        multi_gpu_detected = analysis.get('min_quantity', 1) > 1
        sxm_required = analysis.get('sxm_required', False)
        
        if sxm_required:
            multi_gpu_confidence = 0.1
            confidence_factors.append("Large-scale multi-GPU requirement identified")
        elif multi_gpu_detected:
            multi_gpu_confidence = 0.08
            confidence_factors.append("Multi-GPU setup detected")
        else:
            multi_gpu_confidence = 0.05
            confidence_factors.append("Single-GPU workload")
        
        base_confidence += multi_gpu_confidence
        
        # Factor 6: LLM Enhancement Impact (can modify total confidence)
        if llm_context:
            llm_confidence = llm_context.get('confidence', 0.5)
            llm_reasoning = analysis.get('llm_reasoning', [])
            
            # Check for LLM-static analysis agreement
            llm_vram = llm_context.get('estimated_vram_gb', 0)
            static_vram = min_vram
            
            if llm_vram > 0 and static_vram > 0:
                vram_agreement = 1.0 - min(abs(llm_vram - static_vram) / max(llm_vram, static_vram), 1.0)
                if vram_agreement > 0.8:
                    confidence_factors.append("LLM analysis strongly agrees with static analysis")
                    base_confidence += 0.1
                elif vram_agreement > 0.6:
                    confidence_factors.append("LLM analysis moderately agrees with static analysis")
                    base_confidence += 0.05
                else:
                    confidence_factors.append("LLM analysis disagrees with static analysis")
                    base_confidence -= 0.05
            
            # Incorporate LLM's own confidence
            if llm_confidence > 0.7:
                confidence_factors.append("LLM analysis has high confidence")
                base_confidence += 0.1
            elif llm_confidence > 0.5:
                confidence_factors.append("LLM analysis has moderate confidence")
                base_confidence += 0.05
            else:
                confidence_factors.append("LLM analysis has low confidence")
                base_confidence -= 0.05
            
            # Check for memory optimizations (increases confidence)
            if llm_context.get('memory_optimizations'):
                confidence_factors.append("Memory optimizations identified by LLM")
                base_confidence += 0.05
        else:
            confidence_factors.append("No LLM enhancement available")
        
        # Factor 7: Pattern Clarity Penalties
        reasoning_text = ' '.join(reasoning).lower()
        
        # Penalty for uncertainty indicators
        uncertainty_patterns = ['might', 'could', 'possibly', 'unclear', 'ambiguous', 'basic']
        uncertainty_count = sum(1 for pattern in uncertainty_patterns if pattern in reasoning_text)
        if uncertainty_count > 0:
            penalty = min(uncertainty_count * 0.05, 0.15)
            base_confidence -= penalty
            confidence_factors.append(f"Uncertainty indicators detected ({uncertainty_count})")
        
        # Bonus for definitive patterns
        definitive_patterns = ['detected', 'identified', 'requires', 'training', 'inference']
        definitive_count = sum(1 for pattern in definitive_patterns if pattern in reasoning_text)
        if definitive_count >= 3:
            bonus = min(definitive_count * 0.02, 0.08)
            base_confidence += bonus
            confidence_factors.append(f"Strong pattern detection ({definitive_count} definitive indicators)")
        
        # Ensure confidence stays within bounds
        final_confidence = max(0.1, min(base_confidence, 1.0))
        
        # Store confidence factors for debugging/transparency
        analysis['confidence_factors'] = confidence_factors
        
        return final_confidence

    def analyze_notebook_with_progress(self, url_or_path: str, progress_callback=None) -> GPURequirement:
        """
        Main entry point for notebook analysis with progress streaming support.
        Coordinates all analysis components and returns comprehensive results.
        """
        if progress_callback:
            progress_callback("📁 Starting notebook analysis...")
        elif not self.quiet_mode:
            print(f"📁 Analyzing notebook: {url_or_path}")
        
        # Extract and parse notebook content
        if progress_callback:
            progress_callback("📂 Loading notebook content...")
        
        try:
            code_cells, markdown_cells = self._extract_notebook_content(url_or_path)
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Failed to load notebook: {str(e)[:50]}...")
            elif not self.quiet_mode:
                print(f"❌ Failed to load notebook: {e}")
            raise
        
        if progress_callback:
            progress_callback(f"✅ Loaded {len(code_cells)} code cells, {len(markdown_cells)} markdown cells")
        elif not self.quiet_mode:
            print(f"✅ Successfully loaded notebook ({len(code_cells)} code cells, {len(markdown_cells)} markdown cells)")
        
        # Perform static analysis
        if progress_callback:
            progress_callback("🔍 Analyzing GPU requirements...")
        
        static_analysis = self._perform_static_analysis(code_cells, markdown_cells)
        
        # Enhance with LLM if available
        llm_context = None
        llm_reasoning = []
        self_review = None
        if self.llm_analyzer:
            if progress_callback:
                progress_callback("🤖 Evaluating workload complexity...")
            elif not self.quiet_mode:
                print("🤖 Enhancing analysis with LLM...")
            
            # Pass progress callback to LLM analyzer
            llm_context = self.llm_analyzer.analyze_notebook_context(code_cells, markdown_cells, progress_callback)
            if llm_context:
                if progress_callback:
                    progress_callback("⚡ Enhancing GPU recommendations...")
                
                enhanced_analysis, llm_reasoning = self.llm_analyzer.enhance_gpu_recommendation(static_analysis, llm_context)
                static_analysis.update(enhanced_analysis)
                
                # CRITICAL FIX: Regenerate comprehensive recommendations after LLM enhancement
                # The LLM may have significantly changed VRAM requirements, so we need to update
                # the consumer/enterprise recommendations based on the new estimates
                self._generate_comprehensive_recommendations(static_analysis)
                
                # CONSISTENCY FIX: Ensure minimum VRAM shows total available VRAM when quantity > 1
                # This must happen AFTER LLM enhancement to get the correct final VRAM value
                if static_analysis.get('min_quantity', 1) > 1:
                    # Convert per-GPU VRAM to total VRAM for multi-GPU minimum setups
                    min_gpu_type = static_analysis.get('min_gpu_type', '')
                    if min_gpu_type in self.gpu_specs:
                        per_gpu_vram = self.gpu_specs[min_gpu_type]['vram']
                        min_quantity = static_analysis.get('min_quantity', 1)
                        static_analysis['min_vram_gb'] = per_gpu_vram * min_quantity
                
                # RUNTIME CONSISTENCY FIX: Update minimum runtime to match consumer/enterprise when same hardware
                # If consumer recommendation exists and uses same GPU, use consumer runtime for consistency
                consumer_gpu_type = static_analysis.get('consumer_gpu_type')
                consumer_quantity = static_analysis.get('consumer_quantity')
                min_gpu_type = static_analysis.get('min_gpu_type', '')
                min_quantity = static_analysis.get('min_quantity', 1)
                
                if (consumer_gpu_type == min_gpu_type and consumer_quantity == min_quantity and
                    static_analysis.get('consumer_runtime_estimate')):
                    # Same hardware - use consumer runtime for consistency
                    static_analysis['min_runtime_estimate'] = static_analysis['consumer_runtime_estimate']
                
                # Recalculate confidence with LLM context
                enhanced_confidence = self._calculate_dynamic_confidence(static_analysis, llm_context)
                static_analysis['confidence'] = enhanced_confidence
                
                # PHASE 2.5: Self-review analysis (only in development environment)
                if self.llm_analyzer.env_config['self_review_enabled']:
                    if progress_callback:
                        progress_callback("🎓 Performing self-review for accuracy...")
                    elif not self.quiet_mode:
                        print("🎓 Performing self-review for accuracy and consistency...")
                    
                    self_review = self.llm_analyzer.self_review_analysis(
                        code_cells, static_analysis, static_analysis.get('reasoning', []), llm_reasoning, progress_callback
                    )
                else:
                    # Skip self-review in production for performance
                    if progress_callback:
                        progress_callback("✅ AI analysis complete")
                    elif not self.quiet_mode:
                        print(f"✅ LLM analysis complete (confidence: {enhanced_confidence*100:.0f}%)")
                    self_review = None
                
                if self_review:
                    if progress_callback:
                        progress_callback("🔧 Applying self-review corrections...")
                    
                    # Apply self-review corrections
                    corrected_analysis, final_reasoning = self.llm_analyzer.apply_self_review_corrections(
                        static_analysis, static_analysis.get('reasoning', []), llm_reasoning, self_review
                    )
                    
                    # Update analysis with corrected values
                    static_analysis.update(corrected_analysis)
                    
                    # Keep original static reasoning, but replace LLM reasoning with unified reasoning from self-review
                    llm_reasoning = final_reasoning
                    
                    if progress_callback:
                        review_status = "passed" if self_review.get('review_passed', True) else "corrected issues"
                        progress_callback(f"✅ Self-review {review_status}")
                    elif not self.quiet_mode:
                        review_status = "passed" if self_review.get('review_passed', True) else "corrected issues"
                        print(f"✅ Self-review {review_status} - enhanced accuracy and consistency")
                else:
                    self_review = None
        
        # Evaluate NVIDIA Best Practices compliance
        if progress_callback:
            progress_callback("📋 Evaluating NVIDIA compliance...")
        elif not self.quiet_mode:
            print("📋 Evaluating NVIDIA compliance...")
        
        structure_assessment = self.evaluate_notebook_structure(code_cells, markdown_cells)
        content_issues = self.assess_content_quality(code_cells, markdown_cells)
        technical_recommendations = self.check_technical_standards(code_cells)
        
        # Get LLM compliance evaluation if available
        llm_compliance = None
        if self.llm_analyzer:
            llm_compliance = self.llm_analyzer.evaluate_notebook_compliance(code_cells, markdown_cells)
        
        # Calculate comprehensive compliance score
        compliance_score = self.calculate_nvidia_compliance_score(
            structure_assessment, content_issues, technical_recommendations, llm_compliance
        )
        
        if progress_callback:
            progress_callback("🏁 Generating final recommendations...")
        elif not self.quiet_mode:
            print(f"✅ Compliance evaluation complete (score: {compliance_score:.0f}/100)")
        
        # Final confidence is already set by self-review if it occurred
        # If no self-review, use intelligent confidence calculation
        if not self_review:
            if llm_context and llm_context.get('confidence', 0) > 0.8:
                # LLM has high confidence - use it
                static_analysis['confidence'] = llm_context['confidence']
            else:
                # Fall back to dynamic calculation
                final_confidence = self._calculate_dynamic_confidence(static_analysis, llm_context)
                static_analysis['confidence'] = final_confidence
        
        # Build final result
        return GPURequirement(
            min_gpu_type=static_analysis['min_gpu_type'],
            min_quantity=static_analysis['min_quantity'],
            min_vram_gb=static_analysis['min_vram_gb'],
            min_runtime_estimate=static_analysis['min_runtime_estimate'],
            recommended_gpu_type=static_analysis.get('consumer_gpu_type'),
            recommended_quantity=static_analysis.get('consumer_quantity'),
            recommended_vram_gb=static_analysis.get('consumer_vram_gb'),
            recommended_runtime_estimate=static_analysis.get('consumer_runtime_estimate'),
            recommended_viable=static_analysis.get('consumer_viable', True),
            recommended_limitation=static_analysis.get('consumer_limitation'),
            optimal_gpu_type=static_analysis.get('enterprise_gpu_type', ""),
            optimal_quantity=static_analysis.get('enterprise_quantity', 1),
            optimal_vram_gb=static_analysis.get('enterprise_vram_gb', 0),
            optimal_runtime_estimate=static_analysis.get('enterprise_runtime_estimate', ""),
            sxm_required=static_analysis['sxm_required'],
            sxm_reasoning=static_analysis['sxm_reasoning'],
            arm_compatibility=static_analysis['arm_compatibility'],
            arm_reasoning=static_analysis['arm_reasoning'],
            confidence=static_analysis['confidence'],
            reasoning=static_analysis['reasoning'],
            llm_enhanced=llm_context is not None,
            llm_reasoning=llm_reasoning,
            self_reviewed=llm_context is not None and self_review is not None,
            llm_model_used=self.llm_analyzer.model if self.llm_analyzer and llm_context is not None else None,
            nvidia_compliance_score=compliance_score,
            structure_assessment=structure_assessment,
            content_quality_issues=content_issues,
            technical_recommendations=technical_recommendations,
            confidence_factors=static_analysis.get('confidence_factors', []),
            workload_detected=static_analysis['workload_detected']
        )

    def get_gpus_by_category(self, category=None, tier=None):
        """
        Filter GPUs from self.gpu_specs by category and/or tier
        """
        filtered_gpus = {}
        for gpu_name, specs in self.gpu_specs.items():
            if category and specs['category'] != category:
                continue
            if tier and specs['tier'] != tier:
                continue
            filtered_gpus[gpu_name] = specs
        return filtered_gpus

    def find_minimum_viable_gpu(self, vram_needed, prefer_consumer=True):
        """
        Find the absolute cheapest GPU that meets VRAM requirements
        """
        viable_gpus = []
        
        # Check consumer GPUs first (usually more cost-effective)
        if prefer_consumer:
            consumer_gpus = self.get_gpus_by_category(category='consumer')
            for gpu_name, specs in consumer_gpus.items():
                if specs['vram'] >= vram_needed:
                    viable_gpus.append((gpu_name, specs, self._calculate_price_score(specs)))
        
        # If no consumer options or enterprise preferred, check enterprise
        if not viable_gpus or not prefer_consumer:
            enterprise_gpus = self.get_gpus_by_category(category='enterprise')
            for gpu_name, specs in enterprise_gpus.items():
                if specs['vram'] >= vram_needed and specs['tier'] in ['mid', 'high']:  # Exclude cutting_edge for minimum
                    viable_gpus.append((gpu_name, specs, self._calculate_price_score(specs)))
        
        # Sort by price score (lower is cheaper), then by VRAM
        if viable_gpus:
            return sorted(viable_gpus, key=lambda x: (x[2], x[1]['vram']))[0]
        return None

    def _calculate_price_score(self, specs):
        """
        Calculate relative price score using real cost factors (lower score = cheaper)
        """
        # Use the cost_factor if available (real-world relative pricing)
        if 'cost_factor' in specs:
            return specs['cost_factor']
        
        # Fallback calculation for any GPUs missing cost_factor
        base_score = specs['vram'] * 2  # VRAM is major cost factor
        
        # Tier multipliers
        tier_multipliers = {
            'entry': 0.5,
            'mid': 1.0,
            'high': 1.5,
            'flagship': 2.0,
            'enterprise': 2.5,
            'cutting_edge': 4.0
        }
        
        # Category multipliers
        category_multipliers = {
            'consumer': 1.0,
            'enterprise': 1.8  # Enterprise typically more expensive
        }
        
        multiplier = tier_multipliers.get(specs['tier'], 1.0) * category_multipliers.get(specs['category'], 1.0)
        
        # Performance factor affects price
        performance_bonus = specs['performance_factor'] * 10
        
        return base_score * multiplier + performance_bonus

    def _find_best_consumer_gpu(self, vram_needed):
        """Find best consumer GPU for given VRAM requirement"""
        consumer_gpus = self.get_gpus_by_category(category='consumer')
        suitable_gpus = []
        
        for gpu_name, specs in consumer_gpus.items():
            if specs['vram'] >= vram_needed:
                # Score based on price/performance balance
                efficiency_score = specs['performance_factor'] / self._calculate_price_score(specs)
                suitable_gpus.append((gpu_name, specs, efficiency_score))
        
        if suitable_gpus:
            return sorted(suitable_gpus, key=lambda x: x[2], reverse=True)[0]  # Best efficiency
        return None

    def _find_best_professional_gpu(self, vram_needed, workload_type):
        """Find best professional/enterprise GPU"""
        enterprise_gpus = self.get_gpus_by_category(category='enterprise')
        suitable_gpus = []
        
        for gpu_name, specs in enterprise_gpus.items():
            if specs['vram'] >= vram_needed:
                # For enterprise, prioritize performance and features
                score = specs['performance_factor']
                
                # Bonus for features relevant to workload
                if workload_type == "training" and specs['nvlink']:
                    score *= 1.2  # NVLink beneficial for multi-GPU training
                if specs['tensor_cores']:
                    score *= 1.1  # Tensor cores beneficial for ML
                
                suitable_gpus.append((gpu_name, specs, score))
        
        if suitable_gpus:
            return sorted(suitable_gpus, key=lambda x: x[2], reverse=True)[0]  # Best performance
        return None

    def _find_optimal_gpu(self, vram_needed, workload_type):
        """Find absolute best GPU regardless of cost"""
        all_suitable_gpus = []
        
        for gpu_name, specs in self.gpu_specs.items():
            if specs['vram'] >= vram_needed:
                score = specs['performance_factor']
                
                # Workload-specific bonuses
                if workload_type in ["training", "large_models"] and specs['nvlink']:
                    score *= 1.3
                if workload_type == "inference" and specs['tier'] in ['cutting_edge', 'flagship']:
                    score *= 1.2
                
                all_suitable_gpus.append((gpu_name, specs, score))
        
        if all_suitable_gpus:
            return sorted(all_suitable_gpus, key=lambda x: x[2], reverse=True)[0]
        return None

    def detect_gpu_benefit_level(self, notebook_content, imports=None, code_patterns=None):
        """
        Detect the level of GPU benefit for a notebook workload.
        CPU-first approach: assume CPU-only until proven otherwise.
        
        Returns:
            dict: {
                'benefit_level': 'none'|'beneficial'|'recommended'|'required',
                'confidence': float (0.0-1.0),
                'reasoning': List[str],
                'estimated_speedup': Optional[str],
                'workload_type': str,
                'vram_estimate': int (GB, 0 if CPU-only)
            }
        """
        import re
        
        full_content = str(notebook_content).lower()
        
        # Score different categories of GPU indicators
        gpu_required_score = 0      # Must have GPU
        gpu_recommended_score = 0   # Significant benefit
        gpu_beneficial_score = 0    # Some benefit
        cpu_optimized_score = 0     # Better on CPU
        
        # === GPU REQUIRED PATTERNS (Cannot run on CPU) ===
        gpu_required_patterns = [
            # Large models that require GPU memory
            r'llama.*[1-9][0-9]b',  # LLaMA 10B+
            r'gpt.*[1-9][0-9]b',    # GPT 10B+
            r'bloom.*[1-9][0-9]b',  # BLOOM 10B+
            r'opt.*[1-9][0-9]b',    # OPT 10B+
            # Explicit CUDA code
            r'@cuda\.jit',
            r'cuda\.grid',
            r'cuda\.blockidx',
            r'cuda\.threadidx',
            r'pycuda\.',
            r'cupy\.',
            r'tensorrt\.',
            # Multi-GPU distributed training
            r'distributeddataparallel',
            r'torch\.distributed',
            # Explicit GPU-only libraries (actual usage, not just mentions)
            r'import\s+cudf',
            r'from\s+cudf',
            r'import\s+cuml',
            r'from\s+cuml',
            r'import\s+cugraph',
            r'from\s+cugraph',
            r'cudf\.[a-zA-Z_]+\(',  # Actual cuDF function calls
            r'cuml\.[a-zA-Z_]+\(',  # Actual cuML function calls
            r'cugraph\.[a-zA-Z_]+\(',  # Actual cuGraph function calls
            # Very large batch sizes or model sizes
            r'batch_size.*[5-9][0-9]{2,}',  # 500+ batch size
            r'hidden_size.*[1-9][0-9]{3,}', # 1000+ hidden size
        ]
        
        # === GPU RECOMMENDED PATTERNS (Significant benefit) ===
        gpu_recommended_patterns = [
            # Training loops
            r'\.train\(\)',
            r'model\.fit\(',
            r'optimizer\.step\(',
            r'loss\.backward\(',
            r'for epoch in',
            r'training_loop',
            # Large neural networks
            r'torch\.nn\..*conv',
            r'torch\.nn\..*lstm',
            r'torch\.nn\..*gru',
            r'torch\.nn\..*transformer',
            r'tensorflow\.keras\.layers',
            # Computer vision
            r'torchvision',
            r'cv2\..*resize.*[5-9][0-9]{2,}',  # Large image processing
            r'image.*classification',
            r'object.*detection',
            r'semantic.*segmentation',
            # NLP with transformers
            r'transformers\.',
            r'bert.*model',
            r'gpt.*model',
            r'from.*transformers.*import',
            # Medium-large models
            r'llama.*[1-9]b',  # 1-9B parameters
            r'gpt.*[1-9]b',
            # GPU acceleration libraries
            r'torch\.cuda',
            r'tensorflow.*gpu',
            r'device.*cuda',
        ]
        
        # === GPU BENEFICIAL PATTERNS (Some benefit) ===
        gpu_beneficial_patterns = [
            # Basic ML training
            r'sklearn\..*fit\(',
            r'xgboost',
            r'lightgbm',
            r'catboost',
            # Basic neural networks
            r'torch\.nn\.linear',
            r'torch\.nn\.sequential',
            r'keras\.sequential',
            # Image processing
            r'cv2\.',
            r'pillow',
            r'imageio',
            # Large dataset processing
            r'dataloader',
            r'dataset.*[1-9][0-9]{4,}',  # 10k+ samples
            # Basic GPU operations
            r'\.cuda\(\)',
            r'\.gpu\(\)',
            r'device.*gpu',
        ]
        
        # === CPU OPTIMIZED PATTERNS (Better on CPU) ===
        cpu_optimized_patterns = [
            # Pure data analysis
            r'pandas\.',
            r'numpy\.',
            r'matplotlib\.',
            r'seaborn\.',
            r'plotly\.',
            r'scipy\.',
            r'statsmodels\.',
            # Small datasets
            r'\.shape.*\([1-9][0-9]{0,2},',  # <1000 rows
            r'len\(.*\).*[1-9][0-9]{0,2}',   # <1000 items
            # Traditional ML (small scale)
            r'sklearn\.linear_model',
            r'sklearn\.tree',
            r'sklearn\.naive_bayes',
            r'sklearn\.svm',
            # Statistical analysis
            r'correlation',
            r'regression',
            r'anova',
            r't-test',
            r'chi-square',
            # Visualization
            r'plt\.plot',
            r'plt\.scatter',
            r'plt\.hist',
            r'sns\.',
        ]
        
        # Count pattern matches
        for pattern in gpu_required_patterns:
            gpu_required_score += len(re.findall(pattern, full_content)) * 5
        
        for pattern in gpu_recommended_patterns:
            gpu_recommended_score += len(re.findall(pattern, full_content)) * 3
        
        for pattern in gpu_beneficial_patterns:
            gpu_beneficial_score += len(re.findall(pattern, full_content)) * 2
        
        for pattern in cpu_optimized_patterns:
            cpu_optimized_score += len(re.findall(pattern, full_content)) * 1
        
        # Determine benefit level
        total_gpu_score = gpu_required_score + gpu_recommended_score + gpu_beneficial_score
        
        # CRITICAL FIX: Adjust scoring for small-scale GPU library usage
        # cuDF/RAPIDS usage on small datasets shouldn't trigger 'required' level
        if gpu_required_score > 0:
            # Check for small dataset indicators that suggest this is a tutorial/demo
            small_dataset_indicators = [
                r'10\s+samples?', r'small.*dataset', r'tutorial', r'example', 
                r'demo', r'test.*data', r'sample.*data', r'\.head\(\)', 
                r'\.shape.*\([1-9][0-9]{0,2}', r'len\(.*\).*[1-9][0-9]{0,2}'
            ]
            
            small_dataset_score = 0
            for pattern in small_dataset_indicators:
                small_dataset_score += len(re.findall(pattern, full_content))
            
            # If we have strong indicators this is a small dataset/tutorial, 
            # downgrade from 'required' to 'recommended'
            if small_dataset_score >= 2 and gpu_required_score <= 20:
                gpu_required_score = 0  # Clear required score
                gpu_recommended_score += 10  # Boost recommended score instead
        
        if gpu_required_score > 0:
            benefit_level = 'required'
            confidence = min(0.95, 0.7 + (gpu_required_score / 20))
            workload_type = 'gpu-computing' if 'cuda' in full_content else 'large-scale-ml'
            # Fix VRAM estimation - be more conservative for simple GPU computing
            if 'cuda' in full_content and 'numba' in full_content:
                # Simple CUDA/Numba workloads typically need much less VRAM
                vram_estimate = 8 if gpu_required_score > 15 else 4
            else:
                vram_estimate = 48 if gpu_required_score > 10 else 24
            estimated_speedup = 'Required - cannot run on CPU'
            reasoning = ['GPU required for this workload']
        
        elif gpu_recommended_score > 3:  # Lowered from 5 to 3
            benefit_level = 'recommended'
            confidence = min(0.9, 0.6 + (gpu_recommended_score / 30))
            # Improved workload type detection
            training_patterns = ['train', 'fit', 'epoch', 'optimizer', 'loss', 'backward', 'gradient', 'learning_rate']
            inference_patterns = ['predict', 'inference', 'eval', 'forward', 'model.load', 'torch.no_grad']
            
            training_score = sum(1 for p in training_patterns if p in full_content.lower())
            inference_score = sum(1 for p in inference_patterns if p in full_content.lower())
            
            if training_score > inference_score:
                workload_type = 'training'
            elif inference_score > 0:
                workload_type = 'inference'
            elif 'cuda' in full_content and 'numba' in full_content:
                workload_type = 'gpu-computing'
            else:
                workload_type = 'inference'  # Default to inference for ML workloads
            
            vram_estimate = 16 if gpu_recommended_score > 15 else 8
            estimated_speedup = '3-10x faster than CPU'
            reasoning = ['GPU provides significant performance benefit']
            
        elif gpu_beneficial_score > 1 and cpu_optimized_score < 10:  # Lowered from 3 to 1
            benefit_level = 'beneficial'
            confidence = min(0.8, 0.5 + (gpu_beneficial_score / 20))
            workload_type = 'small-ml'
            vram_estimate = 8
            estimated_speedup = '2-5x faster than CPU'
            reasoning = ['GPU provides moderate performance benefit']
            
        else:
            # Default to CPU-only
            benefit_level = 'none'
            workload_type = 'data-analysis' if cpu_optimized_score > 5 else 'general'
            vram_estimate = 0
            estimated_speedup = None
            reasoning = ['CPU-optimized workload - GPU provides minimal benefit']
            
            # High confidence for pure data analysis
            if cpu_optimized_score > 10 and total_gpu_score == 0:
                confidence = 0.9
                reasoning = ['Pure data analysis workload - CPU is optimal']
            else:
                confidence = 0.7
        
        # Add specific reasoning based on detected patterns (avoid duplicates)
        if gpu_required_score > 0:
            reasoning.append(f"High GPU requirement indicators detected (score: {gpu_required_score})")
        if gpu_recommended_score > 0:
            reasoning.append(f"Training/ML patterns detected (score: {gpu_recommended_score})")
        if gpu_beneficial_score > 0:
            reasoning.append(f"GPU-accelerated operations detected (score: {gpu_beneficial_score})")
        if cpu_optimized_score > 0 and benefit_level != 'none':
            # Only add CPU-optimized reasoning if not already covered by the main CPU-only reasoning
            reasoning.append(f"CPU-optimized libraries detected (score: {cpu_optimized_score})")
        
        return {
            'benefit_level': benefit_level,
            'confidence': confidence,
            'reasoning': reasoning,
            'estimated_speedup': estimated_speedup,
            'workload_type': workload_type,
            'vram_estimate': vram_estimate
        }



    def calculate_vram_requirements_with_gpu_context(self, notebook_analysis):
        """
        Enhanced VRAM estimation that considers GPU architecture capabilities
        """
        base_estimate = self._calculate_base_vram_requirement(notebook_analysis)
        
        # Apply optimizations based on modern GPU capabilities
        optimized_estimate = base_estimate
        
        # Tensor Core optimization (available on newer GPUs)
        if notebook_analysis.get("uses_mixed_precision", False):
            optimized_estimate *= 0.6  # Mixed precision significantly reduces VRAM
        
        # Memory optimization techniques
        if notebook_analysis.get("uses_lora", False):
            optimized_estimate *= 0.3  # LoRA dramatically reduces memory needs
        
        if notebook_analysis.get("uses_quantization", False):
            quant_factor = {
                "int8": 0.5,
                "int4": 0.25,
                "fp16": 0.5
            }.get(notebook_analysis.get("quantization_type", "int8"), 0.5)
            optimized_estimate *= quant_factor
        
        if notebook_analysis.get("uses_gradient_checkpointing", False):
            optimized_estimate *= 0.7  # Gradient checkpointing trades compute for memory
        
        # Modern GPU architectural benefits
        minimum_gpu = self.find_minimum_viable_gpu(optimized_estimate)
        if minimum_gpu and minimum_gpu[1]['compute_capability'] >= 8.0:
            # Newer architectures have better memory efficiency
            optimized_estimate *= 0.9
        
        return {
            "base_vram_estimate": base_estimate,
            "optimized_vram_estimate": optimized_estimate,
            "optimization_savings": base_estimate - optimized_estimate,
            "confidence": self._calculate_estimation_confidence(notebook_analysis),
            "assumptions": self._list_vram_assumptions(notebook_analysis)
        }

    def _calculate_base_vram_requirement(self, notebook_analysis):
        """Calculate base VRAM requirement from notebook analysis"""
        # Start with detected model requirements
        base_vram = notebook_analysis.get('min_vram_gb', 8)
        
        # Adjust based on workload type
        workload_type = notebook_analysis.get('workload_type', 'inference')
        if workload_type == 'training':
            base_vram *= 1.5  # Training needs more memory
        elif workload_type == 'fine-tuning':
            base_vram *= 1.3  # Fine-tuning needs moderate increase
        
        # Adjust for multi-GPU setups
        if notebook_analysis.get('min_quantity', 1) > 1:
            # Multi-GPU can distribute memory load
            base_vram *= 0.8
        
        return max(4, base_vram)  # Minimum 4GB

    def _calculate_estimation_confidence(self, notebook_analysis):
        """Calculate confidence in VRAM estimation"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if we detected specific models
        if notebook_analysis.get('detected_models'):
            confidence += 0.2
        
        # Increase confidence if we have clear workload patterns
        if notebook_analysis.get('workload_type') in ['training', 'inference']:
            confidence += 0.2
        
        # Decrease confidence if analysis is uncertain
        if notebook_analysis.get('confidence', 0) < 0.7:
            confidence -= 0.1
        
        return min(1.0, max(0.1, confidence))

    def _list_vram_assumptions(self, notebook_analysis):
        """List assumptions made in VRAM estimation"""
        assumptions = []
        
        workload_type = notebook_analysis.get('workload_type', 'unknown')
        assumptions.append(f"Workload type: {workload_type}")
        
        if notebook_analysis.get('min_quantity', 1) > 1:
            assumptions.append("Multi-GPU setup assumed for memory distribution")
        
        if not notebook_analysis.get('detected_models'):
            assumptions.append("No specific models detected - using generic estimates")
        
        return assumptions

    def generate_tiered_recommendations(self, vram_requirement, workload_type, use_case_context, consumer_viable=True):
        """
        Generate tiered recommendations like Windows power modes:
        - Minimum: Power Saver (cheapest that works)
        - Recommended: Balanced (best price/performance)
        - Optimal: Performance (best performance within reason)
        
        Args:
            vram_requirement: VRAM needed in GB
            workload_type: Type of workload (training, inference, etc.)
            use_case_context: Context about the use case
            consumer_viable: Whether consumer GPUs are viable for this workload
        """
        recommendations = {}
        
        # MINIMUM - "Power Saver Mode" - Absolute cheapest that will work
        min_gpu = self._find_minimum_viable_gpu_strict(vram_requirement, consumer_viable)
        if min_gpu:
            recommendations["budget_minimum"] = {
                "gpu_name": min_gpu[0],
                "specs": min_gpu[1],
                "vram_gb": min_gpu[1]['vram'],
                "category": min_gpu[1]['category'],
                "use_case": "Minimum viable GPU - will work but may be slow",
                "estimated_runtime_multiplier": 1.0 / min_gpu[1]['performance_factor']
            }
        
        # RECOMMENDED - "Balanced Mode" - Best price/performance ratio
        rec_gpu = self._find_balanced_gpu(vram_requirement, workload_type, consumer_viable)
        if rec_gpu:
            recommendations["recommended"] = {
                "gpu_name": rec_gpu[0],
                "specs": rec_gpu[1],
                "vram_gb": rec_gpu[1]['vram'],
                "category": rec_gpu[1]['category'],
                "use_case": "Balanced price/performance - recommended for most users",
                "estimated_runtime_multiplier": 1.0 / rec_gpu[1]['performance_factor']
            }
        
        # OPTIMAL - "Performance Mode" - Best performance within reason
        optimal_gpu = self._find_performance_gpu(vram_requirement, workload_type, consumer_viable)
        if optimal_gpu:
            recommendations["optimal"] = {
                "gpu_name": optimal_gpu[0],
                "specs": optimal_gpu[1],
                "vram_gb": optimal_gpu[1]['vram'],
                "category": optimal_gpu[1]['category'],
                "use_case": "High performance within reasonable cost bounds",
                "estimated_runtime_multiplier": 1.0 / optimal_gpu[1]['performance_factor']
            }
        
        return recommendations

    def provide_honest_assessment(self, notebook_analysis, vram_estimate, gpu_benefit_analysis):
        """
        Provide transparent, honest recommendations using GPU specs context
        """
        import math
        
        assessment = {
            "requires_gpu": True,
            "confidence_level": "high",
            "honesty_factors": [],
            "caveats": [],
            "alternative_recommendations": []
        }
        
        # CPU-only assessment based on GPU benefit analysis
        if gpu_benefit_analysis['benefit_level'] == 'none':
            assessment.update({
                "requires_gpu": False,
                "confidence_level": "high" if gpu_benefit_analysis['confidence'] > 0.8 else "medium",
                "honesty_factors": [
                    f"Workload appears CPU-optimized (confidence: {gpu_benefit_analysis['confidence']:.1%})",
                    # REMOVED: "GPU may provide minimal benefit for this workload" - this is already in reasoning from detect_gpu_benefit_level
                ],
                "alternative_recommendations": ["Consider CPU-only execution first", "Evaluate actual performance before GPU investment"]
            })
        elif gpu_benefit_analysis['benefit_level'] == 'beneficial':
            assessment.update({
                "requires_gpu": False,
                "confidence_level": "medium",
                "honesty_factors": [
                    f"GPU provides moderate benefit (confidence: {gpu_benefit_analysis['confidence']:.1%})",
                    f"Estimated speedup: {gpu_benefit_analysis['estimated_speedup']}"
                ],
                "alternative_recommendations": ["CPU execution is viable but slower", "GPU recommended for better performance"]
            })
        
        # VRAM uncertainty flags
        if vram_estimate["confidence"] < 0.7:
            assessment["honesty_factors"].append(f"VRAM estimation has moderate uncertainty ({vram_estimate['confidence']:.1%} confidence)")
            assessment["confidence_level"] = "medium"
            assessment["caveats"].append("Monitor actual memory usage during execution")
        
        # GPU availability reality check
        required_vram = vram_estimate["optimized_vram_estimate"]
        available_gpus = [(name, specs['vram']) for name, specs in self.gpu_specs.items() if specs['vram'] >= required_vram]
        
        if not available_gpus:
            assessment["honesty_factors"].append(f"Required {required_vram:.1f}GB exceeds largest available GPU")
            assessment["alternative_recommendations"].append("Consider model optimization techniques or distributed computing")
        
        # Multi-GPU considerations
        if required_vram > 80:  # Larger than single H100
            multi_gpu_options = self._suggest_multi_gpu_options(required_vram)
            assessment["alternative_recommendations"].extend(multi_gpu_options)
        
        return assessment

    def _suggest_multi_gpu_options(self, total_vram_needed):
        """Suggest multi-GPU configurations when single GPU isn't sufficient"""
        import math
        
        suggestions = []
        
        # Find GPUs with NVLink for efficient multi-GPU
        nvlink_gpus = {name: specs for name, specs in self.gpu_specs.items() if specs['nvlink']}
        
        for gpu_name, specs in nvlink_gpus.items():
            num_gpus_needed = math.ceil(total_vram_needed / specs['vram'])
            if num_gpus_needed <= specs['max_reasonable_quantity']:
                suggestions.append(f"{num_gpus_needed}x {gpu_name} with NVLink ({num_gpus_needed * specs['vram']}GB total)")
        
        return suggestions

    def format_recommendations_with_gpu_specs(self, recommendations, assessment, vram_estimate):
        """
        Format output using actual GPU specifications
        """
        output = {
            "workload_assessment": {
                "requires_gpu": assessment["requires_gpu"],
                "confidence": assessment["confidence_level"],
                "vram_estimate": {
                    "base_requirement": vram_estimate["base_vram_estimate"],
                    "optimized_requirement": vram_estimate["optimized_vram_estimate"],
                    "confidence": vram_estimate["confidence"]
                }
            },
            "gpu_recommendations": {},
            "honest_assessment": assessment["honesty_factors"],
            "caveats": assessment["caveats"],
            "gpu_specs_used": True  # Flag that we're using the proper GPU database
        }
        
        # Add each recommendation tier if available
        for tier_name, rec in recommendations.items():
            if rec:
                output["gpu_recommendations"][tier_name] = {
                    "gpu_model": rec["gpu_name"],
                    "vram_gb": rec["vram_gb"],
                    "category": rec["category"],
                    "tier": rec["specs"]["tier"],
                    "form_factor": rec["specs"]["form_factor"],
                    "nvlink_support": rec["specs"]["nvlink"],
                    "tensor_cores": rec["specs"]["tensor_cores"],
                    "estimated_runtime_factor": rec["estimated_runtime_multiplier"],
                    "use_case_description": rec["use_case"],
                    "max_reasonable_quantity": rec["specs"]["max_reasonable_quantity"]
                }
        
        # CPU-only recommendation
        if not assessment["requires_gpu"]:
            output["cpu_only_recommendation"] = {
                "recommended": True,
                "reasoning": "Analysis indicates CPU-optimized workload",
                "performance_impact": "GPU would provide minimal benefit",
                "cost_savings": "Significant cost savings vs GPU deployment"
            }
        
        return output

    def _find_minimum_viable_gpu_strict(self, vram_needed, consumer_viable=True):
        """Find the absolute cheapest GPU that meets VRAM requirements"""
        suitable_gpus = []
        
        for gpu_name, specs in self.gpu_specs.items():
            if specs['vram'] >= vram_needed:
                # Skip consumer GPUs if they're not viable for this workload
                if not consumer_viable and specs['category'] == 'consumer':
                    continue
                    
                # Calculate cost score (lower is cheaper)
                cost_score = self._calculate_price_score(specs)
                suitable_gpus.append((gpu_name, specs, cost_score))
        
        if suitable_gpus:
            # Sort by cost (lowest first)
            return sorted(suitable_gpus, key=lambda x: x[2])[0]
        return None

    def _find_balanced_gpu(self, vram_needed, workload_type, consumer_viable=True):
        """Find GPU with best price/performance ratio"""
        suitable_gpus = []
        target_vram = vram_needed * 1.3  # 30% headroom for balanced mode
        
        for gpu_name, specs in self.gpu_specs.items():
            if specs['vram'] >= target_vram:
                # Skip consumer GPUs if they're not viable for this workload
                if not consumer_viable and specs['category'] == 'consumer':
                    continue
                    
                # Calculate efficiency score (performance per dollar)
                cost_score = self._calculate_price_score(specs)
                performance_score = specs['performance_factor']
                
                # Workload-specific bonuses
                if workload_type == "training" and specs['tensor_cores']:
                    performance_score *= 1.1
                if workload_type == "inference" and specs['category'] == 'consumer':
                    performance_score *= 1.05  # Slight preference for consumer in inference
                
                efficiency_score = performance_score / cost_score
                suitable_gpus.append((gpu_name, specs, efficiency_score))
        
        if suitable_gpus:
            # Sort by efficiency (highest first)
            return sorted(suitable_gpus, key=lambda x: x[2], reverse=True)[0]
        return None

    def _find_performance_gpu(self, vram_needed, workload_type, consumer_viable=True):
        """Find high-performance GPU within reasonable cost bounds"""
        suitable_gpus = []
        target_vram = vram_needed * 1.5  # 50% headroom for performance mode
        
        # Define "reasonable" cost bounds to avoid always recommending B200 SXM
        max_reasonable_cost = 20.0  # Excludes ultra-expensive enterprise GPUs (H100+)
        
        for gpu_name, specs in self.gpu_specs.items():
            if specs['vram'] >= target_vram:
                # Skip consumer GPUs if they're not viable for this workload
                if not consumer_viable and specs['category'] == 'consumer':
                    continue
                    
                cost_score = self._calculate_price_score(specs)
                
                # Skip ultra-expensive GPUs unless workload truly requires them
                if cost_score > max_reasonable_cost:
                    # Only include if workload has specific enterprise requirements AND high VRAM needs
                    if not (workload_type in ["training", "large_models"] and specs['nvlink'] and target_vram > 32):
                        continue
                
                # Performance score with workload bonuses
                performance_score = specs['performance_factor']
                
                if workload_type == "training":
                    if specs['nvlink']:
                        performance_score *= 1.2
                    if specs['tensor_cores']:
                        performance_score *= 1.15
                elif workload_type == "inference":
                    if specs['tier'] in ['flagship', 'high']:
                        performance_score *= 1.1
                
                suitable_gpus.append((gpu_name, specs, performance_score))
        
        if suitable_gpus:
            # Sort by performance (highest first)
            return sorted(suitable_gpus, key=lambda x: x[2], reverse=True)[0]
        return None

    def _ensure_minimum_gpu_consistency(self, analysis: Dict):
        """
        Ensure minimum GPU is consistent with consumer viability.
        If consumer GPUs are not viable, minimum should be enterprise too.
        """
        if not analysis.get('consumer_viable', True):
            # Consumer is not viable, ensure minimum is enterprise
            current_min_gpu = analysis.get('min_gpu_type', '')
            consumer_cards = ['RTX 4060', 'RTX 4070', 'RTX 4080', 'RTX 4090', 'RTX 5080', 'RTX 5090']
            
            if current_min_gpu in consumer_cards:
                # Upgrade minimum to enterprise card based on VRAM needs
                vram_needed = analysis.get('min_vram_gb', 16)
                if vram_needed <= 24:
                    analysis['min_gpu_type'] = 'L4'
                    analysis['min_vram_gb'] = 24
                elif vram_needed <= 48:
                    analysis['min_gpu_type'] = 'L40S'
                    analysis['min_vram_gb'] = 48
                else:
                    analysis['min_gpu_type'] = 'A100 PCIe 80G'
                    analysis['min_vram_gb'] = 80
                
                # Add reasoning for the upgrade
                analysis['reasoning'].append(f"Upgraded minimum GPU from {current_min_gpu} to {analysis['min_gpu_type']} since recommended GPUs are not viable")

    def _ensure_consumer_fields_populated(self, analysis: Dict):
        """
        Ensure consumer/recommended fields are never null in the new 3-tier system.
        With the new approach, we always show 3 tiers when GPUs are beneficial.
        CRITICAL: Respect consumer viability - if consumer not viable, all tiers should be enterprise.
        """
        # If consumer_gpu_type is set but quantity/runtime are null, populate them
        if analysis.get('consumer_gpu_type') and not analysis.get('consumer_quantity'):
            analysis['consumer_quantity'] = analysis.get('min_quantity', 1)
            
        if analysis.get('consumer_gpu_type') and not analysis.get('consumer_runtime_estimate'):
            analysis['consumer_runtime_estimate'] = analysis.get('min_runtime_estimate', '30-60 minutes')
            
        # NEW 3-TIER APPROACH: Always ensure consumer fields are populated if GPU workload detected
        # BUT respect consumer viability
        if not analysis.get('consumer_gpu_type') and analysis.get('min_gpu_type') and analysis.get('min_gpu_type') != 'CPU-only':
            consumer_viable = analysis.get('consumer_viable', True)
            
            if consumer_viable:
                # Consumer viable - use minimum as fallback for recommended tier
                analysis['consumer_gpu_type'] = analysis.get('min_gpu_type')
                analysis['consumer_quantity'] = analysis.get('min_quantity', 1)
                analysis['consumer_vram_gb'] = analysis.get('min_vram_gb', 8)
                analysis['consumer_runtime_estimate'] = analysis.get('min_runtime_estimate', '30-60 minutes')
                
                # Set viability based on GPU category
                consumer_cards = ['RTX 4060', 'RTX 4070', 'RTX 4080', 'RTX 4090', 'RTX 5080', 'RTX 5090']
                if analysis.get('min_gpu_type') in consumer_cards:
                    analysis['consumer_viable'] = True
                    analysis['consumer_limitation'] = None
                else:
                    analysis['consumer_viable'] = False
                    analysis['consumer_limitation'] = f"Workload optimized for enterprise hardware - consumer alternatives may be slower"
            else:
                # Consumer NOT viable - ensure recommended tier is also enterprise
                # Use enterprise/optimal GPU for the recommended tier
                enterprise_gpu = analysis.get('enterprise_gpu_type') or analysis.get('optimal_gpu_type')
                if enterprise_gpu:
                    analysis['consumer_gpu_type'] = enterprise_gpu
                    analysis['consumer_quantity'] = analysis.get('enterprise_quantity') or analysis.get('optimal_quantity', 1)
                    analysis['consumer_vram_gb'] = analysis.get('enterprise_vram_gb') or analysis.get('optimal_vram_gb', 48)
                    analysis['consumer_runtime_estimate'] = analysis.get('enterprise_runtime_estimate') or analysis.get('optimal_runtime_estimate', '15-30 minutes')
                else:
                    # Fallback to minimum if no enterprise GPU set (shouldn't happen but safety)
                    analysis['consumer_gpu_type'] = analysis.get('min_gpu_type')
                    analysis['consumer_quantity'] = analysis.get('min_quantity', 1)
                    analysis['consumer_vram_gb'] = analysis.get('min_vram_gb', 48)
                    analysis['consumer_runtime_estimate'] = analysis.get('min_runtime_estimate', '30-60 minutes')
                
                # Ensure viability is set correctly
                analysis['consumer_viable'] = False

# Environment detection for optimization
def is_production_environment():
    """Detect if running in production environment (Vercel, etc.)"""
    return (
        os.getenv('VERCEL') == '1' or
        os.getenv('VERCEL_ENV') is not None or
        os.getenv('RAILWAY_ENVIRONMENT') is not None or
        os.getenv('RENDER') is not None or
        'vercel' in os.getenv('VERCEL_URL', '').lower() or
        os.getenv('NODE_ENV') == 'production'
    )

def get_environment_config():
    """Get configuration based on environment"""
    # Check for debug flag that forces detailed phases
    force_detailed_phases = os.getenv('DEBUG_DETAILED_PHASES', '').lower() == 'true'
    
    if is_production_environment():
        # Check if we're on Vercel Pro (has more resources)
        # Vercel doesn't always set VERCEL_PLAN, so we use multiple detection methods
        is_vercel_pro = (
            # Explicit Pro features flag
            os.getenv('VERCEL_PRO_FEATURES') == 'true' or
            # Self-review enablement flag
            os.getenv('ENABLE_SELF_REVIEW') == 'true' or
            # Official Vercel plan detection
            os.getenv('VERCEL_PLAN') == 'pro' or 
            os.getenv('VERCEL_PLAN') == 'enterprise' or
            # Detect Pro by checking if we're in production with enhanced features enabled
            (os.getenv('VERCEL_ENV') == 'production' and 
             (os.getenv('ENHANCED_FEATURES') == 'true' or 
              os.getenv('FULL_TRANSPARENCY') == 'true'))
        )
        
        if is_vercel_pro:
            return {
                'llm_timeout': 25,  # Longer timeout for Pro plan
                'progress_batching': False,  # Real-time progress on Pro
                'detailed_phases': True,  # Full transparency on Pro
                'self_review_enabled': True,  # Enable self-review on Pro
                'max_content_length': 10000,  # More content on Pro
                'connection_timeout': 20,
                'smart_self_review': True,  # Intelligent self-review
                'environment_type': 'vercel_pro'
            }
        else:
            return {
                'llm_timeout': 15,  # Shorter timeout for free plan
                'progress_batching': True,  # Batch progress messages
                'detailed_phases': force_detailed_phases,  # Allow debug override
                'self_review_enabled': False,  # Disable self-review on free plan
                'max_content_length': 8000,  # Smaller content limit
                'connection_timeout': 10,
                'smart_self_review': False,
                'environment_type': 'vercel_free'
            }
    else:
        return {
            'llm_timeout': 30,  # Full timeout for local development
            'progress_batching': False,  # Real-time progress locally
            'detailed_phases': True,  # Full transparency locally
            'self_review_enabled': True,  # Full self-review locally
            'max_content_length': 12000,  # Full content locally
            'connection_timeout': 30,
            'smart_self_review': False,  # Standard self-review locally
            'environment_type': 'local_development'
        }