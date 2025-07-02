#!/usr/bin/env python3
"""
Notebook Analyzer for Jupyter Notebooks

This script analyzes Jupyter notebooks to determine minimum GPU requirements
based on the code patterns, libraries used, and computational workloads.
"""

import requests
import json
import re
import ast
import argparse
import os
import sys
import ipaddress
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from urllib.parse import urlparse, quote_plus
import sys

@dataclass
class GPURequirement:
    min_gpu_type: str
    min_quantity: int
    min_vram_gb: int
    optimal_gpu_type: str
    optimal_quantity: int
    optimal_vram_gb: int
    min_runtime_estimate: str
    optimal_runtime_estimate: str
    sxm_required: bool
    sxm_reasoning: List[str]
    arm_compatibility: str
    arm_reasoning: List[str]
    confidence: float
    reasoning: List[str]
    llm_enhanced: bool = False
    llm_reasoning: List[str] = None
    nvidia_compliance_score: float = 0.0
    structure_assessment: Dict[str, str] = None
    content_quality_issues: List[str] = None
    technical_recommendations: List[str] = None
    
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

class LLMAnalyzer:
    def __init__(self, base_url: str, model: str, api_key: str):
        # Remove trailing slashes and /v1 suffix to avoid double /v1 in URLs
        self.base_url = base_url.rstrip('/').rstrip('/v1')
        self.model = model
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def analyze_notebook_context(self, code_cells: List[str], markdown_cells: List[str]) -> Dict:
        """Send notebook to LLM for contextual analysis."""
        try:
            # Combine code and markdown for context
            notebook_content = "\n".join([
                "=== CODE CELLS ===",
                *code_cells[:10],  # Limit to first 10 cells to avoid token limits
                "\n=== MARKDOWN CELLS ===", 
                *markdown_cells[:5]  # Limit markdown cells too
            ])
            
            # Truncate if too long (rough estimate for token limits)
            if len(notebook_content) > 15000:
                notebook_content = notebook_content[:15000] + "\n... [truncated]"
            
            prompt = f"""Analyze this Jupyter notebook for GPU requirements. Focus on:

1. **Workload Type**: Is this inference, fine-tuning, training from scratch, or other?
2. **Model Details**: What models are being used? What are their memory requirements?
3. **Batch Sizes**: What batch sizes are used and are they for experimentation or production?
4. **Memory Optimization**: Are there techniques like gradient checkpointing, LoRA, quantization?
5. **Multi-GPU Patterns**: Is distributed training or model parallelism used?
6. **Dataset Scale**: How large is the dataset being processed?
7. **Training Duration**: Estimated epochs, steps, or convergence time?

Notebook Content:
{notebook_content}

Respond in JSON format with:
{{
    "workload_type": "inference|fine-tuning|training|other",
    "complexity": "simple|moderate|complex|extreme",
    "models_detected": ["model1", "model2"],
    "estimated_vram_gb": number,
    "multi_gpu_required": boolean,
    "memory_optimizations": ["technique1", "technique2"],
    "batch_size_analysis": "description",
    "runtime_factors": ["factor1", "factor2"],
    "confidence": 0.0-1.0,
    "reasoning": ["reason1", "reason2"]
}}"""

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert in GPU computing and machine learning workloads. Analyze notebooks for accurate GPU requirements."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Try to extract JSON from the response
                try:
                    # Look for JSON block in the response
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        return json.loads(json_str)
                except:
                    pass
                
                # Fallback: return basic analysis if JSON parsing fails
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
                print(f"LLM API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return None
    
    def enhance_gpu_recommendation(self, static_analysis: Dict, llm_context: Dict) -> Tuple[Dict, List[str]]:
        """Enhance static analysis with LLM insights."""
        enhanced_analysis = static_analysis.copy()
        llm_reasoning = []
        
        # Enhance VRAM estimate with LLM insights
        if 'estimated_vram_gb' in llm_context and llm_context['estimated_vram_gb']:
            llm_vram = llm_context['estimated_vram_gb']
            static_vram = static_analysis.get('vram_required', 8)
            
            # Use higher estimate but cap at reasonable limits
            enhanced_vram = max(llm_vram, static_vram)
            enhanced_analysis['vram_required'] = min(enhanced_vram, 200)  # Cap at 200GB
            
            if abs(llm_vram - static_vram) > 4:  # Significant difference
                llm_reasoning.append(f"LLM estimated {llm_vram}GB vs static analysis {static_vram}GB")
        
        # Enhance complexity analysis
        if 'complexity' in llm_context:
            enhanced_analysis['workload_complexity'] = llm_context['complexity']
            llm_reasoning.append(f"LLM identified workload complexity: {llm_context['complexity']}")
        
        # Add memory optimization insights
        if 'memory_optimizations' in llm_context and llm_context['memory_optimizations']:
            optimizations = llm_context['memory_optimizations']
            if optimizations:
                # Reduce VRAM estimate if memory optimizations are detected
                reduction_factor = 0.7 if len(optimizations) > 1 else 0.85
                enhanced_analysis['vram_required'] = int(enhanced_analysis['vram_required'] * reduction_factor)
                llm_reasoning.append(f"Memory optimizations detected: {', '.join(optimizations)}")
        
        # Multi-GPU insights
        if 'multi_gpu_required' in llm_context and llm_context['multi_gpu_required']:
            if not static_analysis.get('multi_gpu_needed', 1) > 1:
                enhanced_analysis['multi_gpu_needed'] = 2
                llm_reasoning.append("LLM detected multi-GPU requirements not caught by static analysis")
        
        # Add LLM reasoning
        if 'reasoning' in llm_context:
            llm_reasoning.extend(llm_context['reasoning'])
        
        return enhanced_analysis, llm_reasoning
    
    def evaluate_notebook_compliance(self, code_cells: List[str], markdown_cells: List[str]) -> Dict:
        """Evaluate notebook compliance with NVIDIA best practices using LLM."""
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
            
            prompt = f"""Evaluate this Jupyter notebook against NVIDIA's official notebook guidelines. Analyze:

**STRUCTURE & LAYOUT (25 points):**
- Title quality: Does it follow "doing X with NVIDIA Product" format?
- Introduction completeness: Target audience, overview, time estimate, tools used?
- Header usage: Proper markdown headers for navigation?
- Conclusion: Summary and call-to-action present?

**CONTENT QUALITY (25 points):**
- Documentation ratio: Good balance of markdown vs code?
- Explanations: Are code cells explained?
- Educational value: Clear learning objectives?
- Professional writing: Complete sentences, good grammar?

**TECHNICAL STANDARDS (25 points):**
- Requirements management: requirements.txt mentioned?
- Environment variables: Proper handling vs hardcoded values?
- Reproducibility: Seeds, deterministic operations?
- File structure: Minimal complexity?

**NVIDIA COMPLIANCE (25 points):**
- Product messaging: Proper NVIDIA product references?
- Brand consistency: Professional presentation?
- Developer focus: Clear value proposition?
- Maintenance quality: Well-structured, complete?

Notebook Content:
{structure_sample}

Respond in JSON format:
{{
    "structure_score": 0-25,
    "content_score": 0-25, 
    "technical_score": 0-25,
    "nvidia_score": 0-25,
    "total_score": 0-100,
    "structure_issues": ["issue1", "issue2"],
    "content_issues": ["issue1", "issue2"],
    "technical_issues": ["issue1", "issue2"],
    "nvidia_issues": ["issue1", "issue2"],
    "strengths": ["strength1", "strength2"],
    "recommendations": ["rec1", "rec2"],
    "confidence": 0.0-1.0
}}"""

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert in technical documentation and NVIDIA's content standards. Evaluate notebooks for compliance with best practices."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1200
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
        
        # URL validation configuration
        self.allowed_domains = self._load_allowed_domains()
        self.allowed_domain_patterns = [
            r'gitlab\..*\.nvidia\.com',  # Any NVIDIA GitLab instance
            r'.*\.github\.io',           # GitHub Pages
            r'.*\.gitlab\.io'            # GitLab Pages
        ]  # Suppress output for JSON mode
        
        # Initialize LLM analyzer if environment variables are set
        self.llm_analyzer = None
        openai_base_url = os.getenv('OPENAI_BASE_URL')
        openai_model = os.getenv('OPENAI_MODEL', 'gpt-4')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if openai_base_url and openai_api_key:
            try:
                self.llm_analyzer = LLMAnalyzer(openai_base_url, openai_model, openai_api_key)
                if not self.quiet_mode:
                    print(f"✅ LLM enhancement enabled using {openai_model}")
            except Exception as e:
                if not self.quiet_mode:
                    print(f"⚠️ LLM initialization failed: {e}")
                self.llm_analyzer = None
        else:
            if not self.quiet_mode:
                print("ℹ️ LLM enhancement disabled (set OPENAI_BASE_URL and OPENAI_API_KEY to enable)")
        
        # GitHub authentication setup
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github_headers = {}
        if self.github_token:
            self.github_headers['Authorization'] = f'token {self.github_token}'
            if not self.quiet_mode:
                print("✅ GitHub authentication enabled")
        else:
            if not self.quiet_mode:
                print("ℹ️ GitHub authentication disabled (set GITHUB_TOKEN for private repos)")
        
        # GitLab authentication setup
        self.gitlab_token = os.getenv('GITLAB_TOKEN')
        self.gitlab_headers = {}
        if self.gitlab_token:
            self.gitlab_headers['Authorization'] = f'Bearer {self.gitlab_token}'
            if not self.quiet_mode:
                print("✅ GitLab authentication enabled")
        else:
            if not self.quiet_mode:
                print("ℹ️ GitLab authentication disabled (set GITLAB_TOKEN for private repos)")
        
        # GPU specifications (simplified mapping)
        self.gpu_specs = {
            # Consumer RTX 50 Series
            'RTX 5090': {'vram': 32, 'compute_capability': 8.9, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': False},
            'RTX 5080': {'vram': 16, 'compute_capability': 8.9, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': False},
            
            # Consumer RTX 40 Series
            'RTX 4090': {'vram': 24, 'compute_capability': 8.9, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': False},
            'RTX 4080': {'vram': 16, 'compute_capability': 8.9, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': False},
            
            # Consumer RTX 30 Series
            'RTX 3090': {'vram': 24, 'compute_capability': 8.6, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': True},
            'RTX 3080': {'vram': 10, 'compute_capability': 8.6, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': False},
            
            # Professional RTX 6000 Series
            'RTX 6000 Pro Server': {'vram': 48, 'compute_capability': 8.9, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': True},
            'RTX 6000 Pro Workstation': {'vram': 48, 'compute_capability': 8.9, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': True},
            
            # Data Center GPUs - SXM variants
            'H200 SXM': {'vram': 141, 'compute_capability': 9.0, 'tensor_cores': True, 'form_factor': 'SXM', 'nvlink': True},
            'H200 NVL': {'vram': 141, 'compute_capability': 9.0, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': True},
            'B200 SXM': {'vram': 192, 'compute_capability': 10.0, 'tensor_cores': True, 'form_factor': 'SXM', 'nvlink': True},
            'B200 PCIe': {'vram': 192, 'compute_capability': 10.0, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': True},
            'H100 SXM': {'vram': 80, 'compute_capability': 9.0, 'tensor_cores': True, 'form_factor': 'SXM', 'nvlink': True},
            'H100 PCIe': {'vram': 80, 'compute_capability': 9.0, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': False},
            'H100 NVL': {'vram': 188, 'compute_capability': 9.0, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': True},
            'A100 SXM 80G': {'vram': 80, 'compute_capability': 8.0, 'tensor_cores': True, 'form_factor': 'SXM', 'nvlink': True},
            'A100 PCIe 80G': {'vram': 80, 'compute_capability': 8.0, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': False},
            'A100 SXM 40G': {'vram': 40, 'compute_capability': 8.0, 'tensor_cores': True, 'form_factor': 'SXM', 'nvlink': True},
            'A100 PCIe 40G': {'vram': 40, 'compute_capability': 8.0, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': False},
            'L40S': {'vram': 48, 'compute_capability': 8.9, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': False},
            'L40': {'vram': 48, 'compute_capability': 8.9, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': False},
            'L4': {'vram': 24, 'compute_capability': 8.9, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': False},
            'V100': {'vram': 32, 'compute_capability': 7.0, 'tensor_cores': True, 'form_factor': 'SXM', 'nvlink': True},
            'T4': {'vram': 16, 'compute_capability': 7.5, 'tensor_cores': True, 'form_factor': 'PCIe', 'nvlink': False},
            
            # Legacy
            'GTX 1080 Ti': {'vram': 11, 'compute_capability': 6.1, 'tensor_cores': False, 'form_factor': 'PCIe', 'nvlink': False},
        }
        
        # Library patterns and their typical GPU requirements
        self.library_patterns = {
            'tensorflow': {'base_vram': 2, 'scaling_factor': 1.5},
            'torch': {'base_vram': 2, 'scaling_factor': 1.5},
            'pytorch': {'base_vram': 2, 'scaling_factor': 1.5},
            'transformers': {'base_vram': 4, 'scaling_factor': 2.0},
            'diffusers': {'base_vram': 6, 'scaling_factor': 2.5},
            'stable-diffusion': {'base_vram': 8, 'scaling_factor': 3.0},
            'detectron2': {'base_vram': 4, 'scaling_factor': 1.8},
            'mmdetection': {'base_vram': 4, 'scaling_factor': 1.8},
            'openai': {'base_vram': 1, 'scaling_factor': 1.0},
            'cupy': {'base_vram': 1, 'scaling_factor': 1.2},
            'cudf': {'base_vram': 2, 'scaling_factor': 1.3},
            'rapids': {'base_vram': 4, 'scaling_factor': 1.5},
            'jax': {'base_vram': 2, 'scaling_factor': 1.4},
        }
        
        # Model size patterns (rough estimates)
        self.model_patterns = {
            r'gpt-3\.5|gpt-4': {'vram': 16, 'gpus': 1},
            r'llama-7b|7b': {'vram': 14, 'gpus': 1},
            r'llama-13b|13b': {'vram': 26, 'gpus': 2},
            r'llama-30b|30b': {'vram': 60, 'gpus': 3},
            r'llama-65b|65b': {'vram': 130, 'gpus': 6},
            r'bert-base': {'vram': 2, 'gpus': 1},
            r'bert-large': {'vram': 4, 'gpus': 1},
            r'resnet50': {'vram': 2, 'gpus': 1},
            r'resnet152': {'vram': 4, 'gpus': 1},
            r'vgg16': {'vram': 2, 'gpus': 1},
            r'efficientnet': {'vram': 3, 'gpus': 1},
        }
        
        # Batch size patterns
        self.batch_size_pattern = re.compile(r'batch_size\s*=\s*(\d+)', re.IGNORECASE)
        
        # Image/data size patterns
        self.image_size_pattern = re.compile(r'(?:image_size|img_size|resolution)\s*=\s*(?:\[)?(\d+)', re.IGNORECASE)
        
        # Training patterns
        self.training_patterns = [
            r'\.train\(\)',
            r'\.fit\(',
            r'optimizer\.',
            r'loss\.backward\(\)',
            r'gradient',
            r'epochs\s*=',
            r'learning_rate',
        ]
        
        # SXM requirement patterns
        self.sxm_patterns = [
            r'nvlink',
            r'nccl',
            r'multi_gpu.*=.*True',
            r'DataParallel',
            r'DistributedDataParallel',
            r'horovod',
            r'distributed.*training',
            r'model.*parallel',
            r'pipeline.*parallel',
            r'tensor.*parallel',
            r'all_reduce',
            r'all_gather',
        ]
        
        # ARM/Grace incompatible patterns
        self.arm_incompatible_patterns = [
            r'cudnn',  # Some CuDNN versions have limited ARM support
            r'tensorrt',  # TensorRT has limited ARM support
            r'triton',  # Triton may have x86 dependencies
            r'apex',  # NVIDIA Apex may have x86 dependencies
            r'flash.attention',  # Flash Attention may have x86 optimizations
            r'xformers',  # xformers may have x86 optimizations
        ]
        
        # GPU performance tiers for optimal recommendations
        self.performance_tiers = {
            'entry': ['GTX 1080 Ti', 'T4', 'RTX 3080'],
            'mid': ['RTX 4080', 'RTX 5080', 'L4', 'RTX 3090'],
            'high': ['RTX 4090', 'RTX 5090', 'L40', 'L40S', 'RTX 6000 Pro Workstation'],
            'enterprise': ['A100 PCIe 40G', 'A100 SXM 40G', 'RTX 6000 Pro Server'],
            'flagship': ['A100 PCIe 80G', 'A100 SXM 80G', 'H100 PCIe', 'H100 SXM'],
            'cutting_edge': ['H100 NVL', 'H200 SXM', 'H200 NVL', 'B200 SXM', 'B200 PCIe']
        }
        
        # Workload complexity indicators
        self.complexity_indicators = {
            'simple': ['inference', 'eval', 'predict', 'generate'],
            'moderate': ['fine.?tun', 'transfer.?learn', 'small.*train'],
            'complex': ['train.*from.*scratch', 'distributed', 'multi.*gpu', 'large.*model'],
            'extreme': ['foundation.*model', 'pretrain', 'billion.*param', 'terabyte']
        }
        
        # ARM/Grace compatible frameworks
        self.arm_compatible_frameworks = [
            'tensorflow',
            'pytorch',
            'torch',
            'jax',
            'cupy',
            'rapids',
            'transformers',
            'diffusers',
        ]

    def sanitize_url_args(self, args: List[str]) -> str:
        """
        Automatically handle shell quoting issues by reconstructing URLs from command line arguments.
        This handles cases where URLs with query parameters get split by the shell.
        """
        if not args:
            return ""
        
        # If we have multiple arguments that look like URL fragments, join them
        url_parts = []
        for arg in args:
            url_parts.append(arg)
        
        # Join all parts and look for URL patterns
        combined = ' '.join(url_parts)
        
        # Look for URL patterns that got split
        url_patterns = [
            r'(https?://[^\s]+)',  # Basic URL
            r'(https?://[^\s]+\?[^\s]*)',  # URL with query params
        ]
        
        for pattern in url_patterns:
            match = re.search(pattern, combined)
            if match:
                return match.group(1)
        
        # If no URL pattern found, return the first argument (could be a file path)
        return args[0]
    
    def _load_allowed_domains(self) -> List[str]:
        """Load allowed domains from environment variable with secure defaults."""
        default_domains = [
            'github.com',
            'gitlab.com', 
            'raw.githubusercontent.com',
            'gitlab-master.nvidia.com'  # NVIDIA internal GitLab
        ]
        
        # Allow custom domains via environment variable
        custom_domains = os.getenv('ALLOWED_DOMAINS', '')
        if custom_domains:
            custom_list = [domain.strip() for domain in custom_domains.split(',') if domain.strip()]
            return custom_list
        
        return default_domains
    
    def _is_private_ip(self, ip_str: str) -> bool:
        """Check if IP address is in private range."""
        try:
            ip = ipaddress.ip_address(ip_str)
            return ip.is_private or ip.is_loopback or ip.is_link_local
        except ValueError:
            return False
    
    def _validate_url_security(self, url: str) -> Tuple[bool, str]:
        """
        Validate URL for security (SSRF prevention).
        Returns (is_valid, error_message)
        """
        try:
            parsed = urlparse(url)
            
            # Block non-HTTP protocols
            if parsed.scheme not in ['http', 'https']:
                return False, f"Unsupported protocol: {parsed.scheme}"
            
            # Require HTTPS for external domains (allow HTTP for internal)
            if parsed.scheme == 'http' and not any(pattern in parsed.netloc for pattern in ['nvidia.com', 'localhost', '127.0.0.1']):
                return False, "HTTP not allowed for external domains, use HTTPS"
            
            # Block localhost variations
            localhost_patterns = ['localhost', '127.', '0.0.0.0', '::1']
            if any(pattern in parsed.netloc.lower() for pattern in localhost_patterns):
                return False, "Localhost access not allowed"
            
            # Block cloud metadata endpoints
            metadata_endpoints = ['169.254.169.254', 'metadata.google.internal', 'metadata']
            if any(endpoint in parsed.netloc.lower() for endpoint in metadata_endpoints):
                return False, "Cloud metadata endpoint access blocked"
            
            # Resolve hostname to IP and check for private ranges
            import socket
            try:
                ip = socket.gethostbyname(parsed.netloc)
                if self._is_private_ip(ip):
                    # Allow internal NVIDIA domains
                    if not any(pattern in parsed.netloc.lower() for pattern in ['nvidia.com']):
                        return False, f"Private IP address access blocked: {ip}"
            except socket.gaierror:
                # If we can't resolve, that's suspicious
                return False, f"Cannot resolve hostname: {parsed.netloc}"
            
            # Check against allowed domains
            hostname = parsed.netloc.lower()
            
            # Direct domain match
            if hostname in [domain.lower() for domain in self.allowed_domains]:
                return True, ""
            
            # Pattern matching for dynamic domains
            for pattern in self.allowed_domain_patterns:
                if re.match(pattern, hostname, re.IGNORECASE):
                    return True, ""
            
            return False, f"Domain not in allowed list: {hostname}"
            
        except Exception as e:
            return False, f"URL validation error: {str(e)}"
    
    def _validate_file_extension(self, url: str) -> Tuple[bool, str]:
        """Validate that URL points to an allowed file type."""
        allowed_extensions = ['.ipynb', '.py']
        
        # Extract path from URL
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Check file extension
        if any(path.endswith(ext) for ext in allowed_extensions):
            return True, ""
        
        # Special handling for GitHub/GitLab URLs that may have query parameters
        if any(service in parsed.netloc for service in ['github.com', 'gitlab']):
            # For GitHub/GitLab blob URLs, the actual file extension might be in the path
            path_parts = path.split('/')
            if path_parts:
                filename = path_parts[-1]
                if any(filename.endswith(ext) for ext in allowed_extensions):
                    return True, ""
        
        return False, f"File extension not allowed. Must be one of: {', '.join(allowed_extensions)}"

    def is_marimo_notebook(self, file_path: str) -> bool:
        """Check if a Python file is a marimo notebook."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for marimo app patterns
            marimo_patterns = [
                r'import\s+marimo',
                r'@app\.cell',
                r'app\s*=\s*marimo\.App',
                r'marimo\.App\s*\(',
            ]
            
            for pattern in marimo_patterns:
                if re.search(pattern, content):
                    return True
            
            return False
        except Exception:
            return False

    def parse_marimo_notebook(self, file_path: str) -> Optional[Dict]:
        """Parse a marimo notebook (.py file) into a notebook-like structure."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not self.quiet_mode:
                print(f"📁 Loading marimo notebook: {file_path}")
            
            # Parse the Python file using AST
            tree = ast.parse(content)
            
            cells = []
            markdown_cells = []
            current_cell_code = []
            
            # Extract cells from the AST
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if this function has @app.cell decorator
                    has_app_cell_decorator = False
                    for decorator in node.decorator_list:
                        if (isinstance(decorator, ast.Attribute) and 
                            isinstance(decorator.value, ast.Name) and
                            decorator.value.id == 'app' and 
                            decorator.attr == 'cell'):
                            has_app_cell_decorator = True
                            break
                        elif (isinstance(decorator, ast.Call) and
                              isinstance(decorator.func, ast.Attribute) and
                              isinstance(decorator.func.value, ast.Name) and
                              decorator.func.value.id == 'app' and 
                              decorator.func.attr == 'cell'):
                            has_app_cell_decorator = True
                            break
                    
                    if has_app_cell_decorator:
                        # Extract the function body as a cell
                        cell_lines = []
                        for stmt in node.body:
                            cell_lines.append(ast.unparse(stmt))
                        
                        cell_content = '\n'.join(cell_lines)
                        cells.append({
                            'cell_type': 'code',
                            'source': cell_content,
                            'metadata': {}
                        })
                
                # Look for module-level docstrings and comments as markdown
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    if isinstance(node.value.value, str) and len(node.value.value) > 50:
                        # Treat long string constants as markdown
                        markdown_cells.append(node.value.value)
                        cells.append({
                            'cell_type': 'markdown',
                            'source': node.value.value,
                            'metadata': {}
                        })
            
            # Create a notebook-like structure
            notebook_data = {
                'cells': cells,
                'metadata': {
                    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                    'language_info': {'name': 'python', 'version': '3.8'},
                    'notebook_type': 'marimo'
                },
                'nbformat': 4,
                'nbformat_minor': 4
            }
            
            if not self.quiet_mode:
                print(f"✅ Successfully loaded marimo notebook ({len(cells)} cells found)")
            
            return notebook_data
            
        except Exception as e:
            if not self.quiet_mode:
                print(f"❌ Error parsing marimo notebook: {e}")
            return None

    def load_local_notebook(self, file_path: str) -> Optional[Dict]:
        """Load notebook from local file system (supports both Jupyter and marimo)."""
        try:
            path = Path(file_path)
            if not path.exists():
                if not self.quiet_mode:
                    print(f"❌ File not found: {file_path}")
                return None
            
            file_extension = path.suffix.lower()
            
            # Handle Jupyter notebooks (.ipynb)
            if file_extension == '.ipynb':
                if not self.quiet_mode:
                    print(f"📁 Loading Jupyter notebook: {file_path}")
                with open(path, 'r', encoding='utf-8') as f:
                    notebook_data = json.load(f)
                
                if not self.quiet_mode:
                    print(f"✅ Successfully loaded Jupyter notebook")
                return notebook_data
            
            # Handle Python files (check for marimo)
            elif file_extension == '.py':
                if self.is_marimo_notebook(file_path):
                    return self.parse_marimo_notebook(file_path)
                else:
                    if not self.quiet_mode:
                        print(f"❌ Python file is not a marimo notebook: {file_path}")
                        print(f"💡 Marimo notebooks should contain @app.cell decorators")
                    return None
            
            else:
                if not self.quiet_mode:
                    print(f"❌ Unsupported file format: {file_extension}")
                    print(f"💡 Supported formats: .ipynb (Jupyter), .py (marimo)")
                return None
            
        except json.JSONDecodeError as e:
            if not self.quiet_mode:
                print(f"❌ Invalid JSON in notebook file: {e}")
            return None
        except Exception as e:
            if not self.quiet_mode:
                print(f"❌ Error reading local file: {e}")
            return None

    def fetch_notebook(self, url_or_path: str) -> Optional[Dict]:
        """Fetch notebook content from URL or load from local file."""
        try:
            # Check if it's a local file path
            if not url_or_path.startswith(('http://', 'https://')):
                return self.load_local_notebook(url_or_path)
            
            # Handle URLs - validate first for security
            url = url_or_path
            
            # Validate URL security (SSRF prevention)
            is_valid_url, url_error = self._validate_url_security(url)
            if not is_valid_url:
                if not self.quiet_mode:
                    print(f"❌ URL validation failed: {url_error}")
                return None
            
            # Validate file extension
            is_valid_ext, ext_error = self._validate_file_extension(url)
            if not is_valid_ext:
                if not self.quiet_mode:
                    print(f"❌ File extension validation failed: {ext_error}")
                return None
            
            if not self.quiet_mode:
                print(f"✅ URL validation passed: {url}")
            
            # GitHub URL conversion with authentication support
            if 'github.com' in url and '/blob/' in url:
                if not self.quiet_mode:
                    print(f"🔄 Converting GitHub URL...")
                
                # Simple conversion: github.com/owner/repo/blob/branch/path → raw.githubusercontent.com/owner/repo/branch/path
                raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
                
                if not self.quiet_mode:
                    print(f"🔗 Trying: {raw_url}")
                
                # Use GitHub token if available
                headers = self.github_headers.copy()
                response = requests.get(raw_url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    if not self.quiet_mode:
                        print(f"✅ Successfully fetched notebook")
                    return response.json()
                elif response.status_code == 404:
                    if not self.quiet_mode:
                        print(f"❌ Not found (404) - Repository may be private or file doesn't exist")
                        if not self.github_token:
                            print(f"💡 For private repositories, set GITHUB_TOKEN environment variable")
                        print(f"🔧 Alternative: Get the raw URL with auth token from GitHub and use it directly")
                elif response.status_code == 403:
                    if not self.quiet_mode:
                        print(f"❌ Forbidden (403) - Authentication required or rate limited")
                        print(f"💡 Set GITHUB_TOKEN environment variable for authentication")
                else:
                    if not self.quiet_mode:
                        print(f"❌ HTTP Error {response.status_code}")
                
                return None
            
            # GitLab URL conversion with authentication support
            elif 'gitlab.' in url and '/-/blob/' in url:
                if not self.quiet_mode:
                    print(f"🔄 Converting GitLab URL...")
                
                # GitLab conversion: gitlab.com/owner/repo/-/blob/branch/path → gitlab.com/owner/repo/-/raw/branch/path
                raw_url = url.replace('/-/blob/', '/-/raw/')
                
                if not self.quiet_mode:
                    print(f"🔗 Trying: {raw_url}")
                
                # Use GitLab token if available
                headers = self.gitlab_headers.copy()
                response = requests.get(raw_url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    if not self.quiet_mode:
                        print(f"✅ Successfully fetched notebook")
                    return response.json()
                elif response.status_code == 404:
                    if not self.quiet_mode:
                        print(f"❌ Not found (404) - Repository may be private or file doesn't exist")
                        if not self.gitlab_token:
                            print(f"💡 For private repositories, set GITLAB_TOKEN environment variable")
                        print(f"🔧 Alternative: Get the raw URL with auth token from GitLab and use it directly")
                elif response.status_code == 403:
                    if not self.quiet_mode:
                        print(f"❌ Forbidden (403) - Authentication required or rate limited")
                        print(f"💡 Set GITLAB_TOKEN environment variable for authentication")
                else:
                    if not self.quiet_mode:
                        print(f"❌ HTTP Error {response.status_code}")
                
                return None
            
            # For raw URLs or other direct URLs
            if not self.quiet_mode:
                print(f"🔗 Fetching: {url}")
            
            # Use appropriate token for raw URLs
            headers = {}
            if 'raw.githubusercontent.com' in url and self.github_token:
                headers = self.github_headers
            elif 'gitlab.' in url and '/-/raw/' in url and self.gitlab_token:
                headers = self.gitlab_headers
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if not self.quiet_mode:
                if e.response.status_code == 404:
                    print(f"❌ Notebook not found (404)")
                    print(f"💡 Check that the URL is correct and the repository/file exists")
                elif e.response.status_code == 403:
                    print(f"❌ Access forbidden (403)")
                    print(f"💡 Repository may be private - set GITHUB_TOKEN or GITLAB_TOKEN environment variable")
                else:
                    print(f"❌ HTTP Error {e.response.status_code}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            if not self.quiet_mode:
                print(f"❌ Network error: {e}")
            return None
        except json.JSONDecodeError as e:
            if not self.quiet_mode:
                print(f"❌ Invalid JSON response - file may not be a valid notebook: {e}")
            return None
        except Exception as e:
            if not self.quiet_mode:
                print(f"❌ Unexpected error: {e}")
            return None

    def extract_code_cells(self, notebook: Dict) -> Tuple[List[str], List[str]]:
        """Extract code and markdown from notebook cells (supports both Jupyter and marimo)."""
        code_cells = []
        markdown_cells = []
        
        if 'cells' in notebook:
            for cell in notebook['cells']:
                source = cell.get('source', [])
                if isinstance(source, list):
                    content = ''.join(source)
                else:
                    content = source
                
                if cell.get('cell_type') == 'code':
                    code_cells.append(content)
                elif cell.get('cell_type') == 'markdown':
                    markdown_cells.append(content)
        
        return code_cells, markdown_cells

    def analyze_imports(self, code: str) -> Dict[str, int]:
        """Analyze import statements to identify GPU-intensive libraries."""
        imports = {}
        
        # Parse imports
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports[alias.name.lower()] = imports.get(alias.name.lower(), 0) + 1
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports[node.module.lower()] = imports.get(node.module.lower(), 0) + 1
        except:
            # Fallback to regex if AST parsing fails
            import_patterns = [
                r'import\s+(\w+)',
                r'from\s+(\w+)\s+import',
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, code, re.IGNORECASE)
                for match in matches:
                    imports[match.lower()] = imports.get(match.lower(), 0) + 1
        
        return imports

    def analyze_model_usage(self, code: str) -> Dict[str, Dict]:
        """Analyze model usage patterns."""
        model_usage = {}
        
        for pattern, requirements in self.model_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                model_usage[pattern] = requirements
        
        return model_usage

    def analyze_batch_size(self, code: str) -> Optional[int]:
        """Extract batch size from code."""
        matches = self.batch_size_pattern.findall(code)
        if matches:
            return max(int(match) for match in matches)
        return None

    def analyze_image_size(self, code: str) -> Optional[int]:
        """Extract image/data size from code."""
        matches = self.image_size_pattern.findall(code)
        if matches:
            return max(int(match) for match in matches)
        return None

    def is_training_code(self, code: str) -> bool:
        """Check if code contains training patterns."""
        for pattern in self.training_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        return False

    def analyze_workload_complexity(self, code: str) -> str:
        """Analyze the complexity of the workload."""
        complexity_scores = {
            'simple': 0,
            'moderate': 0,
            'complex': 0,
            'extreme': 0
        }
        
        for complexity, patterns in self.complexity_indicators.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, code, re.IGNORECASE))
                complexity_scores[complexity] += matches
        
        # Determine complexity based on scores
        max_complexity = max(complexity_scores, key=complexity_scores.get)
        if complexity_scores[max_complexity] == 0:
            return 'simple'  # Default to simple if no patterns found
        
        return max_complexity

    def estimate_runtime(self, gpu_name: str, workload_complexity: str, vram_usage_ratio: float, is_training: bool) -> str:
        """Estimate runtime based on GPU performance and workload."""
        gpu_specs = self.gpu_specs.get(gpu_name, {})
        
        # Base runtime multipliers based on compute capability
        compute_capability = gpu_specs.get('compute_capability', 6.0)
        has_tensor_cores = gpu_specs.get('tensor_cores', False)
        
        # Performance scaling factors
        base_performance = {
            'simple': 1.0,
            'moderate': 3.0,
            'complex': 10.0,
            'extreme': 50.0
        }
        
        # GPU performance multipliers (relative to RTX 4090 = 1.0)
        gpu_performance_multipliers = {
            'B200 SXM': 0.2, 'B200 PCIe': 0.25,
            'H200 SXM': 0.3, 'H200 NVL': 0.35,
            'H100 SXM': 0.4, 'H100 PCIe': 0.5, 'H100 NVL': 0.35,
            'A100 SXM 80G': 0.6, 'A100 PCIe 80G': 0.7,
            'A100 SXM 40G': 0.6, 'A100 PCIe 40G': 0.7,
            'RTX 5090': 0.8, 'RTX 4090': 1.0,
            'RTX 5080': 1.3, 'RTX 4080': 1.5,
            'L40S': 1.1, 'L40': 1.2, 'L4': 2.0,
            'RTX 6000 Pro Server': 1.1, 'RTX 6000 Pro Workstation': 1.1,
            'RTX 3090': 1.4, 'RTX 3080': 2.0,
            'V100': 2.5, 'T4': 4.0, 'GTX 1080 Ti': 6.0
        }
        
        # Get base runtime and performance multiplier
        base_runtime = base_performance.get(workload_complexity, 5.0)
        perf_multiplier = gpu_performance_multipliers.get(gpu_name, 3.0)
        
        # Training overhead
        if is_training:
            base_runtime *= 5.0  # Training takes much longer
        
        # Tensor core acceleration
        if has_tensor_cores and workload_complexity in ['moderate', 'complex', 'extreme']:
            perf_multiplier *= 0.6  # 40% speedup with tensor cores
        
        # VRAM pressure penalty
        if vram_usage_ratio > 0.8:
            perf_multiplier *= 1.5  # Slowdown when VRAM is nearly full
        elif vram_usage_ratio > 0.9:
            perf_multiplier *= 2.0  # Significant slowdown
        
        # Calculate estimated runtime
        estimated_minutes = base_runtime * perf_multiplier
        
        # Format runtime estimate
        if estimated_minutes < 1:
            return f"{estimated_minutes * 60:.0f} seconds"
        elif estimated_minutes < 60:
            return f"{estimated_minutes:.1f} minutes"
        elif estimated_minutes < 1440:  # Less than 24 hours
            hours = estimated_minutes / 60
            return f"{hours:.1f} hours"
        else:
            days = estimated_minutes / 1440
            return f"{days:.1f} days"

    def evaluate_notebook_structure(self, code_cells: List[str], markdown_cells: List[str]) -> Dict[str, str]:
        """Evaluate notebook structure against NVIDIA guidelines."""
        assessment = {}
        
        # Check title quality (first markdown cell)
        if markdown_cells:
            first_cell = markdown_cells[0].lower()
            if 'with nvidia' in first_cell or 'using nvidia' in first_cell:
                assessment['title'] = "✅ Good title format"
            elif 'nvidia' in first_cell:
                assessment['title'] = "⚠️ Title could follow 'doing X with NVIDIA Product' format"
            else:
                assessment['title'] = "❌ Title should mention NVIDIA products"
        else:
            assessment['title'] = "❌ No title found"
        
        # Check introduction completeness
        intro_content = ' '.join(markdown_cells[:3]).lower() if len(markdown_cells) >= 3 else ''
        intro_score = 0
        intro_elements = []
        
        if any(word in intro_content for word in ['audience', 'for', 'target', 'developer']):
            intro_score += 1
            intro_elements.append("target audience")
        if any(word in intro_content for word in ['overview', 'learn', 'will', 'tutorial']):
            intro_score += 1
            intro_elements.append("overview")
        if any(word in intro_content for word in ['minutes', 'hours', 'time', 'complete']):
            intro_score += 1
            intro_elements.append("time estimate")
        if 'nvidia' in intro_content:
            intro_score += 1
            intro_elements.append("NVIDIA tools")
        
        if intro_score >= 3:
            assessment['introduction'] = f"✅ Good introduction ({', '.join(intro_elements)})"
        elif intro_score >= 2:
            assessment['introduction'] = f"⚠️ Introduction present but could be enhanced"
        else:
            assessment['introduction'] = "❌ Missing key introduction elements"
        
        # Check header usage
        header_count = sum(1 for cell in markdown_cells if re.search(r'^#+\s', cell, re.MULTILINE))
        if header_count >= 3:
            assessment['navigation'] = "✅ Good use of headers for navigation"
        elif header_count >= 1:
            assessment['navigation'] = "⚠️ Some headers present, could use more"
        else:
            assessment['navigation'] = "❌ Missing headers for navigation"
        
        # Check conclusion
        if markdown_cells:
            last_cells = ' '.join(markdown_cells[-2:]).lower()
            if any(word in last_cells for word in ['summary', 'conclusion', 'learned', 'next steps']):
                assessment['conclusion'] = "✅ Has summary/conclusion"
            else:
                assessment['conclusion'] = "⚠️ Could benefit from a clear conclusion"
        else:
            assessment['conclusion'] = "❌ No conclusion found"
        
        return assessment

    def assess_content_quality(self, code_cells: List[str], markdown_cells: List[str]) -> List[str]:
        """Assess content quality issues."""
        issues = []
        
        # Check documentation ratio
        total_cells = len(code_cells) + len(markdown_cells)
        if total_cells > 0:
            markdown_ratio = len(markdown_cells) / total_cells
            if markdown_ratio < 0.3:
                issues.append("Low documentation ratio - consider adding more explanatory text")
            elif markdown_ratio > 0.7:
                issues.append("High documentation ratio - ensure code examples are sufficient")
        
        # Check for code explanation patterns
        explained_code_blocks = 0
        for i, code_cell in enumerate(code_cells):
            # Check if code cell is preceded or followed by markdown
            has_explanation = False
            if i > 0 and i-1 < len(markdown_cells):
                has_explanation = True
            if i < len(markdown_cells):
                has_explanation = True
            if has_explanation:
                explained_code_blocks += 1
        
        if code_cells and explained_code_blocks / len(code_cells) < 0.5:
            issues.append("Many code cells lack explanatory text")
        
        # Check for links (basic pattern)
        all_markdown = ' '.join(markdown_cells)
        link_count = len(re.findall(r'\[.*?\]\(.*?\)', all_markdown))
        if link_count == 0:
            issues.append("No links to external resources found")
        elif link_count < 3:
            issues.append("Consider adding more links to relevant documentation")
        
        # Check sentence structure
        if markdown_cells:
            incomplete_sentences = 0
            for cell in markdown_cells:
                sentences = re.split(r'[.!?]+', cell)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 10 and not sentence[0].isupper():
                        incomplete_sentences += 1
            
            if incomplete_sentences > len(markdown_cells) * 0.2:
                issues.append("Some text may not follow proper sentence structure")
        
        return issues

    def check_technical_standards(self, code_cells: List[str]) -> List[str]:
        """Check technical standards compliance."""
        recommendations = []
        all_code = '\n'.join(code_cells)
        
        # Check for requirements.txt
        if 'requirements.txt' not in all_code.lower():
            recommendations.append("Add requirements.txt file installation")
        
        # Check for version pinning
        pip_install_lines = re.findall(r'pip install.*', all_code, re.IGNORECASE)
        unpinned_packages = []
        for line in pip_install_lines:
            if '==' not in line and 'requirements.txt' not in line:
                packages = re.findall(r'\b([a-zA-Z0-9_-]+)\b', line.replace('pip install', ''))
                unpinned_packages.extend(packages[:3])  # Limit to avoid spam
        
        if unpinned_packages:
            recommendations.append(f"Pin package versions (e.g., {unpinned_packages[0]}==1.2.3)")
        
        # Check for environment variables
        env_patterns = [
            r'os\.environ\[',
            r'getenv\(',
            r'API_KEY\s*=\s*["\']',
            r'TOKEN\s*=\s*["\']'
        ]
        hardcoded_vars = []
        for pattern in env_patterns:
            if re.search(pattern, all_code):
                if '=' in pattern and '"' in pattern:
                    hardcoded_vars.append("hardcoded credentials")
                break
        
        if hardcoded_vars:
            recommendations.append("Avoid hardcoding API keys - use environment variables")
        elif not re.search(r'os\.environ|getenv', all_code):
            recommendations.append("Consider using environment variables for configuration")
        
        # Check for reproducibility
        has_seed = bool(re.search(r'seed\s*=|random_state\s*=|np\.random\.seed|torch\.manual_seed', all_code, re.IGNORECASE))
        if not has_seed and any(lib in all_code.lower() for lib in ['random', 'numpy', 'torch', 'tensorflow']):
            recommendations.append("Set seeds for reproducibility")
        
        # Check file complexity
        file_refs = len(re.findall(r'\.py["\']|\.json["\']|\.yaml["\']|\.csv["\']', all_code))
        if file_refs > 5:
            recommendations.append("Consider simplifying file dependencies")
        
        return recommendations

    def calculate_nvidia_compliance_score(self, structure_assessment: Dict, content_issues: List, technical_recommendations: List, llm_evaluation: Dict = None) -> float:
        """Calculate overall NVIDIA compliance score."""
        if llm_evaluation:
            return min(llm_evaluation.get('total_score', 70), 100)
        
        # Fallback static scoring
        structure_score = 0
        for status in structure_assessment.values():
            if status.startswith('✅'):
                structure_score += 6.25  # 25 points / 4 categories
            elif status.startswith('⚠️'):
                structure_score += 3.75
        
        content_score = max(0, 25 - len(content_issues) * 5)  # Deduct 5 points per issue
        technical_score = max(0, 25 - len(technical_recommendations) * 4)  # Deduct 4 points per recommendation
        nvidia_score = 20  # Default score for brand compliance (harder to assess statically)
        
        return min(structure_score + content_score + technical_score + nvidia_score, 100)

    def analyze_sxm_requirements(self, code: str) -> Tuple[bool, List[str]]:
        """Analyze if SXM form factor GPUs are required."""
        sxm_required = False
        reasons = []
        
        for pattern in self.sxm_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                sxm_required = True
                reasons.append(f"SXM pattern detected: {pattern}")
        
        # Check for high-bandwidth memory requirements
        if re.search(r'batch_size\s*=\s*([1-9]\d{2,})', code):  # batch size >= 100
            batch_matches = re.findall(r'batch_size\s*=\s*(\d+)', code)
            if batch_matches:
                max_batch = max(int(b) for b in batch_matches)
                if max_batch >= 512:
                    sxm_required = True
                    reasons.append(f"Large batch size ({max_batch}) benefits from NVLink")
        
        # Check for multi-GPU model patterns
        multi_gpu_patterns = [
            r'torch\.nn\.parallel',
            r'tf\.distribute',
            r'jax\.pmap',
            r'device_count\(\)',
            r'gpu.*count',
        ]
        
        for pattern in multi_gpu_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                sxm_required = True
                reasons.append(f"Multi-GPU pattern: {pattern}")
                break
        
        return sxm_required, reasons

    def analyze_arm_compatibility(self, code: str) -> Tuple[str, List[str]]:
        """Analyze ARM/Grace compatibility."""
        compatibility = "Compatible"
        issues = []
        warnings = []
        
        # Check for incompatible patterns
        for pattern in self.arm_incompatible_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                compatibility = "Incompatible"
                issues.append(f"ARM incompatible library/pattern: {pattern}")
        
        # Check for x86-specific CUDA operations
        x86_cuda_patterns = [
            r'__device__.*asm',
            r'inline.*assembly',
            r'ptx.*inline',
            r'cub::', # CUB library may have x86 optimizations
        ]
        
        for pattern in x86_cuda_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                compatibility = "Likely Incompatible"
                issues.append(f"x86-specific CUDA code: {pattern}")
        
        # Check for potentially problematic patterns
        warning_patterns = [
            (r'conda.*install.*(?!.*linux-aarch64)', "Conda packages may not have ARM builds"),
            (r'pip.*install.*(?:tensorflow-gpu|torch.*cuda)', "May need ARM-specific package versions"),
            (r'docker.*pull.*(?!.*arm)', "Docker images may not have ARM variants"),
            (r'\.so\b|\.dll\b', "Binary dependencies may not support ARM"),
        ]
        
        for pattern, warning in warning_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                if compatibility == "Compatible":
                    compatibility = "Likely Compatible"
                warnings.append(warning)
        
        # Check for ARM-friendly frameworks
        arm_friendly_count = 0
        for framework in self.arm_compatible_frameworks:
            if framework in code.lower():
                arm_friendly_count += 1
        
        if arm_friendly_count > 0 and compatibility == "Compatible":
            warnings.append(f"Uses {arm_friendly_count} ARM-compatible frameworks")
        
        all_reasons = issues + warnings
        return compatibility, all_reasons

    def estimate_vram_requirements(self, analysis: Dict) -> Tuple[int, List[str]]:
        """Estimate VRAM requirements based on analysis."""
        base_vram = 2  # Minimum for basic GPU operations
        reasoning = []
        
        # Library-based estimation
        for lib, count in analysis.get('imports', {}).items():
            if lib in self.library_patterns:
                lib_req = self.library_patterns[lib]
                additional_vram = lib_req['base_vram'] * lib_req['scaling_factor']
                base_vram += additional_vram
                reasoning.append(f"Library {lib}: +{additional_vram:.1f}GB")
        
        # Model-based estimation
        for model, requirements in analysis.get('models', {}).items():
            model_vram = requirements['vram']
            if model_vram > base_vram:
                base_vram = model_vram
                reasoning.append(f"Model {model}: {model_vram}GB required")
        
        # Batch size scaling
        batch_size = analysis.get('batch_size')
        if batch_size and batch_size > 32:
            scaling = (batch_size / 32) * 1.5
            base_vram *= scaling
            reasoning.append(f"Large batch size ({batch_size}): {scaling:.1f}x scaling")
        
        # Image size scaling
        image_size = analysis.get('image_size')
        if image_size and image_size > 512:
            scaling = (image_size / 512) ** 2
            base_vram *= scaling
            reasoning.append(f"Large image size ({image_size}): {scaling:.1f}x scaling")
        
        # Training overhead
        if analysis.get('is_training'):
            base_vram *= 2.5  # Training typically requires 2-3x more memory
            reasoning.append("Training detected: 2.5x memory overhead")
        
        return int(base_vram), reasoning

    def recommend_gpu(self, vram_required: int, multi_gpu_needed: int = 1, sxm_required: bool = False, workload_complexity: str = 'simple') -> Tuple[Tuple[str, int], Tuple[str, int]]:
        """Recommend both minimum and optimal GPU configurations."""
        # Filter GPUs based on SXM requirement and VRAM needs
        suitable_gpus = []
        
        for gpu_name, specs in self.gpu_specs.items():
            # Check VRAM requirement
            if specs['vram'] < vram_required:
                continue
                
            # Check SXM requirement
            if sxm_required:
                # For SXM requirement, prefer SXM form factor, but allow PCIe with NVLink
                if specs['form_factor'] == 'SXM' or (specs['form_factor'] == 'PCIe' and specs['nvlink']):
                    suitable_gpus.append((gpu_name, specs))
            else:
                # No SXM requirement - all GPUs are suitable
                suitable_gpus.append((gpu_name, specs))
        
        if not suitable_gpus:
            return ("No suitable GPU found", 0), ("No suitable GPU found", 0)
        
        # Sort by VRAM (ascending) to find most cost-effective option
        suitable_gpus.sort(key=lambda x: x[1]['vram'])
        
        # MINIMUM GPU: First suitable GPU (lowest VRAM/cost)
        min_gpu_candidates = suitable_gpus[:3]  # Consider top 3 cheapest options
        if sxm_required:
            sxm_candidates = [gpu for gpu in min_gpu_candidates if gpu[1]['form_factor'] == 'SXM']
            if sxm_candidates:
                min_gpu_name = sxm_candidates[0][0]
            else:
                min_gpu_name = min_gpu_candidates[0][0]
        else:
            min_gpu_name = min_gpu_candidates[0][0]
        
        # OPTIMAL GPU: Based on workload complexity and performance requirements
        optimal_gpu_name = self.select_optimal_gpu(suitable_gpus, workload_complexity, sxm_required)
        
        return (min_gpu_name, max(1, multi_gpu_needed)), (optimal_gpu_name, max(1, multi_gpu_needed))

    def select_optimal_gpu(self, suitable_gpus: List[Tuple[str, Dict]], workload_complexity: str, sxm_required: bool) -> str:
        """Select optimal GPU based on workload complexity."""
        # Map complexity to performance tier
        complexity_to_tier = {
            'simple': 'mid',
            'moderate': 'high', 
            'complex': 'enterprise',
            'extreme': 'flagship'
        }
        
        target_tier = complexity_to_tier.get(workload_complexity, 'mid')
        
        # Find GPUs in the target performance tier
        tier_gpus = []
        for gpu_name, specs in suitable_gpus:
            for tier, gpu_list in self.performance_tiers.items():
                if gpu_name in gpu_list:
                    if tier == target_tier:
                        tier_gpus.append((gpu_name, specs))
                    break
        
        # If no GPUs in target tier, fall back to next best tier
        if not tier_gpus:
            tier_order = ['entry', 'mid', 'high', 'enterprise', 'flagship', 'cutting_edge']
            target_idx = tier_order.index(target_tier) if target_tier in tier_order else 1
            
            # Try higher tiers first, then lower
            for i in range(target_idx, len(tier_order)):
                tier_gpus = [(name, specs) for name, specs in suitable_gpus 
                           for gpu_list in [self.performance_tiers.get(tier_order[i], [])]
                           if name in gpu_list]
                if tier_gpus:
                    break
            
            # If still no match, try lower tiers
            if not tier_gpus:
                for i in range(target_idx - 1, -1, -1):
                    tier_gpus = [(name, specs) for name, specs in suitable_gpus 
                               for gpu_list in [self.performance_tiers.get(tier_order[i], [])]
                               if name in gpu_list]
                    if tier_gpus:
                        break
        
        # If still no tier match, use the highest VRAM GPU
        if not tier_gpus:
            tier_gpus = suitable_gpus
        
        # Prefer SXM if required
        if sxm_required:
            sxm_tier_gpus = [gpu for gpu in tier_gpus if gpu[1]['form_factor'] == 'SXM']
            if sxm_tier_gpus:
                tier_gpus = sxm_tier_gpus
        
        # Sort by compute capability and VRAM, return best
        tier_gpus.sort(key=lambda x: (x[1]['compute_capability'], x[1]['vram']), reverse=True)
        return tier_gpus[0][0]

    def analyze_notebook(self, url_or_path: str) -> GPURequirement:
        """Main analysis function."""
        if not self.quiet_mode:
            print(f"Analyzing: {url_or_path}")
        notebook = self.fetch_notebook(url_or_path)
        
        if not notebook:
            return GPURequirement(
                min_gpu_type="Unable to analyze",
                min_quantity=0,
                min_vram_gb=0,
                optimal_gpu_type="Unable to analyze",
                optimal_quantity=0,
                optimal_vram_gb=0,
                min_runtime_estimate="Unknown",
                optimal_runtime_estimate="Unknown",
                sxm_required=False,
                sxm_reasoning=["Failed to fetch notebook"],
                arm_compatibility="Unknown",
                arm_reasoning=["Failed to fetch notebook"],
                confidence=0.0,
                reasoning=["Failed to fetch notebook"],
                llm_enhanced=False,
                llm_reasoning=[],
                nvidia_compliance_score=0.0,
                structure_assessment={},
                content_quality_issues=[],
                technical_recommendations=[]
            )
        
        code_cells, markdown_cells = self.extract_code_cells(notebook)
        all_code = '\n'.join(code_cells)
        
        if not all_code.strip():
            return GPURequirement(
                min_gpu_type="No GPU required",
                min_quantity=0,
                min_vram_gb=0,
                optimal_gpu_type="No GPU required",
                optimal_quantity=0,
                optimal_vram_gb=0,
                min_runtime_estimate="N/A",
                optimal_runtime_estimate="N/A",
                sxm_required=False,
                sxm_reasoning=["No code cells found"],
                arm_compatibility="Compatible",
                arm_reasoning=["No GPU-specific code detected"],
                confidence=1.0,
                reasoning=["No code cells found"],
                llm_enhanced=False,
                llm_reasoning=[],
                nvidia_compliance_score=50.0,  # Basic structure still evaluable
                structure_assessment={"overall": "⚠️ Minimal content detected"},
                content_quality_issues=["Very little content to evaluate"],
                technical_recommendations=["Add substantial content and code examples"]
            )
        
        # Perform static analysis
        analysis = {
            'imports': self.analyze_imports(all_code),
            'models': self.analyze_model_usage(all_code),
            'batch_size': self.analyze_batch_size(all_code),
            'image_size': self.analyze_image_size(all_code),
            'is_training': self.is_training_code(all_code),
        }
        
        # Analyze SXM and ARM compatibility
        sxm_required, sxm_reasoning = self.analyze_sxm_requirements(all_code)
        arm_compatibility, arm_reasoning = self.analyze_arm_compatibility(all_code)
        
        # Analyze workload complexity
        workload_complexity = self.analyze_workload_complexity(all_code)
        
        # Estimate requirements
        vram_required, reasoning = self.estimate_vram_requirements(analysis)
        
        # Check for multi-GPU patterns
        multi_gpu_needed = 1
        if any(req['gpus'] > 1 for req in analysis['models'].values()):
            multi_gpu_needed = max(req['gpus'] for req in analysis['models'].values())
            reasoning.append(f"Multi-GPU model detected: {multi_gpu_needed} GPUs")
        
        # Prepare analysis for potential LLM enhancement
        analysis_data = {
            'vram_required': vram_required,
            'workload_complexity': workload_complexity,
            'multi_gpu_needed': multi_gpu_needed,
            'is_training': analysis['is_training']
        }
        
        # LLM Enhancement (if available)
        llm_enhanced = False
        llm_reasoning = []
        llm_compliance_evaluation = None
        
        if self.llm_analyzer:
            if not self.quiet_mode:
                print("🤖 Enhancing analysis with LLM...")
            llm_context = self.llm_analyzer.analyze_notebook_context(code_cells, markdown_cells)
            
            if llm_context:
                enhanced_analysis, llm_reasons = self.llm_analyzer.enhance_gpu_recommendation(analysis_data, llm_context)
                
                # Update analysis with LLM insights
                vram_required = enhanced_analysis.get('vram_required', vram_required)
                workload_complexity = enhanced_analysis.get('workload_complexity', workload_complexity)
                multi_gpu_needed = enhanced_analysis.get('multi_gpu_needed', multi_gpu_needed)
                
                llm_enhanced = True
                llm_reasoning = llm_reasons
                reasoning.extend([f"LLM: {reason}" for reason in llm_reasons])
                
                if not self.quiet_mode:
                    print(f"✅ LLM analysis complete (confidence: {llm_context.get('confidence', 0.5):.1%})")
            else:
                if not self.quiet_mode:
                    print("⚠️ LLM analysis failed, using static analysis only")
            
            # NVIDIA Compliance Evaluation with LLM
            if not self.quiet_mode:
                print("📋 Evaluating NVIDIA compliance...")
            llm_compliance_evaluation = self.llm_analyzer.evaluate_notebook_compliance(code_cells, markdown_cells)
            if llm_compliance_evaluation:
                if not self.quiet_mode:
                    print(f"✅ Compliance evaluation complete (score: {llm_compliance_evaluation.get('total_score', 0)}/100)")
            else:
                if not self.quiet_mode:
                    print("⚠️ LLM compliance evaluation failed, using static analysis")
        
        # Static compliance evaluation (always run as fallback/baseline)
        structure_assessment = self.evaluate_notebook_structure(code_cells, markdown_cells)
        content_quality_issues = self.assess_content_quality(code_cells, markdown_cells)
        technical_recommendations = self.check_technical_standards(code_cells)
        
        # Calculate compliance score
        nvidia_compliance_score = self.calculate_nvidia_compliance_score(
            structure_assessment, content_quality_issues, technical_recommendations, llm_compliance_evaluation
        )
        
        # Recommend GPUs (both minimum and optimal)
        (min_gpu_type, min_gpu_quantity), (optimal_gpu_type, optimal_gpu_quantity) = self.recommend_gpu(
            vram_required, multi_gpu_needed, sxm_required, workload_complexity
        )
        
        # Calculate VRAM usage ratios for runtime estimation
        min_gpu_vram = self.gpu_specs.get(min_gpu_type, {}).get('vram', vram_required)
        optimal_gpu_vram = self.gpu_specs.get(optimal_gpu_type, {}).get('vram', vram_required)
        
        min_vram_ratio = min(vram_required / min_gpu_vram, 1.0) if min_gpu_vram > 0 else 0.8
        optimal_vram_ratio = min(vram_required / optimal_gpu_vram, 1.0) if optimal_gpu_vram > 0 else 0.6
        
        # Estimate runtimes
        min_runtime = self.estimate_runtime(min_gpu_type, workload_complexity, min_vram_ratio, analysis['is_training'])
        optimal_runtime = self.estimate_runtime(optimal_gpu_type, workload_complexity, optimal_vram_ratio, analysis['is_training'])
        
        # Calculate confidence based on analysis depth
        base_confidence = 0.5  # Base confidence
        if analysis['imports']:
            base_confidence += 0.2
        if analysis['models']:
            base_confidence += 0.2
        if analysis['batch_size']:
            base_confidence += 0.1
        
        # LLM enhancement boosts confidence
        if llm_enhanced:
            base_confidence = min(base_confidence * 1.3, 1.0)  # 30% boost with LLM
        
        confidence = min(base_confidence, 1.0)
        
        return GPURequirement(
            min_gpu_type=min_gpu_type,
            min_quantity=min_gpu_quantity,
            min_vram_gb=vram_required,
            optimal_gpu_type=optimal_gpu_type,
            optimal_quantity=optimal_gpu_quantity,
            optimal_vram_gb=int(optimal_gpu_vram),
            min_runtime_estimate=min_runtime,
            optimal_runtime_estimate=optimal_runtime,
            sxm_required=sxm_required,
            sxm_reasoning=sxm_reasoning,
            arm_compatibility=arm_compatibility,
            arm_reasoning=arm_reasoning,
            confidence=confidence,
            reasoning=reasoning,
            llm_enhanced=llm_enhanced,
            llm_reasoning=llm_reasoning or [],
            nvidia_compliance_score=nvidia_compliance_score,
            structure_assessment=structure_assessment,
            content_quality_issues=content_quality_issues,
            technical_recommendations=technical_recommendations
        )

def main():
    parser = argparse.ArgumentParser(
        description='Analyze Jupyter notebook GPU requirements',
        epilog="""
Environment Variables:
  OPENAI_BASE_URL    OpenAI API endpoint (for LLM enhancement)
  OPENAI_API_KEY     OpenAI API key (for LLM enhancement)  
  OPENAI_MODEL       Model name (default: gpt-4)
  GITHUB_TOKEN       GitHub personal access token (for private repos)
  GITLAB_TOKEN       GitLab personal access token (for private repos)

Examples:
  # Analyze public notebook
  python notebook-analyzer.py https://github.com/user/repo/blob/main/notebook.ipynb
  
  # Analyze private GitHub notebook (set GITHUB_TOKEN first)
  export GITHUB_TOKEN=your_token_here
  python notebook-analyzer.py https://github.com/private/repo/blob/main/notebook.ipynb
  
  # Analyze GitLab notebook (public or private)
  export GITLAB_TOKEN=your_gitlab_token_here
  python notebook-analyzer.py https://gitlab.com/user/repo/-/blob/main/notebook.ipynb
  
  # Analyze local notebook file
  python notebook-analyzer.py ./path/to/notebook.ipynb
  
  # Use raw URL with token (automatically handles shell quoting)
  python notebook-analyzer.py https://raw.githubusercontent.com/repo/file.ipynb?token=...
  
  # Get JSON output for automation/scripting
  python notebook-analyzer.py --json https://github.com/user/repo/blob/main/notebook.ipynb
  
  # Verbose JSON output (pretty-printed)
  python notebook-analyzer.py --json --verbose ./notebook.ipynb
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('url_or_path', nargs='*', help='URL to notebook or local file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output with detailed reasoning')
    parser.add_argument('--json', '-j', action='store_true', help='Output results in JSON format')
    
    args = parser.parse_args()
    
    # Handle shell quoting issues and multiple arguments
    if not args.url_or_path:
        parser.print_help()
        sys.exit(1)
    
    analyzer = GPUAnalyzer(quiet_mode=args.json)
    
    # Automatically handle URL reconstruction from shell arguments
    if len(args.url_or_path) > 1:
        if not args.json:
            print("🔧 Multiple arguments detected - reconstructing URL...")
        url_or_path = analyzer.sanitize_url_args(args.url_or_path)
        if not args.json:
            print(f"📝 Reconstructed: {url_or_path}")
    else:
        url_or_path = args.url_or_path[0]
    
    result = analyzer.analyze_notebook(url_or_path)
    
    # Handle JSON output
    if args.json:
        import json as json_module
        
        # Convert dataclass to dictionary
        result_dict = {
            'min_gpu_type': result.min_gpu_type,
            'min_quantity': result.min_quantity,
            'min_vram_gb': result.min_vram_gb,
            'optimal_gpu_type': result.optimal_gpu_type,
            'optimal_quantity': result.optimal_quantity,
            'optimal_vram_gb': result.optimal_vram_gb,
            'min_runtime_estimate': result.min_runtime_estimate,
            'optimal_runtime_estimate': result.optimal_runtime_estimate,
            'sxm_required': result.sxm_required,
            'sxm_reasoning': result.sxm_reasoning,
            'arm_compatibility': result.arm_compatibility,
            'arm_reasoning': result.arm_reasoning,
            'confidence': result.confidence,
            'reasoning': result.reasoning,
            'llm_enhanced': result.llm_enhanced,
            'llm_reasoning': result.llm_reasoning,
            'nvidia_compliance_score': result.nvidia_compliance_score,
            'structure_assessment': result.structure_assessment,
            'content_quality_issues': result.content_quality_issues,
            'technical_recommendations': result.technical_recommendations,
            'analysis_metadata': {
                'analyzed_url_or_path': url_or_path,
                'timestamp': __import__('datetime').datetime.now().isoformat(),
                'version': '3.0.0'
            }
        }
        
        print(json_module.dumps(result_dict, indent=2 if args.verbose else None))
        return
    
    print("\n" + "="*70)
    print("GPU REQUIREMENTS ANALYSIS")
    print("="*70)
    
    print("\n📊 MINIMUM REQUIREMENTS:")
    print(f"   GPU Type: {result.min_gpu_type}")
    print(f"   Quantity: {result.min_quantity}")
    print(f"   VRAM: {result.min_vram_gb} GB")
    print(f"   Estimated Runtime: {result.min_runtime_estimate}")
    
    print("\n🚀 OPTIMAL CONFIGURATION:")
    print(f"   GPU Type: {result.optimal_gpu_type}")
    print(f"   Quantity: {result.optimal_quantity}")
    print(f"   VRAM: {result.optimal_vram_gb} GB")
    print(f"   Estimated Runtime: {result.optimal_runtime_estimate}")
    
    print(f"\n📋 NVIDIA NOTEBOOK COMPLIANCE: {result.nvidia_compliance_score:.0f}/100")
    
    # Color-code compliance score
    if result.nvidia_compliance_score >= 85:
        compliance_icon = "🟢"
    elif result.nvidia_compliance_score >= 70:
        compliance_icon = "🟡"
    else:
        compliance_icon = "🔴"
    
    print(f"{compliance_icon} Overall Quality Score")
    
    # Always show structure assessment (even in non-verbose mode for compliance)
    if hasattr(result, 'structure_assessment') and result.structure_assessment:
        print("\n📚 Structure & Layout Assessment:")
        for category, status in result.structure_assessment.items():
            print(f"     {category.title()}: {status}")
    
    # Show top content and technical recommendations (even in non-verbose)
    if hasattr(result, 'content_quality_issues') and result.content_quality_issues:
        print("\n🎯 Content Quality Recommendations:")
        # Show first 3 issues in non-verbose, all in verbose
        issues_to_show = result.content_quality_issues[:3] if not args.verbose else result.content_quality_issues
        for issue in issues_to_show:
            print(f"     • {issue}")
        if not args.verbose and len(result.content_quality_issues) > 3:
            print(f"     • ... and {len(result.content_quality_issues) - 3} more (use -v for all)")
    
    if hasattr(result, 'technical_recommendations') and result.technical_recommendations:
        print("\n🔧 Technical Standards Recommendations:")
        # Show first 3 recommendations in non-verbose, all in verbose
        recs_to_show = result.technical_recommendations[:3] if not args.verbose else result.technical_recommendations
        for rec in recs_to_show:
            print(f"     • {rec}")
        if not args.verbose and len(result.technical_recommendations) > 3:
            print(f"     • ... and {len(result.technical_recommendations) - 3} more (use -v for all)")
    
    print(f"\n💡 ADDITIONAL INFO:")
    print(f"   SXM Form Factor Required: {'Yes' if result.sxm_required else 'No'}")
    print(f"   ARM/Grace Compatibility: {result.arm_compatibility}")
    print(f"   Analysis Confidence: {result.confidence:.1%}")
    print(f"   LLM Enhanced: {'Yes' if result.llm_enhanced else 'No'}")
    
    if args.verbose:
        if result.reasoning:
            print("\n🔍 GPU Requirements Reasoning:")
            for reason in result.reasoning:
                print(f"     • {reason}")
        
        if result.llm_enhanced and result.llm_reasoning:
            print("\n🤖 LLM Analysis Insights:")
            for reason in result.llm_reasoning:
                print(f"     • {reason}")
        
        if result.sxm_reasoning:
            print("\n🔗 SXM Requirements:")
            for reason in result.sxm_reasoning:
                print(f"     • {reason}")
        
        if result.arm_reasoning:
            print(f"\n🖥️  ARM/Grace Compatibility ({result.arm_compatibility}):")
            for reason in result.arm_reasoning:
                print(f"     • {reason}")
    
    if not result.llm_enhanced:
        print(f"\n💡 Tip: Set OPENAI_BASE_URL and OPENAI_API_KEY environment variables for enhanced LLM analysis")
    
    print("="*70)

if __name__ == "__main__":
    main()

