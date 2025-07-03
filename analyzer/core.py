#!/usr/bin/env python3
"""
Notebook Analyzer Core Module

Enhanced analysis classes that use comprehensive NVIDIA Best Practices
for evaluating Jupyter notebooks and determining GPU requirements.
"""

import requests
import json
import re
import ast
import os
import sys
import ipaddress
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from urllib.parse import urlparse, quote_plus


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
    llm_reasoning: Optional[List[str]] = None
    nvidia_compliance_score: float = 0.0
    structure_assessment: Optional[Dict[str, str]] = None
    content_quality_issues: Optional[List[str]] = None
    technical_recommendations: Optional[List[str]] = None
    
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
    def __init__(self, base_url: str, model: str, api_key: str):
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
    
    def analyze_notebook_context(self, code_cells: List[str], markdown_cells: List[str]) -> Optional[Dict]:
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
        
        # Check if LLM detected no workload that requires GPU
        llm_complexity = llm_context.get('complexity', '').lower()
        if llm_complexity in ['simple', 'none', 'basic']:
            # Check if LLM reasoning indicates no GPU workload
            llm_reasoning_text = ' '.join(llm_context.get('reasoning', [])).lower()
            no_workload_indicators = [
                'no explicit models', 'no training', 'no inference', 'no gpu',
                'triggering compatibility warnings', 'no workload', 'no ml workload',
                'no evidence of training', 'no evidence of inference'
            ]
            
            if any(indicator in llm_reasoning_text for indicator in no_workload_indicators):
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
        
        # Only proceed with GPU enhancement if workload was detected
        if not static_analysis.get('workload_detected', True):
            llm_reasoning.append("No GPU workload detected - maintaining CPU-only recommendation")
            return enhanced_analysis, llm_reasoning
        
        # Enhance VRAM estimate with LLM insights
        original_vram = static_analysis.get('min_vram_gb', 8)
        updated_vram = original_vram
        
        if 'estimated_vram_gb' in llm_context and llm_context['estimated_vram_gb']:
            llm_vram = llm_context['estimated_vram_gb']
            static_vram = static_analysis.get('min_vram_gb', 8)
            
            # Use higher estimate but cap at reasonable limits
            enhanced_vram = max(llm_vram, static_vram)
            updated_vram = min(enhanced_vram, 200)  # Cap at 200GB
            enhanced_analysis['min_vram_gb'] = updated_vram
            
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
                # Only recommend SXM GPUs when multi-GPU is required
                if updated_vram <= 80:
                    enhanced_analysis['min_gpu_type'] = 'A100 SXM 80G'
                    enhanced_analysis['min_vram_gb'] = max(updated_vram, 80)
                    enhanced_analysis['optimal_gpu_type'] = 'H100 SXM'
                    enhanced_analysis['optimal_vram_gb'] = max(updated_vram, 80)
                    enhanced_analysis['min_runtime_estimate'] = '3-6 hours'
                    enhanced_analysis['optimal_runtime_estimate'] = '1-2 hours'
                elif updated_vram <= 141:
                    enhanced_analysis['min_gpu_type'] = 'H100 SXM'
                    enhanced_analysis['min_vram_gb'] = max(updated_vram, 80)
                    enhanced_analysis['optimal_gpu_type'] = 'H200 SXM'
                    enhanced_analysis['optimal_vram_gb'] = max(updated_vram, 141)
                    enhanced_analysis['min_runtime_estimate'] = '4-8 hours'
                    enhanced_analysis['optimal_runtime_estimate'] = '1-3 hours'
                else:
                    enhanced_analysis['min_gpu_type'] = 'H200 SXM'
                    enhanced_analysis['min_vram_gb'] = max(updated_vram, 141)
                    enhanced_analysis['optimal_gpu_type'] = 'B200 SXM'
                    enhanced_analysis['optimal_vram_gb'] = max(updated_vram, 192)
                    enhanced_analysis['min_runtime_estimate'] = '6-12 hours'
                    enhanced_analysis['optimal_runtime_estimate'] = '2-4 hours'
                
                llm_reasoning.append(f"SXM GPUs selected due to multi-GPU requirement: {enhanced_analysis['min_gpu_type']} -> {enhanced_analysis['optimal_gpu_type']}")
            else:
                # Apply standard GPU selection logic for single-GPU workloads
                if updated_vram <= 8:
                    # Entry-level workload
                    enhanced_analysis['min_gpu_type'] = 'RTX 4060'
                    enhanced_analysis['min_vram_gb'] = updated_vram
                    enhanced_analysis['optimal_gpu_type'] = 'RTX 4070'
                    enhanced_analysis['optimal_vram_gb'] = max(updated_vram, 12)
                    enhanced_analysis['min_runtime_estimate'] = '1-2 hours'
                    enhanced_analysis['optimal_runtime_estimate'] = '30-60 minutes'
                elif updated_vram <= 16:
                    # Mid-tier workload  
                    enhanced_analysis['min_gpu_type'] = 'RTX 4070'
                    enhanced_analysis['min_vram_gb'] = max(updated_vram, 12)
                    enhanced_analysis['optimal_gpu_type'] = 'RTX 4080'
                    enhanced_analysis['optimal_vram_gb'] = max(updated_vram, 16)
                    enhanced_analysis['min_runtime_estimate'] = '2-4 hours'
                    enhanced_analysis['optimal_runtime_estimate'] = '1-2 hours'
                elif updated_vram <= 24:
                    # High-end workload
                    enhanced_analysis['min_gpu_type'] = 'RTX 4090'
                    enhanced_analysis['min_vram_gb'] = max(updated_vram, 24)
                    enhanced_analysis['optimal_gpu_type'] = 'L4'
                    enhanced_analysis['optimal_vram_gb'] = max(updated_vram, 24)
                    enhanced_analysis['min_runtime_estimate'] = '3-6 hours'
                    enhanced_analysis['optimal_runtime_estimate'] = '1-2 hours'
                else:
                    # Enterprise workload - use PCIe GPUs for single-GPU high-VRAM needs
                    enhanced_analysis['min_gpu_type'] = 'L40S'
                    enhanced_analysis['min_vram_gb'] = max(updated_vram, 48)
                    enhanced_analysis['optimal_gpu_type'] = 'A100 PCIe 80G'
                    enhanced_analysis['optimal_vram_gb'] = max(updated_vram, 80)
                    enhanced_analysis['min_runtime_estimate'] = '4-8 hours'
                    enhanced_analysis['optimal_runtime_estimate'] = '1-3 hours'
                
                llm_reasoning.append(f"Updated GPU recommendations: {enhanced_analysis['min_gpu_type']} (min) -> {enhanced_analysis['optimal_gpu_type']} (optimal)")
        
        # Multi-GPU insights
        if 'multi_gpu_required' in llm_context and llm_context['multi_gpu_required']:
            if not static_analysis.get('optimal_quantity', 1) > 1:
                enhanced_analysis['optimal_quantity'] = 2
                enhanced_analysis['sxm_required'] = True
                llm_reasoning.append("LLM detected multi-GPU requirements not caught by static analysis")
        
        # Add LLM reasoning
        if 'reasoning' in llm_context:
            llm_reasoning.extend(llm_context['reasoning'])
        
        return enhanced_analysis, llm_reasoning
    
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
        
        # Initialize LLM if API key is available
        openai_api_key = os.getenv('OPENAI_API_KEY')
        openai_base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com')
        openai_model = os.getenv('OPENAI_MODEL', 'gpt-4')
        
        if openai_api_key:
            self.llm_analyzer = LLMAnalyzer(
                base_url=openai_base_url,
                model=openai_model,
                api_key=openai_api_key
            )
            if not self.quiet_mode:
                if 'nvidia.com' in openai_base_url:
                    print(f"✅ LLM enhancement enabled using NVIDIA API with {openai_model}")
                else:
                    print(f"✅ LLM enhancement enabled using {openai_model}")
        else:
            if not self.quiet_mode:
                print("⚠️ No LLM API key found. Analysis will use static patterns only.")
        
        # Enhanced GPU specifications with detailed metadata
        self.gpu_specs = {
            # RTX 50 Series
            'RTX 5090': {
                'vram': 32, 'compute_capability': 9.0, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2025, 'tier': 'flagship', 'tensor_cores': True
            },
            'RTX 5080': {
                'vram': 16, 'compute_capability': 9.0, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2025, 'tier': 'high', 'tensor_cores': True
            },
            
            # RTX 40 Series
            'RTX 4090': {
                'vram': 24, 'compute_capability': 8.9, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2022, 'tier': 'flagship', 'tensor_cores': True
            },
            'RTX 4080': {
                'vram': 16, 'compute_capability': 8.9, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2022, 'tier': 'high', 'tensor_cores': True
            },
            
            # Professional GPUs
            'L4': {
                'vram': 24, 'compute_capability': 8.9, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2023, 'tier': 'mid', 'tensor_cores': True
            },
            'L40': {
                'vram': 48, 'compute_capability': 8.9, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2023, 'tier': 'high', 'tensor_cores': True
            },
            'L40S': {
                'vram': 48, 'compute_capability': 8.9, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2023, 'tier': 'high', 'tensor_cores': True
            },
            
            # Data Center GPUs
            'A100 SXM 80G': {
                'vram': 80, 'compute_capability': 8.0, 'form_factor': 'SXM', 'nvlink': True,
                'release_year': 2020, 'tier': 'enterprise', 'tensor_cores': True
            },
            'A100 PCIe 80G': {
                'vram': 80, 'compute_capability': 8.0, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2020, 'tier': 'enterprise', 'tensor_cores': True
            },
            'H100 SXM': {
                'vram': 80, 'compute_capability': 9.0, 'form_factor': 'SXM', 'nvlink': True,
                'release_year': 2022, 'tier': 'cutting_edge', 'tensor_cores': True
            },
            'H100 PCIe': {
                'vram': 80, 'compute_capability': 9.0, 'form_factor': 'PCIe', 'nvlink': False,
                'release_year': 2022, 'tier': 'cutting_edge', 'tensor_cores': True
            },
            'H200 SXM': {
                'vram': 141, 'compute_capability': 9.0, 'form_factor': 'SXM', 'nvlink': True,
                'release_year': 2023, 'tier': 'cutting_edge', 'tensor_cores': True
            },
            'B200 SXM': {
                'vram': 192, 'compute_capability': 10.0, 'form_factor': 'SXM', 'nvlink': True,
                'release_year': 2024, 'tier': 'cutting_edge', 'tensor_cores': True
            }
        }
        
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
            'cuda_computing': [r'numba\.cuda', r'from numba import cuda', r'cuda\.', r'pycuda\.', r'import pycuda'],
            'nvidia_frameworks': [r'nemo\.', r'import nemo', r'triton\.', r'import triton', r'tensorrt\.', r'import tensorrt']
        }
        
        # SXM requirement patterns (enhanced with more precise detection)
        self.sxm_patterns = [
            # Real distributed training patterns
            r'\btorch\.nn\.DataParallel\b', r'\btorch\.nn\.parallel\.DistributedDataParallel\b',
            r'\bmodel\s*=\s*nn\.DataParallel\b', r'\bmodel\s*=\s*DDP\b',
            r'\bhorovod\b', r'\bnccl\b', r'\ball_reduce\b', r'\ball_gather\b',
            
            # Actual multi-GPU training setup (not just mentions)
            r'\btorch\.cuda\.device_count\(\)\s*>\s*1', r'\bworld_size\s*>\s*1',
            r'\baccelerator\s*=\s*["\']ddp["\']', r'\bstrategy\s*=\s*["\']ddp["\']',
            r'\bdist\.init_process_group\b', r'\bdist\.barrier\b',
            
            # Pipeline/tensor parallelism (actual usage, not comments)
            r'\bPipelineParallel\b', r'\bTensorParallel\b',
            r'\btensor_model_parallel_size\s*>\s*1',
            r'\bpipeline_model_parallel_size\s*>\s*1',
            
            # DeepSpeed multi-GPU
            r'\bdeepspeed\.initialize\b.*world_size', r'\bzero_stage\s*[>=]\s*2',
            
            # NVIDIA NeMo multi-GPU
            r'\btrainer\.num_nodes\s*>\s*1', r'\btrainer\.devices\s*>\s*1'
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
        if self.llm_analyzer:
            if not self.quiet_mode:
                print("🤖 Enhancing analysis with LLM...")
            llm_context = self.llm_analyzer.analyze_notebook_context(code_cells, markdown_cells)
            if llm_context:
                enhanced_analysis, llm_reasoning = self.llm_analyzer.enhance_gpu_recommendation(static_analysis, llm_context)
                static_analysis.update(enhanced_analysis)
                if not self.quiet_mode:
                    print(f"✅ LLM analysis complete (confidence: {llm_context.get('confidence', 0.5)*100:.0f}%)")
        
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
            optimal_gpu_type=static_analysis['optimal_gpu_type'],
            optimal_quantity=static_analysis['optimal_quantity'],
            optimal_vram_gb=static_analysis['optimal_vram_gb'],
            min_runtime_estimate=static_analysis['min_runtime_estimate'],
            optimal_runtime_estimate=static_analysis['optimal_runtime_estimate'],
            sxm_required=static_analysis['sxm_required'],
            sxm_reasoning=static_analysis['sxm_reasoning'],
            arm_compatibility=static_analysis['arm_compatibility'],
            arm_reasoning=static_analysis['arm_reasoning'],
            confidence=static_analysis['confidence'],
            reasoning=static_analysis['reasoning'],
            llm_enhanced=llm_context is not None,
            llm_reasoning=llm_reasoning,
            nvidia_compliance_score=compliance_score,
            structure_assessment=structure_assessment,
            content_quality_issues=content_issues,
            technical_recommendations=technical_recommendations
        )
        
    def _extract_notebook_content(self, url_or_path: str) -> Tuple[List[str], List[str]]:
        """Extract code and markdown cells from notebook."""
        import urllib.parse
        import tempfile
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
            
            # Download notebook content
            response = requests.get(url_or_path, timeout=30)
            response.raise_for_status()
            content = response.text
            
            # Create temporary file
            if url_or_path.endswith('.py'):
                # marimo notebook
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
                notebook_data = json.loads(content)
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
        """Perform static analysis of notebook content."""
        # Start with CPU-only defaults and scale up based on detected patterns
        analysis = {
            'min_gpu_type': 'CPU-only',
            'min_quantity': 0,
            'min_vram_gb': 0,
            'optimal_gpu_type': 'CPU-only',
            'optimal_quantity': 0,
            'optimal_vram_gb': 0,
            'min_runtime_estimate': 'N/A',
            'optimal_runtime_estimate': 'N/A',
            'sxm_required': False,
            'sxm_reasoning': [],
            'arm_compatibility': 'Likely Compatible',
            'arm_reasoning': [],
            'confidence': 0.7,
            'reasoning': [],
            'workload_detected': False,
            'workload_type': 'none'
        }
        
        all_code = '\n'.join(code_cells)
        
        # First, detect if there's any actual GPU/ML workload
        gpu_workload_patterns = [
            # Training patterns
            r'\b(?:train|fit|epoch|optimizer|loss|backward|gradient)\b',
            # Inference patterns
            r'\b(?:predict|inference|forward|model\.eval|torch\.no_grad)\b',
            # Model loading/creation
            r'\b(?:from_pretrained|load_model|create_model|Model\(|Sequential\()\b',
            # GPU-specific operations
            r'\b(?:\.cuda\(\)|\.gpu\(\)|device=.*["\']cuda["\']|torch\.cuda|CUDA_VISIBLE_DEVICES)\b',
            # Deep learning frameworks in actual usage (not just imports)
            r'\b(?:torch\.|tf\.|tensorflow\.|model\.|transformers\.|datasets\.)\w+',
            # Data processing for ML
            r'\b(?:DataLoader|Dataset|batch_size|transform|preprocessing)\b',
            # GPU-accelerated data processing (cuDF/RAPIDS)
            r'\b(?:cudf\.|cupy\.|cuml\.|cugraph\.)\w+',
            r'backend=["\']cudf["\']|engine=["\']cudf["\']',
            # NVIDIA GPU computing
            r'\b(?:numba\.cuda|pycuda\.|tensorrt\.)\w+',
            # GPU memory management
            r'\b(?:cuda_memory|gpu_memory|cuda\.mem)\b'
        ]
        
        workload_detected = any(re.search(pattern, all_code, re.IGNORECASE) for pattern in gpu_workload_patterns)
        
        if not workload_detected:
            # Check for simple library imports that might indicate ML intent
            simple_ml_imports = [
                r'import torch\b', r'import tensorflow\b', r'import transformers\b',
                r'from torch', r'from tensorflow', r'from transformers',
                r'import sklearn\b', r'from sklearn'
            ]
            has_ml_imports = any(re.search(pattern, all_code, re.IGNORECASE) for pattern in simple_ml_imports)
            
            if has_ml_imports:
                # Has ML imports but no actual workload - might be a demo/tutorial
                analysis['workload_type'] = 'demonstration'
                analysis['reasoning'].append("ML libraries imported but no active GPU workload detected")
            else:
                # No ML workload at all
                analysis['workload_type'] = 'none'
                analysis['reasoning'].append("No GPU workload or ML frameworks detected - CPU-only notebook")
                return analysis
        
        # If we get here, there's some form of GPU/ML workload
        analysis['workload_detected'] = True
        
        # Detect frameworks and patterns
        detected_frameworks = []
        for framework, patterns in self.framework_patterns.items():
            if any(re.search(pattern, all_code, re.IGNORECASE) for pattern in patterns):
                detected_frameworks.append(framework)
        
        if not detected_frameworks:
            # Has workload patterns but no clear framework detection
            analysis['workload_type'] = 'basic'
            analysis['min_gpu_type'] = 'RTX 4060'
            analysis['min_quantity'] = 1
            analysis['min_vram_gb'] = 8
            analysis['optimal_gpu_type'] = 'RTX 4070'
            analysis['optimal_quantity'] = 1
            analysis['optimal_vram_gb'] = 12
            analysis['min_runtime_estimate'] = '30-60 minutes'
            analysis['optimal_runtime_estimate'] = '15-30 minutes'
            analysis['reasoning'].append("Basic GPU workload detected - entry-level GPU recommended")
            return analysis
        
        # Estimate VRAM requirements based on detected models
        estimated_vram = 8  # Start with entry-level requirement
        
        for model, specs in self.model_specs.items():
            if model in all_code.lower():
                estimated_vram = max(estimated_vram, specs['base_vram'])
                analysis['reasoning'].append(f"Detected {model} model requiring {specs['base_vram']}GB+ VRAM")
        
        # Check for training vs inference patterns
        training_patterns = ['train', 'fit', 'epoch', 'optimizer', 'loss', 'backward']
        is_training = any(pattern in all_code.lower() for pattern in training_patterns)
        
        if is_training:
            estimated_vram = int(estimated_vram * 1.5)  # Training requires more memory
            analysis['reasoning'].append("Training workload detected - increased VRAM estimate")
            analysis['workload_type'] = 'training'
        else:
            analysis['workload_type'] = 'inference'
            analysis['reasoning'].append("Inference workload detected")
        
        # Check for multi-GPU patterns
        for pattern in self.sxm_patterns:
            if re.search(pattern, all_code, re.IGNORECASE):
                analysis['sxm_required'] = True
                analysis['sxm_reasoning'].append(f"Multi-GPU pattern detected: {pattern}")
        
        # Set GPU recommendations based on estimated VRAM needs
        if estimated_vram <= 8:
            # Entry-level workload
            analysis['min_gpu_type'] = 'RTX 4060'
            analysis['min_vram_gb'] = 8
            analysis['optimal_gpu_type'] = 'RTX 4070'
            analysis['optimal_vram_gb'] = 12
            analysis['min_runtime_estimate'] = '1-2 hours'
            analysis['optimal_runtime_estimate'] = '30-60 minutes'
        elif estimated_vram <= 16:
            # Mid-tier workload
            analysis['min_gpu_type'] = 'RTX 4070'
            analysis['min_vram_gb'] = 12
            analysis['optimal_gpu_type'] = 'RTX 4080'
            analysis['optimal_vram_gb'] = 16
            analysis['min_runtime_estimate'] = '2-4 hours'
            analysis['optimal_runtime_estimate'] = '1-2 hours'
        elif estimated_vram <= 24:
            # High-end workload
            analysis['min_gpu_type'] = 'RTX 4090'
            analysis['min_vram_gb'] = 24
            analysis['optimal_gpu_type'] = 'L4'
            analysis['optimal_vram_gb'] = 24
            analysis['min_runtime_estimate'] = '3-6 hours'
            analysis['optimal_runtime_estimate'] = '1-2 hours'
        else:
            # Enterprise workload
            analysis['min_gpu_type'] = 'L4'
            analysis['min_vram_gb'] = 24
            analysis['optimal_gpu_type'] = 'A100 SXM 80G'
            analysis['optimal_vram_gb'] = 80
            analysis['min_runtime_estimate'] = '4-8 hours'
            analysis['optimal_runtime_estimate'] = '1-3 hours'
        
        # Set quantities
        analysis['min_quantity'] = 1
        analysis['optimal_quantity'] = 1
        if analysis['sxm_required']:
            analysis['optimal_quantity'] = 2
            analysis['reasoning'].append("Multi-GPU setup recommended for optimal performance")

        # CRITICAL FIX: Validate SXM requirements against selected GPUs
        analysis = self._validate_sxm_requirements(analysis)

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
                if self._compare_versions(version, min_version) < 0:
                    arm_issues.append(f"{package} v{version} may have ARM issues (min recommended: v{min_version})")
                    arm_compatibility_score -= 20
                else:
                    optimal_version = self.arm_compatibility_versions[package]['optimal']
                    if self._compare_versions(version, optimal_version) >= 0:
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
        if is_training:
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
                if estimated_vram > 40:
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
            # Check if this is likely a false positive by examining the reasoning
            sxm_reasoning = ' '.join(analysis.get('sxm_reasoning', [])).lower()
            
            # If the detected pattern is vague or likely false positive, clear the requirement
            false_positive_indicators = [
                'model.*parallel',  # Our old broad pattern
                'pipeline.*parallel',  # Might be just mentions
                'tensor.*parallel'  # Might be just mentions
            ]
            
            is_likely_false_positive = any(indicator in sxm_reasoning for indicator in false_positive_indicators)
            
            if is_likely_false_positive:
                # Clear false positive SXM requirement
                analysis['sxm_required'] = False
                analysis['sxm_reasoning'] = ["SXM requirement cleared - pattern detected was likely a false positive"]
                analysis['reasoning'].append("Cleared false positive SXM requirement - workload appears to be single-GPU")
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
                
                analysis['reasoning'].append(f"Upgraded to SXM GPUs due to multi-GPU requirement: {analysis['min_gpu_type']} -> {analysis['optimal_gpu_type']}")
        
        return analysis

    def _compare_versions(self, version1: str, version2: str) -> int:
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
    
    def sanitize_file_content(self, file_content: bytes, filename: str) -> tuple:
        """
        Sanitize uploaded file content for security.
        Returns (is_safe: bool, error_msg: str, sanitized_content: dict/str)
        """
        try:
            if filename.lower().endswith('.ipynb'):
                # Jupyter notebook - validate JSON structure
                try:
                    import json
                    content_str = file_content.decode('utf-8')
                    notebook_data = json.loads(content_str)
                    
                    # Basic validation
                    if not isinstance(notebook_data, dict):
                        return False, "Invalid notebook format - not a JSON object", None
                    
                    if 'cells' not in notebook_data:
                        return False, "Invalid notebook format - missing cells", None
                    
                    if not isinstance(notebook_data['cells'], list):
                        return False, "Invalid notebook format - cells must be an array", None
                    
                    # Basic security checks
                    content_lower = content_str.lower()
                    dangerous_patterns = [
                        'subprocess.', 'os.system', 'eval(', 'exec(',
                        '__import__', 'open(', 'file(', 'input('
                    ]
                    
                    for pattern in dangerous_patterns:
                        if pattern in content_lower:
                            return False, f"Potentially unsafe code pattern detected: {pattern}", None
                    
                    return True, "", notebook_data
                    
                except json.JSONDecodeError:
                    return False, "Invalid JSON format", None
                except UnicodeDecodeError:
                    return False, "Invalid file encoding - please use UTF-8", None
                    
            elif filename.lower().endswith('.py'):
                # Python/marimo file
                try:
                    content_str = file_content.decode('utf-8')
                    
                    # Basic security checks
                    content_lower = content_str.lower()
                    dangerous_patterns = [
                        'subprocess.', 'os.system', 'eval(', 'exec(',
                        '__import__', 'open(', 'file(', 'input('
                    ]
                    
                    for pattern in dangerous_patterns:
                        if pattern in content_lower:
                            return False, f"Potentially unsafe code pattern detected: {pattern}", None
                    
                    # Basic Python syntax validation
                    import ast
                    try:
                        ast.parse(content_str)
                    except SyntaxError:
                        return False, "Invalid Python syntax", None
                    
                    return True, "", {"content": content_str}
                    
                except UnicodeDecodeError:
                    return False, "Invalid file encoding - please use UTF-8", None
            else:
                return False, "Unsupported file type", None
                
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