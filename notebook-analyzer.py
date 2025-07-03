#!/usr/bin/env python3
"""
Notebook Analyzer for Jupyter Notebooks

Enhanced analyzer with comprehensive NVIDIA Best Practices integration.
This script analyzes Jupyter notebooks to determine minimum GPU requirements
based on the code patterns, libraries used, and computational workloads.
"""

import argparse
import json as json_module
import sys

# Import enhanced analyzer components
from analyzer import GPUAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Notebook Analyzer with NVIDIA Best Practices',
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
    
    # Initialize enhanced analyzer with NVIDIA Best Practices
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
    
    # Perform enhanced analysis
    result = analyzer.analyze_notebook(url_or_path)
    
    # Handle JSON output
    if args.json:
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
                'version': '3.1.0',
                'enhanced_features': 'NVIDIA Best Practices Integration'
            }
        }
        
        print(json_module.dumps(result_dict, indent=2 if args.verbose else None))
        return
    
    # Enhanced human-readable output
    print("\n" + "="*70)
    print("📋 ENHANCED GPU REQUIREMENTS ANALYSIS")
    print("🎯 WITH NVIDIA BEST PRACTICES COMPLIANCE")
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
    
    # Enhanced compliance score display
    if result.nvidia_compliance_score >= 85:
        compliance_icon = "🟢 Excellent"
        compliance_msg = "Meets NVIDIA's highest standards"
    elif result.nvidia_compliance_score >= 70:
        compliance_icon = "🟡 Good"
        compliance_msg = "Generally follows NVIDIA best practices"
    elif result.nvidia_compliance_score >= 50:
        compliance_icon = "🟠 Fair" 
        compliance_msg = "Some improvements needed for NVIDIA standards"
    else:
        compliance_icon = "🔴 Needs Improvement"
        compliance_msg = "Significant improvements required for NVIDIA compliance"
    
    print(f"{compliance_icon} - {compliance_msg}")
    
    # Always show structure assessment
    if hasattr(result, 'structure_assessment') and result.structure_assessment:
        print("\n📚 Structure & Layout Assessment:")
        for category, status in result.structure_assessment.items():
            print(f"     {category.title()}: {status}")
    
    # Show enhanced content and technical recommendations
    if hasattr(result, 'content_quality_issues') and result.content_quality_issues:
        print("\n🎯 Content Quality Recommendations:")
        issues_to_show = result.content_quality_issues[:3] if not args.verbose else result.content_quality_issues
        for issue in issues_to_show:
            print(f"     • {issue}")
        if not args.verbose and len(result.content_quality_issues) > 3:
            print(f"     • ... and {len(result.content_quality_issues) - 3} more (use -v for all)")
    
    if hasattr(result, 'technical_recommendations') and result.technical_recommendations:
        print("\n🔧 Technical Standards Recommendations:")
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
    print(f"   NVIDIA Best Practices: ✅ Loaded")
    
    # Show confidence factors if available and verbose mode
    if hasattr(result, 'confidence_factors') and result.confidence_factors and args.verbose:
        print(f"\n🔍 CONFIDENCE FACTORS:")
        for factor in result.confidence_factors:
            print(f"     • {factor}")
    elif hasattr(result, 'confidence_factors') and result.confidence_factors:
        print(f"\n🔍 CONFIDENCE FACTORS (top 3):")
        for factor in result.confidence_factors[:3]:
            print(f"     • {factor}")
        if len(result.confidence_factors) > 3:
            print(f"     • ... and {len(result.confidence_factors) - 3} more (use -v for all)")
    
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
        
        # Show best practices summary in verbose mode
        best_practices_summary = analyzer.get_best_practices_summary()
        if best_practices_summary.get("status") == "NVIDIA Best Practices loaded":
            print(f"\n📋 NVIDIA Best Practices Summary:")
            print(f"     Status: {best_practices_summary['status']}")
            print(f"     Scoring: {best_practices_summary['scoring_framework']}")
            print(f"     Source: {best_practices_summary['guidelines_source']}")
    
    if not result.llm_enhanced:
        print(f"\n💡 Tip: Set OPENAI_API_KEY environment variables for enhanced LLM analysis")
    
    print("="*70)


if __name__ == "__main__":
    main()

