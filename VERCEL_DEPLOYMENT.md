# Vercel Deployment Guide

## Overview

This Flask application **can be deployed to Vercel**, but with important limitations due to the serverless nature of Vercel's platform.

## ⚠️ Important Limitations

### **Execution Time**
- **Hobby Plan**: 10 seconds max
- **Pro Plan**: 60 seconds max
- **Issue**: Complex notebook analysis may exceed these limits

### **Memory Usage**
- Large notebooks may hit memory limits
- LLM analysis adds processing overhead

### **File Processing**
- Uses `/tmp` directory (max 500MB)
- Files are cleaned up after each request

## 🚀 Deployment Steps

### 1. **Prepare Your Project**

Ensure you have the modified files:
- `vercel.json` - Vercel configuration
- `api/app.py` - Serverless-optimized Flask app
- `requirements.txt` - Dependencies

### 2. **Install Vercel CLI**

```bash
npm i -g vercel
```

### 3. **Configure Environment Variables**

Set up your environment variables in Vercel:

```bash
# Set environment variables
vercel env add OPENAI_BASE_URL
vercel env add OPENAI_API_KEY
vercel env add OPENAI_MODEL
vercel env add GITHUB_TOKEN
vercel env add GITLAB_TOKEN
```

Or use the [Vercel Dashboard](https://vercel.com/dashboard) to set them.

### 4. **Deploy**

```bash
# Deploy to Vercel
vercel --prod
```

### 5. **Test the Deployment**

```bash
# Test health endpoint
curl https://your-app.vercel.app/health

# Test analysis endpoint
curl -X POST https://your-app.vercel.app/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/user/repo/blob/main/simple-notebook.ipynb"}'
```

## 📁 Required File Structure

```
notebook-analyzer/
├── api/
│   └── app.py                    # Vercel-optimized Flask app
├── templates/                    # HTML templates (unchanged)
├── notebook-analyzer.py          # Core analyzer (unchanged)
├── requirements.txt              # Python dependencies
├── vercel.json                   # Vercel configuration
└── VERCEL_DEPLOYMENT.md          # This file
```

## ⚙️ Configuration

### `vercel.json`
```json
{
  "functions": {
    "api/app.py": {
      "maxDuration": 60
    }
  },
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/api/app"
    }
  ]
}
```

### Environment Variables
Set these in your Vercel project settings:

- `OPENAI_BASE_URL` - LLM API endpoint (without /v1)
- `OPENAI_API_KEY` - LLM API key
- `OPENAI_MODEL` - Model name (optional)
- `GITHUB_TOKEN` - For private repos (optional)
- `GITLAB_TOKEN` - For private repos (optional)

## 🔧 Troubleshooting

### **Function Timeout**
If you get timeout errors:
1. Upgrade to Pro plan (60s timeout)
2. Test with simpler notebooks first
3. Consider pre-processing large notebooks

### **Memory Issues**
If analysis fails due to memory:
1. The notebook might be too large
2. Try analyzing smaller sections
3. Consider alternative deployment (see below)

### **Import Issues**
If you get import errors:
1. Ensure `notebook-analyzer.py` is in the root directory
2. Check that all dependencies are in `requirements.txt`

## 🚨 **When NOT to Use Vercel**

Consider alternatives if you need:

1. **Long-running analysis** (>60 seconds)
2. **Large file processing** (>16MB notebooks)
3. **Persistent storage** between requests
4. **Complex multi-step workflows**

## 🔄 **Recommended Alternatives**

### **For Heavy Workloads:**

1. **Railway** - Better for long-running processes
   ```bash
   # Deploy to Railway
   railway login
   railway init
   railway up
   ```

2. **Render** - Good for background processing
   ```bash
   # Deploy to Render
   # Use render.yaml configuration
   ```

3. **Google Cloud Run** - Flexible container deployment
   ```bash
   # Deploy to Cloud Run
   gcloud run deploy notebook-analyzer \
     --source . \
     --platform managed \
     --timeout 900
   ```

4. **AWS Lambda** (with increased timeout)
   ```bash
   # Use serverless framework
   serverless deploy
   ```

### **Hybrid Approach:**
- Use Vercel for simple/quick analysis
- Use background service for complex notebooks
- Queue system for batch processing

## 📊 **Performance Comparison**

| Platform | Timeout | Memory | Cost | Best For |
|----------|---------|--------|------|----------|
| Vercel | 10s/60s | ~1GB | $20/mo | Simple analysis |
| Railway | 30min | 8GB | $5/mo | Medium complexity |
| Cloud Run | 60min | 32GB | Pay-per-use | Heavy workloads |
| Docker | Unlimited | Unlimited | $5-50/mo | Full control |

## 💡 **Optimization Tips**

1. **Cache Analysis Results** - Store results in edge config
2. **Stream Responses** - Use Server-Sent Events for progress
3. **Pre-process** - Analyze notebooks offline for complex cases
4. **Batch Processing** - Queue multiple notebooks
5. **CDN Assets** - Use Vercel's CDN for static files

## 🎯 **Conclusion**

Vercel deployment works well for:
- ✅ Simple notebook analysis
- ✅ API-only usage
- ✅ Quick prototyping
- ✅ Low-traffic applications

Consider alternatives for:
- ❌ Complex/large notebooks
- ❌ Long-running analysis
- ❌ High-traffic production use
- ❌ Background processing needs

The modified Flask app is ready for Vercel deployment with the limitations noted above. 