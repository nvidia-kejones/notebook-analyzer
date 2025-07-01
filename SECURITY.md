# Security Guidelines

## 🔒 Secrets and Sensitive Data Protection

This repository is configured to protect sensitive information from being accidentally committed to version control. Please follow these guidelines to maintain security.

### ⚠️ NEVER COMMIT THESE FILES:

#### Environment Files
- `.env` - Contains API keys, tokens, and secrets
- `.env.local`, `.env.production`, etc. - Environment-specific configurations
- Any file containing API keys or passwords

#### API Keys and Tokens
- `*api*key*` - Any file with "apikey" in the name
- `*token*` - Authentication tokens
- `*secret*` - Secret keys or passwords
- `*credentials*` - Credential files
- SSL certificates (`.pem`, `.key`, `.crt` files)

#### Application Data
- `uploads/` - User-uploaded notebook files
- `logs/` - Application log files
- `sessions/` - Session data
- Database files (`.db`, `.sqlite`)

### ✅ SAFE TO COMMIT:

#### Configuration Templates
- `.env.example` - Template showing required environment variables (no real values)
- `mcp_config.json` - Example MCP configuration (no real credentials)
- `requirements.txt` - Python dependencies
- All source code files (`.py`, `.html`, `.js`)

### 🛡️ Security Best Practices

#### 1. Environment Variables
Use environment variables for all sensitive configuration:

```bash
# ✅ GOOD - Use environment variables
export OPENAI_API_KEY="your-secret-key"
export GITHUB_TOKEN="ghp_your_token"

# ❌ BAD - Never hardcode in source code
api_key = "sk-1234567890abcdef"  # DON'T DO THIS!
```

#### 2. Local Configuration
Create local configuration files that are gitignored:

```bash
# Copy the example file
cp .env.example .env

# Edit with your real credentials (this file is gitignored)
nano .env
```

#### 3. Docker Secrets
For production deployments, use Docker secrets or environment files:

```yaml
# docker-compose.yml
services:
  app:
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_key
    secrets:
      - openai_key

secrets:
  openai_key:
    external: true
```

#### 4. API Key Rotation
- Regularly rotate API keys and tokens
- Use environment-specific keys (dev/staging/prod)
- Monitor for exposed credentials using tools like `git-secrets`

### 🔍 Before Committing

Always check for sensitive data before committing:

```bash
# Check what files will be committed
git status

# Review changes in detail
git diff --cached

# Verify no secrets are staged
git log --oneline -n 5
```

### 🚨 If You Accidentally Commit Secrets

1. **Immediately rotate the exposed credentials**
2. **Remove from git history:**
   ```bash
   # Remove file from history (DANGER: rewrites history)
   git filter-branch --force --index-filter \
     'git rm --cached --ignore-unmatch path/to/secret/file' \
     --prune-empty --tag-name-filter cat -- --all
   ```
3. **Force push (if you have write access):**
   ```bash
   git push origin --force --all
   ```
4. **Report the incident if it's a shared repository**

### 📋 Security Checklist

Before making your repository public:

- [ ] All API keys are in environment variables or `.env` (gitignored)
- [ ] No hardcoded passwords or tokens in source code
- [ ] `.env.example` contains template values only
- [ ] All sensitive files are listed in `.gitignore`
- [ ] Test with `git log --all --full-history -- .env` (should be empty)
- [ ] Review git history for any accidentally committed secrets

### 🔗 Additional Resources

- [GitHub: Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [Git Secrets Tool](https://github.com/awslabs/git-secrets)
- [OWASP Secrets Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)

### 🆘 Reporting Security Issues

If you discover a security vulnerability, please:

1. **DO NOT** open a public issue
2. Email the maintainers privately
3. Include details about the vulnerability
4. Allow time for the issue to be addressed before public disclosure

---

**Remember: It's much easier to prevent secrets from being committed than to remove them from git history after the fact!** 