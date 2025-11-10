# MCP Setup & Configuration Guide

This guide walks you through setting up the BMCIS Knowledge MCP server with Claude Desktop and other MCP clients. Follow these steps to get started in under 5 minutes.

## Prerequisites

Before setting up the MCP server, ensure you have:

### Required Software

1. **Python 3.11 or higher**
   ```bash
   python --version
   # Should output: Python 3.11.x or higher
   ```

2. **Claude Desktop** (latest version)
   - Download from: https://claude.ai/desktop
   - Ensure you're running version 0.4.0 or higher

3. **PostgreSQL Database** (provided by your administrator)
   - Connection details will be provided separately
   - No local installation needed if using remote database

### Required Access

1. **BMCIS API Key**
   - Contact your system administrator
   - Keep this key secure and never share it

2. **Database Credentials** (if not using default)
   - PostgreSQL connection string
   - Provided by your database administrator

## Quick Start (5 Minutes)

### Step 1: Set Environment Variable

Set your API key as an environment variable:

#### macOS/Linux
```bash
# Add to your shell configuration file
echo 'export BMCIS_API_KEY="your-api-key-here"' >> ~/.bashrc
# Or for Zsh (macOS default)
echo 'export BMCIS_API_KEY="your-api-key-here"' >> ~/.zshrc

# Apply changes
source ~/.bashrc  # or source ~/.zshrc
```

#### Windows (PowerShell)
```powershell
# Set user environment variable permanently
[System.Environment]::SetEnvironmentVariable('BMCIS_API_KEY', 'your-api-key-here', 'User')

# Restart PowerShell or run:
$env:BMCIS_API_KEY = "your-api-key-here"
```

#### Windows (Command Prompt)
```cmd
# Set permanently
setx BMCIS_API_KEY "your-api-key-here"

# Note: You'll need to restart Command Prompt after this
```

### Step 2: Configure Claude Desktop

1. **Locate your Claude configuration directory:**

   - **macOS**: `~/Library/Application Support/Claude/`
   - **Windows**: `%APPDATA%\Claude\` or `C:\Users\[username]\AppData\Roaming\Claude\`
   - **Linux**: `~/.config/Claude/`

2. **Create or edit `.mcp.json` in the configuration directory:**

```json
{
  "mcpServers": {
    "bmcis-knowledge": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/bmcis-knowledge-mcp-local",
      "env": {
        "BMCIS_API_KEY": "your-api-key-here",
        "PYTHONPATH": "/path/to/bmcis-knowledge-mcp-local"
      }
    }
  }
}
```

Replace `/path/to/bmcis-knowledge-mcp-local` with the actual path to your project directory.

### Step 3: Install Python Dependencies

Navigate to the project directory and install requirements:

```bash
cd /path/to/bmcis-knowledge-mcp-local

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Restart Claude Desktop

1. Completely quit Claude Desktop (not just close the window)
2. Start Claude Desktop again
3. The MCP server should initialize automatically

### Step 5: Test the Connection

In Claude, test the tools:

```
Use the semantic_search tool to search for "test query"
```

If successful, you should see search results. If not, check the troubleshooting section below.

## Production Setup

For production environments, follow these additional configuration steps:

### Environment Variables

Create a `.env` file in the project root (never commit this to version control):

```bash
# Required
BMCIS_API_KEY=your-production-api-key

# Database Configuration
DATABASE_URL=postgresql://username:password@host:port/database
DATABASE_MIN_CONNECTIONS=2
DATABASE_MAX_CONNECTIONS=10

# Rate Limiting (optional - defaults shown)
BMCIS_RATE_LIMIT_MINUTE=100
BMCIS_RATE_LIMIT_HOUR=1000
BMCIS_RATE_LIMIT_DAY=10000

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/bmcis-mcp/server.log

# Performance
CACHE_TTL=3600
CACHE_MAX_SIZE=1000
SEARCH_TIMEOUT=30

# Security
API_KEY_HEADER=X-API-Key
ENABLE_CORS=false
ALLOWED_ORIGINS=https://claude.ai
```

### MCP Configuration with All Options

Complete `.mcp.json` with all configuration options:

```json
{
  "mcpServers": {
    "bmcis-knowledge": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/opt/bmcis-knowledge-mcp",
      "env": {
        // API Authentication
        "BMCIS_API_KEY": "${BMCIS_API_KEY}",

        // Database
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/bmcis",
        "DATABASE_MIN_CONNECTIONS": "2",
        "DATABASE_MAX_CONNECTIONS": "10",

        // Rate Limiting
        "BMCIS_RATE_LIMIT_MINUTE": "100",
        "BMCIS_RATE_LIMIT_HOUR": "1000",
        "BMCIS_RATE_LIMIT_DAY": "10000",

        // Logging
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": "json",

        // Python
        "PYTHONPATH": "/opt/bmcis-knowledge-mcp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1"
      },
      "timeout": 60000,
      "restart": true,
      "restartDelay": 5000
    }
  }
}
```

### Using Virtual Environment in Production

Update `.mcp.json` to use virtual environment:

```json
{
  "mcpServers": {
    "bmcis-knowledge": {
      "command": "/opt/bmcis-knowledge-mcp/venv/bin/python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/opt/bmcis-knowledge-mcp"
      // ... rest of configuration
    }
  }
}
```

### Systemd Service (Linux Production)

Create `/etc/systemd/system/bmcis-mcp.service`:

```ini
[Unit]
Description=BMCIS Knowledge MCP Server
After=network.target postgresql.service

[Service]
Type=simple
User=bmcis
Group=bmcis
WorkingDirectory=/opt/bmcis-knowledge-mcp
Environment="PYTHONPATH=/opt/bmcis-knowledge-mcp"
Environment="BMCIS_API_KEY=your-api-key"
ExecStart=/opt/bmcis-knowledge-mcp/venv/bin/python -m src.mcp.server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable bmcis-mcp
sudo systemctl start bmcis-mcp
sudo systemctl status bmcis-mcp
```

## Monitoring and Logging

### Log Locations

Default log locations by platform:

- **macOS**: `~/Library/Logs/Claude/mcp-server.log`
- **Windows**: `%LOCALAPPDATA%\Claude\Logs\mcp-server.log`
- **Linux**: `~/.local/share/Claude/logs/mcp-server.log`

### Viewing Logs

```bash
# Tail logs in real-time
tail -f /path/to/mcp-server.log

# View errors only
grep ERROR /path/to/mcp-server.log

# View last 100 lines
tail -n 100 /path/to/mcp-server.log
```

### Log Levels

Set `LOG_LEVEL` environment variable:

- `DEBUG`: Detailed debugging information
- `INFO`: General operational information (default)
- `WARNING`: Warning messages
- `ERROR`: Error messages only
- `CRITICAL`: Critical issues only

### Health Checks

Test server health with curl:

```bash
# Check if server is responding
curl -X POST http://localhost:8000/health \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"method": "ping"}'
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Tools not appearing in Claude"

**Symptoms:** Semantic search and find_vendor_info tools don't show up in Claude.

**Solutions:**

1. **Verify MCP configuration:**
   ```bash
   # Check if .mcp.json exists
   ls -la ~/Library/Application\ Support/Claude/.mcp.json  # macOS
   dir %APPDATA%\Claude\.mcp.json  # Windows
   ```

2. **Validate JSON syntax:**
   ```bash
   python -m json.tool < .mcp.json
   ```

3. **Check server path:**
   - Ensure `cwd` points to correct directory
   - Verify Python can find the module:
     ```bash
     cd /path/to/bmcis-knowledge-mcp
     python -m src.mcp.server
     ```

4. **Restart Claude completely:**
   - Quit Claude (Cmd+Q on Mac, Alt+F4 on Windows)
   - Wait 5 seconds
   - Start Claude again

#### Issue: "Authentication required" error

**Symptoms:** Tools return authentication error when used.

**Solutions:**

1. **Verify API key is set:**
   ```bash
   echo $BMCIS_API_KEY  # Should show your key (partially)
   ```

2. **Check MCP configuration includes API key:**
   ```json
   "env": {
     "BMCIS_API_KEY": "your-actual-key-here"
   }
   ```

3. **Test API key directly:**
   ```python
   import os
   os.environ['BMCIS_API_KEY'] = 'your-key'
   from src.mcp.auth import validate_api_key
   print(validate_api_key('your-key'))  # Should return True
   ```

#### Issue: "Rate limit exceeded" errors

**Symptoms:** Frequent rate limit errors even with light usage.

**Solutions:**

1. **Check current limits:**
   ```bash
   echo $BMCIS_RATE_LIMIT_MINUTE
   echo $BMCIS_RATE_LIMIT_HOUR
   echo $BMCIS_RATE_LIMIT_DAY
   ```

2. **Increase limits in configuration:**
   ```json
   "env": {
     "BMCIS_RATE_LIMIT_MINUTE": "200",
     "BMCIS_RATE_LIMIT_HOUR": "2000",
     "BMCIS_RATE_LIMIT_DAY": "20000"
   }
   ```

3. **Implement request spacing:**
   - Add delays between requests
   - Use batch operations where possible

#### Issue: "Module not found" errors

**Symptoms:** Python can't find src.mcp.server module.

**Solutions:**

1. **Set PYTHONPATH correctly:**
   ```json
   "env": {
     "PYTHONPATH": "/absolute/path/to/bmcis-knowledge-mcp"
   }
   ```

2. **Verify project structure:**
   ```bash
   ls -la src/mcp/server.py  # Should exist
   ```

3. **Check Python version:**
   ```bash
   python --version  # Should be 3.11+
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

#### Issue: "Database connection failed"

**Symptoms:** Server fails to start or queries fail with database errors.

**Solutions:**

1. **Test database connection:**
   ```python
   import psycopg2
   conn = psycopg2.connect("postgresql://user:pass@host/db")
   print("Connected successfully")
   ```

2. **Check DATABASE_URL format:**
   ```
   postgresql://username:password@hostname:port/database
   ```

3. **Verify network connectivity:**
   ```bash
   ping database-host
   telnet database-host 5432
   ```

4. **Check PostgreSQL logs:**
   - Look for connection attempts
   - Verify authentication settings

#### Issue: "Slow response times"

**Symptoms:** Searches take several seconds to complete.

**Solutions:**

1. **Check database performance:**
   ```sql
   -- In PostgreSQL
   EXPLAIN ANALYZE SELECT * FROM knowledge_chunks LIMIT 10;
   ```

2. **Verify indexes exist:**
   ```sql
   -- Check for vector and text search indexes
   \di knowledge_*
   ```

3. **Monitor connection pool:**
   ```python
   from src.core.database import DatabasePool
   pool = DatabasePool()
   print(f"Active connections: {pool.size}")
   ```

4. **Enable query caching:**
   - Cache is enabled by default
   - Increase cache size if needed

### Debug Mode

Enable detailed debugging:

1. **Set debug environment variables:**
   ```json
   "env": {
     "LOG_LEVEL": "DEBUG",
     "PYTHONDEBUG": "1",
     "MCP_DEBUG": "true"
   }
   ```

2. **View debug output:**
   ```bash
   # Start Claude from terminal to see output
   # macOS
   /Applications/Claude.app/Contents/MacOS/Claude

   # Windows
   "C:\Program Files\Claude\Claude.exe"
   ```

3. **Check MCP communication:**
   - Debug logs show all MCP protocol messages
   - Look for tool registration and invocation

### Performance Optimization

#### Database Optimization

1. **Increase connection pool:**
   ```json
   "env": {
     "DATABASE_MIN_CONNECTIONS": "5",
     "DATABASE_MAX_CONNECTIONS": "20"
   }
   ```

2. **Tune PostgreSQL:**
   ```sql
   -- Increase work memory for better query performance
   ALTER SYSTEM SET work_mem = '256MB';
   ALTER SYSTEM SET shared_buffers = '1GB';
   SELECT pg_reload_conf();
   ```

#### Caching Configuration

1. **Increase cache size:**
   ```json
   "env": {
     "CACHE_MAX_SIZE": "5000",
     "CACHE_TTL": "7200"
   }
   ```

2. **Monitor cache hits:**
   ```python
   from src.search.cache import QueryCache
   cache = QueryCache()
   print(f"Cache hit rate: {cache.hit_rate()}")
   ```

### Getting Help

If you're still experiencing issues:

1. **Check the logs** for error messages
2. **Review the API documentation** for correct usage
3. **Contact support** with:
   - Error messages from logs
   - Your configuration (without API keys)
   - Steps to reproduce the issue
   - System information (OS, Python version, Claude version)

## Security Best Practices

### API Key Management

1. **Never commit API keys to version control:**
   ```bash
   # Add to .gitignore
   echo ".env" >> .gitignore
   echo ".mcp.json" >> .gitignore
   ```

2. **Use environment variables:**
   - Store keys in system environment variables
   - Reference them in configuration files
   - Rotate keys every 90 days

3. **Restrict key permissions:**
   ```bash
   # Unix/Linux: Restrict file permissions
   chmod 600 .env
   chmod 600 .mcp.json
   ```

### Network Security

1. **Use HTTPS for remote databases:**
   ```
   postgresql://user:pass@host:5432/db?sslmode=require
   ```

2. **Restrict database access:**
   - Use IP allowlisting
   - Implement VPN for remote access
   - Use connection pooling with limits

3. **Monitor access logs:**
   - Review authentication failures
   - Track unusual usage patterns
   - Set up alerting for anomalies

### Audit and Compliance

1. **Enable audit logging:**
   ```json
   "env": {
     "AUDIT_LOG": "true",
     "AUDIT_LOG_FILE": "/var/log/bmcis-audit.log"
   }
   ```

2. **Regular security reviews:**
   - Review access logs monthly
   - Rotate API keys quarterly
   - Update dependencies regularly

3. **Data protection:**
   - Ensure database encryption at rest
   - Use encrypted connections
   - Implement data retention policies

## Next Steps

After successful setup:

1. **Read the API Reference**: Review [mcp-tools.md](../api-reference/mcp-tools.md) for detailed tool documentation
2. **Explore Examples**: Check [mcp-usage-examples.md](./mcp-usage-examples.md) for practical workflows
3. **Optimize Performance**: Configure caching and connection pooling for your workload
4. **Set Up Monitoring**: Implement logging and alerting for production use
5. **Plan Scaling**: Consider load balancing for high-traffic scenarios

---

*For additional support, contact your system administrator or refer to the complete documentation.*