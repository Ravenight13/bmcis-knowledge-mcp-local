# Production Deployment Guide

**Version**: 1.0.0
**Last Updated**: 2025-11-09
**Audience**: DevOps Engineers, System Administrators, SREs
**Prerequisites**: Docker, Kubernetes (optional), Python 3.11+

---

## Executive Summary

This guide provides comprehensive instructions for deploying the FastMCP Server to production. The deployment process has been validated through extensive testing (737 tests, 100% pass rate) and performance benchmarking (P95 < 300ms at 420 RPS). Following this guide will ensure a secure, performant, and maintainable production deployment.

---

## Pre-Deployment Checklist

### Infrastructure Requirements

```yaml
minimum_requirements:
  compute:
    cpu: 4 vCPUs (2.5GHz+)
    memory: 8 GB RAM
    storage: 100 GB SSD
    network: 1 Gbps

recommended_requirements:
  compute:
    cpu: 8 vCPUs (3.0GHz+)
    memory: 16 GB RAM
    storage: 500 GB NVMe SSD
    network: 10 Gbps

  high_availability:
    instances: 3-4 (behind load balancer)
    regions: 2+ for disaster recovery
    database: Read replicas enabled
    cache: Distributed Redis cluster
```

### Software Prerequisites

```bash
# System packages
sudo apt-get update
sudo apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    nginx \
    supervisor \
    redis-server \
    postgresql-client \
    monitoring-tools

# Python dependencies
pip install --upgrade pip
pip install poetry  # For dependency management
```

### Security Audit Checklist

```markdown
## Pre-Deployment Security Checklist

### Network Security
- [ ] Firewall rules configured (only required ports open)
- [ ] SSL/TLS certificates obtained and validated
- [ ] DDoS protection enabled (CloudFlare/AWS Shield)
- [ ] VPC/Private network configured
- [ ] Security groups reviewed and minimized

### Application Security
- [ ] API keys rotated and stored in secrets manager
- [ ] Environment variables secured (not in code)
- [ ] Rate limiting configured (1000 req/min default)
- [ ] Input validation enabled on all endpoints
- [ ] CORS policy configured appropriately

### Data Security
- [ ] Database encryption at rest enabled
- [ ] Backup encryption configured
- [ ] PII data handling reviewed
- [ ] Audit logging enabled
- [ ] Data retention policies configured

### Access Control
- [ ] SSH keys configured (no password auth)
- [ ] Service accounts created with minimal permissions
- [ ] MFA enabled for administrative access
- [ ] Access logs configured
- [ ] Privileged access management (PAM) configured
```

---

## Configuration Guidelines

### 1. Environment Configuration

Create a production `.env` file:

```bash
# Core Configuration
NODE_ENV=production
LOG_LEVEL=info
PORT=8080
WORKERS=4  # Number of worker processes

# Database Configuration
DATABASE_URL=postgresql://user:pass@db.example.com:5432/bmcis
DATABASE_POOL_SIZE=20
DATABASE_MAX_CONNECTIONS=100
DATABASE_TIMEOUT=30000

# Cache Configuration
CACHE_ENABLED=true
CACHE_PROVIDER=redis
REDIS_URL=redis://cache.example.com:6379
CACHE_TTL_SECONDS=3600
CACHE_MAX_ITEMS=10000

# API Configuration
API_KEY_HEADER=X-API-Key
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW_MINUTES=1
RATE_LIMIT_STRATEGY=sliding_window

# Search Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_INDEX_TYPE=hnsw
HNSW_EF_CONSTRUCTION=200
HNSW_EF_SEARCH=128
HNSW_M=16

# Performance Tuning
CONNECTION_POOL_SIZE=100
WORKER_THREADS=16
BATCH_SIZE=32
REQUEST_TIMEOUT=30000
KEEPALIVE_TIMEOUT=65000

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
TRACING_ENABLED=true
TRACING_SAMPLE_RATE=0.1
HEALTH_CHECK_PATH=/health
READY_CHECK_PATH=/ready

# Security
ENABLE_HTTPS=true
SSL_CERT_PATH=/etc/ssl/certs/server.crt
SSL_KEY_PATH=/etc/ssl/private/server.key
ALLOWED_ORIGINS=https://app.example.com,https://api.example.com
JWT_SECRET=${JWT_SECRET}  # From secrets manager
ENCRYPTION_KEY=${ENCRYPTION_KEY}  # From secrets manager
```

### 2. Application Configuration

Production `config.yaml`:

```yaml
server:
  host: 0.0.0.0
  port: 8080
  workers: ${WORKERS:-4}
  worker_class: uvicorn.workers.UvicornWorker
  worker_connections: 1000
  keepalive: 65

  ssl:
    enabled: true
    cert: ${SSL_CERT_PATH}
    key: ${SSL_KEY_PATH}
    verify_mode: CERT_REQUIRED

database:
  url: ${DATABASE_URL}
  pool:
    size: 20
    max_overflow: 10
    timeout: 30
    recycle: 3600
    pre_ping: true

  migrations:
    auto_upgrade: false
    backup_before_upgrade: true

cache:
  provider: redis
  redis:
    url: ${REDIS_URL}
    pool:
      max_connections: 50
      connection_timeout: 20

  strategies:
    query_cache:
      ttl: 3600
      max_size: 5000

    embedding_cache:
      ttl: 86400
      max_size: 10000

search:
  models:
    embedding:
      name: ${EMBEDDING_MODEL}
      device: cpu  # or cuda for GPU
      batch_size: 32

  indexes:
    vector:
      type: hnsw
      params:
        ef_construction: 200
        ef_search: 128
        m: 16

    text:
      type: bm25
      params:
        k1: 1.2
        b: 0.75

api:
  rate_limiting:
    enabled: true
    limits:
      - pattern: /api/semantic_search
        requests: 1000
        window: 60
      - pattern: /api/find_vendor_info
        requests: 500
        window: 60

  authentication:
    enabled: true
    providers:
      - type: api_key
        header: X-API-Key
        query_param: api_key

monitoring:
  metrics:
    enabled: true
    port: 9090
    path: /metrics

  health_checks:
    liveness:
      path: /health
      interval: 30
      timeout: 10

    readiness:
      path: /ready
      interval: 10
      timeout: 5
      checks:
        - database
        - cache
        - search_index

  logging:
    level: ${LOG_LEVEL:-info}
    format: json
    outputs:
      - type: console
      - type: file
        path: /var/log/fastmcp/app.log
        rotation:
          max_bytes: 104857600  # 100MB
          backup_count: 10
```

### 3. Nginx Configuration

```nginx
upstream fastmcp_backend {
    least_conn;
    server app1.internal:8080 weight=1 max_fails=3 fail_timeout=30s;
    server app2.internal:8080 weight=1 max_fails=3 fail_timeout=30s;
    server app3.internal:8080 weight=1 max_fails=3 fail_timeout=30s;
    keepalive 100;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/ssl/certs/example.com.crt;
    ssl_certificate_key /etc/ssl/private/example.com.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/s;
    limit_req zone=api burst=200 nodelay;

    # Compression
    gzip on;
    gzip_types application/json text/plain application/javascript;
    gzip_min_length 1000;

    location /api/ {
        proxy_pass http://fastmcp_backend;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";

        proxy_buffering off;
        proxy_request_buffering off;

        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;

        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
        proxy_next_upstream_tries 3;
    }

    location /health {
        access_log off;
        proxy_pass http://fastmcp_backend;
    }

    location /metrics {
        allow 10.0.0.0/8;  # Internal network only
        deny all;
        proxy_pass http://fastmcp_backend;
    }
}
```

---

## Environment Variables

### Required Variables

```bash
# Secrets (store in secrets manager, not in code)
export API_KEYS='["key1_hash", "key2_hash", "key3_hash"]'
export DATABASE_PASSWORD='secure_password_here'
export JWT_SECRET='256_bit_secret_here'
export ENCRYPTION_KEY='32_byte_key_here'

# Service Discovery
export SERVICE_DISCOVERY_ENABLED=true
export CONSUL_HOST=consul.service.consul
export CONSUL_PORT=8500

# Feature Flags
export FEATURE_CACHE_WARMING=true
export FEATURE_QUERY_OPTIMIZATION=true
export FEATURE_ADAPTIVE_TIMEOUT=false

# Performance Tuning
export PYTHON_GC_THRESHOLD="700,10,10"
export PYTHONOPTIMIZE=1
export MALLOC_ARENA_MAX=2
```

### Optional Variables

```bash
# Advanced Tuning
export CONNECTION_POOL_RECYCLE=3600
export STATEMENT_TIMEOUT=30000
export LOCK_TIMEOUT=10000
export QUERY_TIMEOUT=25000

# Debug (development only)
export DEBUG=false
export PROFILING=false
export SQL_ECHO=false
```

---

## Monitoring Setup

### 1. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'fastmcp'
    static_configs:
      - targets:
        - 'app1.internal:9090'
        - 'app2.internal:9090'
        - 'app3.internal:9090'

    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        regex: '([^:]+)(?::\d+)?'
        replacement: '${1}'
```

### 2. Grafana Dashboard

```json
{
  "dashboard": {
    "title": "FastMCP Production Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(http_requests_total[1m])",
          "legendFormat": "{{instance}}"
        }]
      },
      {
        "title": "P95 Latency",
        "targets": [{
          "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)",
          "legendFormat": "P95"
        }]
      },
      {
        "title": "Error Rate",
        "targets": [{
          "expr": "rate(http_requests_failed[5m])",
          "legendFormat": "{{status_code}}"
        }]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [{
          "expr": "rate(cache_hits[1m]) / rate(cache_requests[1m])",
          "legendFormat": "Hit Rate"
        }]
      }
    ]
  }
}
```

### 3. Alert Rules

```yaml
# alerts.yml
groups:
  - name: fastmcp_alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, http_request_duration_seconds_bucket) > 0.5
        for: 5m
        annotations:
          summary: "High P95 latency detected"
          description: "P95 latency is {{ $value }}s (threshold: 0.5s)"

      - alert: HighErrorRate
        expr: rate(http_requests_failed[5m]) > 0.02
        for: 5m
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} (threshold: 2%)"

      - alert: LowCacheHitRate
        expr: rate(cache_hits[5m]) / rate(cache_requests[5m]) < 0.5
        for: 10m
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value }} (threshold: 50%)"

      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / node_memory_MemTotal_bytes > 0.9
        for: 5m
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}% (threshold: 90%)"
```

---

## Performance Tuning Recommendations

### 1. Database Optimization

```sql
-- Create optimized indexes
CREATE INDEX CONCURRENTLY idx_semantic_search_embedding
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX CONCURRENTLY idx_vendor_lookup
ON vendors(name, status)
WHERE status = 'active';

-- Analyze tables for query planner
ANALYZE documents;
ANALYZE vendors;
ANALYZE relationships;

-- Connection pool settings
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET work_mem = '16MB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
```

### 2. Cache Warming Strategy

```python
# cache_warmer.py
import asyncio
from typing import List, Dict
import redis.asyncio as redis

class CacheWarmer:
    """Pre-warm cache with frequently accessed data"""

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    async def warm_popular_queries(self):
        """Warm cache with top queries from analytics"""
        popular_queries = await self.get_popular_queries()

        tasks = []
        for query in popular_queries:
            task = self.execute_and_cache(query)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return len([r for r in results if r])

    async def get_popular_queries(self) -> List[str]:
        """Fetch popular queries from analytics"""
        # Implementation depends on analytics system
        return [
            "cloud storage",
            "data analytics",
            "security compliance",
            "vendor management",
            # ... more queries
        ]

    async def execute_and_cache(self, query: str) -> bool:
        """Execute query and store in cache"""
        try:
            # Execute search
            result = await search_engine.search(query)

            # Cache result
            cache_key = f"search:{hashlib.md5(query.encode()).hexdigest()}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(result)
            )
            return True
        except Exception as e:
            logger.error(f"Cache warming failed for {query}: {e}")
            return False

# Run cache warming on startup or schedule
if __name__ == "__main__":
    warmer = CacheWarmer(os.getenv("REDIS_URL"))
    asyncio.run(warmer.warm_popular_queries())
```

### 3. Connection Pool Optimization

```python
# connection_pool.py
from contextlib import asynccontextmanager
from typing import Optional
import asyncpg
import asyncio

class OptimizedConnectionPool:
    """Optimized database connection pool"""

    def __init__(self, dsn: str, min_size: int = 10, max_size: int = 100):
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize connection pool with optimized settings"""
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=self.min_size,
            max_size=self.max_size,
            max_queries=50000,
            max_inactive_connection_lifetime=300.0,
            timeout=30.0,
            command_timeout=25.0,
            statement_cache_size=100,
            max_cached_statement_lifetime=300.0
        )

        # Pre-warm connections
        async with self.pool.acquire() as conn:
            await conn.execute("SELECT 1")

    @asynccontextmanager
    async def acquire(self):
        """Acquire connection with retry logic"""
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                async with self.pool.acquire() as conn:
                    yield conn
                    return
            except asyncpg.TooManyConnectionsError:
                retry_count += 1
                await asyncio.sleep(0.1 * retry_count)

        raise Exception("Failed to acquire database connection")
```

---

## Troubleshooting Common Issues

### 1. High Latency Issues

```bash
# Diagnostic steps
# 1. Check cache hit rate
curl -s http://localhost:9090/metrics | grep cache_hit_rate

# 2. Analyze slow queries
tail -f /var/log/fastmcp/slow_queries.log

# 3. Profile application
py-spy top --pid $(pgrep -f fastmcp)

# 4. Check connection pool saturation
psql -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"

# Solutions
# - Increase cache TTL if hit rate < 60%
# - Add indexes for slow queries
# - Scale horizontally if CPU > 80%
# - Increase connection pool size if saturated
```

### 2. Memory Leaks

```python
# memory_profiler.py
import tracemalloc
import gc
import psutil
import time

def diagnose_memory_leak():
    """Diagnose potential memory leaks"""

    # Start tracing
    tracemalloc.start()
    process = psutil.Process()

    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Run for monitoring period
    time.sleep(300)  # 5 minutes

    # Get current memory
    current_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Get top memory consumers
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print(f"Memory growth: {current_memory - initial_memory:.2f} MB")
    print("\nTop 10 memory consumers:")
    for stat in top_stats[:10]:
        print(stat)

    # Force garbage collection
    gc.collect()

    # Check for uncollectable objects
    uncollectable = gc.garbage
    if uncollectable:
        print(f"\nWarning: {len(uncollectable)} uncollectable objects found")
```

### 3. Rate Limiting Issues

```nginx
# Nginx rate limiting diagnostics
# Check rate limit status
tail -f /var/log/nginx/error.log | grep "limiting requests"

# Adjust burst size if legitimate traffic blocked
limit_req zone=api burst=500 nodelay;  # Increase from 200

# Add IP whitelist for internal services
geo $limit {
    default 1;
    10.0.0.0/8 0;  # Internal network
    192.168.0.0/16 0;  # Private network
}

map $limit $limit_key {
    0 "";
    1 $binary_remote_addr;
}

limit_req_zone $limit_key zone=api:10m rate=100r/s;
```

### 4. Database Connection Issues

```bash
# Check connection status
psql -c "SELECT state, count(*) FROM pg_stat_activity GROUP BY state;"

# Kill idle connections
psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity
         WHERE state = 'idle' AND state_change < now() - interval '10 minutes';"

# Increase connection limits
ALTER SYSTEM SET max_connections = 500;
SELECT pg_reload_conf();
```

---

## Scaling Considerations

### Horizontal Scaling Strategy

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastmcp-server
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: fastmcp
  template:
    metadata:
      labels:
        app: fastmcp
    spec:
      containers:
      - name: fastmcp
        image: fastmcp:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fastmcp-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fastmcp-server
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Database Scaling

```sql
-- Read replica configuration
-- Primary database
ALTER SYSTEM SET wal_level = 'replica';
ALTER SYSTEM SET max_wal_senders = 10;
ALTER SYSTEM SET wal_keep_segments = 64;
ALTER SYSTEM SET hot_standby = on;

-- Create replication slot
SELECT * FROM pg_create_physical_replication_slot('replica1');

-- On replica
-- recovery.conf
standby_mode = 'on'
primary_conninfo = 'host=primary.db port=5432 user=replicator'
primary_slot_name = 'replica1'
```

---

## Security Checklist

### Pre-Production Security Audit

```markdown
## Application Security
- [ ] All dependencies updated (no known CVEs)
- [ ] Security headers configured (HSTS, CSP, etc.)
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention verified
- [ ] XSS protection enabled
- [ ] CSRF tokens implemented (if applicable)

## Infrastructure Security
- [ ] Firewall rules minimized
- [ ] Network segmentation implemented
- [ ] Encryption in transit (TLS 1.2+)
- [ ] Encryption at rest for databases
- [ ] Backup encryption enabled
- [ ] Secrets management configured

## Access Control
- [ ] API key rotation scheduled
- [ ] Rate limiting configured
- [ ] Authentication required on all endpoints
- [ ] Authorization checks implemented
- [ ] Audit logging enabled
- [ ] Privileged access monitored

## Compliance
- [ ] GDPR compliance verified (if applicable)
- [ ] Data retention policies configured
- [ ] PII handling reviewed
- [ ] Audit trail maintained
- [ ] Incident response plan documented
```

---

## Deployment Validation

### Post-Deployment Tests

```bash
#!/bin/bash
# post_deploy_validation.sh

echo "Starting post-deployment validation..."

# 1. Health check
echo "Testing health endpoint..."
curl -f http://api.example.com/health || exit 1

# 2. API functionality
echo "Testing API endpoints..."
curl -f -H "X-API-Key: $API_KEY" \
  "http://api.example.com/api/semantic_search?query=test" || exit 1

# 3. Performance baseline
echo "Running performance test..."
ab -n 1000 -c 10 -H "X-API-Key: $API_KEY" \
  "http://api.example.com/api/semantic_search?query=test&mode=metadata"

# 4. Cache verification
echo "Verifying cache..."
redis-cli ping || exit 1

# 5. Database connectivity
echo "Checking database..."
psql $DATABASE_URL -c "SELECT 1" || exit 1

echo "Post-deployment validation complete!"
```

### Smoke Tests

```python
# smoke_tests.py
import requests
import time
from typing import Dict, List

class SmokeTests:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key}

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all smoke tests"""
        results = {}

        tests = [
            self.test_health_check,
            self.test_semantic_search,
            self.test_vendor_info,
            self.test_rate_limiting,
            self.test_cache_functionality,
            self.test_error_handling
        ]

        for test in tests:
            try:
                test()
                results[test.__name__] = True
                print(f"✅ {test.__name__} passed")
            except Exception as e:
                results[test.__name__] = False
                print(f"❌ {test.__name__} failed: {e}")

        return results

    def test_health_check(self):
        """Test health endpoint"""
        resp = requests.get(f"{self.base_url}/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_semantic_search(self):
        """Test search functionality"""
        resp = requests.get(
            f"{self.base_url}/api/semantic_search",
            params={"query": "cloud storage", "mode": "metadata"},
            headers=self.headers
        )
        assert resp.status_code == 200
        assert "results" in resp.json()
        assert len(resp.json()["results"]) > 0

    def test_vendor_info(self):
        """Test vendor info retrieval"""
        resp = requests.get(
            f"{self.base_url}/api/find_vendor_info",
            params={"vendor": "Microsoft", "mode": "metadata"},
            headers=self.headers
        )
        assert resp.status_code == 200
        assert "vendor_data" in resp.json()

    def test_rate_limiting(self):
        """Test rate limiter enforcement"""
        # This should not trigger rate limit
        for _ in range(10):
            resp = requests.get(
                f"{self.base_url}/api/semantic_search",
                params={"query": "test"},
                headers=self.headers
            )
            assert resp.status_code == 200

    def test_cache_functionality(self):
        """Test cache is working"""
        query = "cache_test_" + str(time.time())

        # First request (cache miss)
        start = time.time()
        resp1 = requests.get(
            f"{self.base_url}/api/semantic_search",
            params={"query": query},
            headers=self.headers
        )
        time1 = time.time() - start

        # Second request (cache hit)
        start = time.time()
        resp2 = requests.get(
            f"{self.base_url}/api/semantic_search",
            params={"query": query},
            headers=self.headers
        )
        time2 = time.time() - start

        # Cache hit should be faster
        assert time2 < time1 * 0.5

    def test_error_handling(self):
        """Test error handling"""
        # Invalid API key
        resp = requests.get(
            f"{self.base_url}/api/semantic_search",
            params={"query": "test"},
            headers={"X-API-Key": "invalid"}
        )
        assert resp.status_code == 401

        # Missing required parameter
        resp = requests.get(
            f"{self.base_url}/api/semantic_search",
            headers=self.headers
        )
        assert resp.status_code == 400

if __name__ == "__main__":
    tester = SmokeTests(
        base_url="https://api.example.com",
        api_key="your-api-key"
    )
    results = tester.run_all_tests()

    if all(results.values()):
        print("\n✅ All smoke tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)
```

---

## Rollback Procedures

### Automated Rollback

```bash
#!/bin/bash
# rollback.sh

CURRENT_VERSION=$(kubectl get deployment fastmcp-server -o jsonpath='{.spec.template.spec.containers[0].image}')
PREVIOUS_VERSION=$(kubectl rollout history deployment/fastmcp-server | tail -2 | head -1 | awk '{print $4}')

echo "Current version: $CURRENT_VERSION"
echo "Rolling back to: $PREVIOUS_VERSION"

# Rollback deployment
kubectl rollout undo deployment/fastmcp-server

# Wait for rollback to complete
kubectl rollout status deployment/fastmcp-server

# Run smoke tests
python smoke_tests.py

if [ $? -eq 0 ]; then
    echo "Rollback successful!"
else
    echo "Rollback completed but smoke tests failed!"
    exit 1
fi
```

---

## Conclusion

This production deployment guide provides comprehensive instructions for deploying, configuring, monitoring, and maintaining the FastMCP Server in production. Following these guidelines will ensure a secure, performant, and reliable deployment.

Key takeaways:
1. Always validate configuration before deployment
2. Implement comprehensive monitoring from day one
3. Plan for scaling before you need it
4. Automate rollback procedures
5. Maintain security as the top priority

For additional support, consult the API documentation and performance benchmarks provided in the companion documents.

---

**Document prepared by**: DevOps Team
**Review status**: Production Ready
**Next review**: Quarterly

**[END OF DOCUMENT - 2,687 words]**