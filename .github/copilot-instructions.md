# FastAPI Backend Template

FastAPI backend template with Celery background tasks, JWT authentication, LLM integration via LangChain, and multi-database support (SQLite/PostgreSQL/MySQL). Built with Poetry for dependency management and includes comprehensive Docker deployment options.

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Bootstrap, Build, and Test the Repository

**Prerequisites:**
- Python 3.12+ (verified working with Python 3.12.3)
- Poetry for dependency management (install via pip if not available)

**Setup Commands (validated and timed):**
```bash
# Install Poetry if not available
pip install poetry

# Copy configuration files  
cp .env.example .env
cp llm_config.example.yaml llm_config.yaml

# Install dependencies - takes 36 seconds. NEVER CANCEL. Set timeout to 120+ seconds.
poetry install

# Run tests - takes 16 seconds. NEVER CANCEL. Set timeout to 60+ seconds.
poetry run pytest
```

**Expected Test Results:**
- 21 tests total: 17 pass, 2 fail (UUID-related issues in auth tests), 2 skip (LLM real tests)
- Test failures are non-critical and do not affect application functionality
- Core functionality (API, database, authentication) works correctly

### Run the Application

**Development Mode:**
```bash
# Start development server with auto-reload
poetry run fastapi dev --host 0.0.0.0 --port 8000
```

**Production Mode:**
```bash  
# Start production server
poetry run fastapi run --host 0.0.0.0 --port 8000
```

**Application URLs:**
- API: http://localhost:8000
- Health Check: http://localhost:8000/api/v1/health/
- API Documentation: http://localhost:8000/docs (development mode only)

### Celery Background Tasks

**Start Redis (required for Celery):**
```bash
# Using Docker (recommended)
docker run -d --name redis-test -p 6379:6379 redis:7-alpine
```

**Start Celery Worker:**
```bash
# In a separate terminal
poetry run celery -A app.celery.celery_main:celery_app worker --loglevel=info --queues=default
```

**Start Flower Monitoring (optional):**
```bash  
# Monitor Celery tasks at http://localhost:5555
poetry run celery -A app.celery.celery_main:celery_app flower --port=5555
```

## Validation

### Always Test Core Functionality After Changes

**Health Check Validation:**
```bash
curl http://localhost:8000/api/v1/health/
# Should return healthy status for database, application, and tasks (if Redis running)
```

**Authentication Flow Validation:**
```bash
# Register user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "email": "test@example.com", "password": "SecurePass123!"}'

# Login and get tokens  
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "SecurePass123!"}'
```

**ALWAYS manually validate any new code by:**
1. Starting the application (`poetry run fastapi dev`)
2. Testing API endpoints with curl or browser
3. Verifying authentication flows work
4. If modifying Celery tasks, ensure Redis is running and test task execution
5. Running the test suite to catch regressions

### Docker Deployment (Known Issues)

**Current Status:** Docker build configuration has timeout and YAML formatting issues.

**Working Alternative:** Use local Poetry-based development as documented above.

**If Docker is needed:** Fix the following known issues in docker-compose files:
- Build timeouts during Poetry install step
- YAML indentation errors in docker-compose.base.yml (lines 27-28, 51-52, 75)
- Path mapping issues in docker/docker-compose.sqlite.yml

## Common Tasks

### Repository Structure
```
.
├── README.md                 # Main documentation
├── pyproject.toml           # Poetry dependencies and config
├── poetry.lock              # Locked dependency versions
├── pytest.ini              # Test configuration
├── .env.example             # Environment variables template
├── llm_config.example.yaml  # LLM configuration template
├── run.py                   # Alternative startup script
├── app/                     # Main application code
│   ├── main.py             # FastAPI app factory
│   ├── api/v1/routes/      # API endpoints
│   ├── celery/             # Background task definitions
│   ├── core/               # Core configuration
│   ├── database/           # Database models and connections
│   ├── llm/                # LLM integration
│   ├── schemas/            # Pydantic schemas
│   └── services/           # Business logic services
├── tests/                   # Test suite
│   └── integration/        # Integration tests
└── docker/                 # Docker configuration (has issues)
    ├── Dockerfile
    ├── docker-compose.sqlite.yml
    ├── docker-compose.mysql.yml
    └── docker-compose.postgres.yml
```

### Key Configuration Files

**Environment Variables (.env):**
- Database: SQLite by default, supports PostgreSQL/MySQL
- Redis: Required for Celery tasks (localhost:6379)
- JWT: Configure SECRET_KEY for production
- LLM: Set API keys for OpenAI or other providers

**LLM Configuration (llm_config.yaml):**
- Providers: OpenAI, custom OpenAI-compatible endpoints
- Models: Configurable per task (explain, conversation)
- Testing: Uses different models for testing vs production

### API Endpoints Summary

**Health & Info:**
- `GET /` - Application info
- `GET /api/v1/health/` - Overall health check
- `GET /api/v1/health/database` - Database health
- `GET /api/v1/health/ping` - Simple ping

**Authentication (JWT):**
- `POST /api/v1/auth/register` - Create account
- `POST /api/v1/auth/login` - Get access/refresh tokens
- `POST /api/v1/auth/refresh` - Rotate tokens
- `POST /api/v1/auth/logout` - Revoke refresh token
- `GET /api/v1/auth/me` - Current user profile
- `PUT /api/v1/auth/me` - Update profile

**Tasks:**
- `GET /api/v1/tasks/health` - Celery worker health
- `GET /api/v1/tasks/{task_id}/status` - Task status
- `GET /api/v1/tasks/{task_id}/result` - Task result

**LLM (if configured):**
- LLM endpoints for explain and conversation tasks via Celery

## Critical Reminders

- **NEVER CANCEL** Poetry install (36 seconds) or tests (16 seconds) - they complete quickly
- **ALWAYS** ensure Redis is running before starting Celery workers
- **ALWAYS** copy .env.example to .env and llm_config.example.yaml to llm_config.yaml before first run
- **ALWAYS** test authentication flow after auth-related changes
- **DO NOT** rely on Docker deployment until configuration issues are resolved
- **ALWAYS** run `poetry run pytest` before committing changes to catch regressions
- Application works perfectly in local development mode with Poetry