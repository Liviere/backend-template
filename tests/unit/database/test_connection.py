"""
Unit tests for app.database.sql.connection module

Tests database engine creation, session management, and health checks.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.database.sql.connection import (
    create_database_engine,
    get_engine,
    get_session_maker,
    DatabaseManager,
)


class TestCreateDatabaseEngine:
    """Test create_database_engine function"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @patch('app.database.sql.connection.create_async_engine')
    @patch('app.database.sql.connection.get_settings')
    def test_create_database_engine_with_settings(self, mock_get_settings, mock_create_engine):
        """Test create_database_engine uses settings configuration"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.database_url = "sqlite+aiosqlite:///./test.db"
        mock_settings.get_database_engine_args.return_value = {
            "echo": True,
            "connect_args": {"check_same_thread": False}
        }
        mock_settings.is_sqlite = True
        mock_settings.is_mysql = False
        mock_get_settings.return_value = mock_settings

        # Mock engine
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        engine = create_database_engine()

        mock_create_engine.assert_called_once_with(
            "sqlite+aiosqlite:///./test.db",
            echo=True,
            connect_args={"check_same_thread": False},
            future=True
        )
        assert engine == mock_engine

    @patch('app.database.sql.connection.create_async_engine')
    @patch('app.database.sql.connection.get_settings')
    def test_create_database_engine_with_custom_url(self, mock_get_settings, mock_create_engine):
        """Test create_database_engine with custom database URL"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.get_database_engine_args.return_value = {"echo": False}
        mock_settings.is_sqlite = False
        mock_settings.is_mysql = False
        mock_get_settings.return_value = mock_settings

        # Mock engine
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        custom_url = "postgresql+asyncpg://user:pass@host/db"
        engine = create_database_engine(custom_url)

        mock_create_engine.assert_called_once_with(
            custom_url,
            echo=False,
            future=True
        )
        assert engine == mock_engine


class TestGetEngine:
    """Test get_engine function"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()
        # Clear the global engine
        import app.database.sql.connection
        app.database.sql.connection._engine = None

    @patch('app.database.sql.connection.create_database_engine')
    def test_get_engine_creates_singleton(self, mock_create_engine):
        """Test get_engine creates and returns singleton engine"""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # First call should create engine
        engine1 = get_engine()
        assert engine1 == mock_engine
        mock_create_engine.assert_called_once_with(None)

        # Second call should return same engine without creating new one
        engine2 = get_engine()
        assert engine2 == mock_engine
        assert engine1 is engine2
        # Should still only be called once
        mock_create_engine.assert_called_once()


class TestGetSessionMaker:
    """Test get_session_maker function"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()
        # Clear the global session maker
        import app.database.sql.connection
        app.database.sql.connection._async_session_maker = None
        app.database.sql.connection._engine = None

    @patch('app.database.sql.connection.async_sessionmaker')
    @patch('app.database.sql.connection.get_engine')
    def test_get_session_maker_creates_singleton(self, mock_get_engine, mock_sessionmaker):
        """Test get_session_maker creates and returns singleton session maker"""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        
        mock_session_maker = MagicMock()
        mock_sessionmaker.return_value = mock_session_maker

        # First call should create session maker
        session_maker1 = get_session_maker()
        assert session_maker1 == mock_session_maker
        
        # Verify sessionmaker was called with correct parameters
        from sqlalchemy.ext.asyncio import AsyncSession
        mock_sessionmaker.assert_called_once_with(
            mock_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )

        # Second call should return same session maker
        session_maker2 = get_session_maker()
        assert session_maker2 == mock_session_maker
        assert session_maker1 is session_maker2
        # Should still only be called once
        mock_sessionmaker.assert_called_once()


class TestDatabaseManager:
    """Test DatabaseManager class functionality"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test health_check returns True when query succeeds"""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result

        result = await DatabaseManager.health_check(mock_session)

        assert result is True
        mock_session.execute.assert_called_once()
        # Verify the query was a SELECT 1
        call_args = mock_session.execute.call_args[0][0]
        assert "SELECT 1" in str(call_args) or "1" in str(call_args)

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health_check returns False when query fails"""
        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("Database error")

        result = await DatabaseManager.health_check(mock_session)

        assert result is False
        mock_session.execute.assert_called_once()

    @patch('app.database.sql.connection.get_settings')
    @pytest.mark.asyncio
    async def test_get_database_info_healthy(self, mock_get_settings):
        """Test get_database_info with healthy database"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.database_url = "postgresql+asyncpg://user:secret@host:5432/db"
        mock_settings.database_pool_size = 10
        mock_settings.database_max_overflow = 20
        mock_settings.database_echo = True
        mock_get_settings.return_value = mock_settings

        with patch.object(DatabaseManager, 'health_check', return_value=True):
            info = await DatabaseManager.get_database_info()

        assert info["url"] == "postgresql+asyncpg://***"  # Credentials masked
        assert info["status"] == "healthy"
        assert info["pool_size"] == 10
        assert info["max_overflow"] == 20
        assert info["echo"] is True

    @patch('app.database.sql.connection.get_settings')
    @pytest.mark.asyncio
    async def test_get_database_info_unhealthy(self, mock_get_settings):
        """Test get_database_info with unhealthy database"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.database_url = "mysql+aiomysql://user:pass@host/db"
        mock_settings.database_pool_size = 5
        mock_settings.database_max_overflow = 15
        mock_settings.database_echo = False
        mock_get_settings.return_value = mock_settings

        with patch.object(DatabaseManager, 'health_check', return_value=False):
            info = await DatabaseManager.get_database_info()

        assert info["url"] == "mysql+aiomysql://***"  # Credentials masked
        assert info["status"] == "unhealthy"
        assert info["pool_size"] == 5
        assert info["max_overflow"] == 15
        assert info["echo"] is False

    @patch('app.database.sql.connection.create_tables')
    @pytest.mark.asyncio
    async def test_init_database_success(self, mock_create_tables):
        """Test init_database successfully initializes database"""
        mock_create_tables.return_value = None

        await DatabaseManager.init_database()

        mock_create_tables.assert_called_once()

    @patch('app.database.sql.connection.create_tables')
    @pytest.mark.asyncio
    async def test_init_database_failure(self, mock_create_tables):
        """Test init_database handles errors"""
        mock_create_tables.side_effect = Exception("Init failed")

        with pytest.raises(Exception, match="Init failed"):
            await DatabaseManager.init_database()

        mock_create_tables.assert_called_once()

    @patch('app.database.sql.connection.create_tables')
    @patch('app.database.sql.connection.drop_tables')
    @pytest.mark.asyncio
    async def test_reset_database_success(self, mock_drop_tables, mock_create_tables):
        """Test reset_database successfully resets database"""
        mock_drop_tables.return_value = None
        mock_create_tables.return_value = None

        await DatabaseManager.reset_database()

        mock_drop_tables.assert_called_once()
        mock_create_tables.assert_called_once()

    @patch('app.database.sql.connection.drop_tables')
    @pytest.mark.asyncio
    async def test_reset_database_failure(self, mock_drop_tables):
        """Test reset_database handles errors"""
        mock_drop_tables.side_effect = Exception("Reset failed")

        with pytest.raises(Exception, match="Reset failed"):
            await DatabaseManager.reset_database()

        mock_drop_tables.assert_called_once()


class TestDatabaseEventListeners:
    """Test database event listeners for different database types"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @patch('app.database.sql.connection.event')
    @patch('app.database.sql.connection.create_async_engine')
    @patch('app.database.sql.connection.get_settings')
    def test_sqlite_event_listener_registered(self, mock_get_settings, mock_create_engine, mock_event):
        """Test SQLite event listener is registered"""
        # Mock settings for SQLite
        mock_settings = MagicMock()
        mock_settings.database_url = "sqlite+aiosqlite:///./test.db"
        mock_settings.get_database_engine_args.return_value = {"echo": False}
        mock_settings.is_sqlite = True
        mock_settings.is_mysql = False
        mock_get_settings.return_value = mock_settings

        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        create_database_engine()

        # Verify event listener was registered for SQLite
        mock_event.listens_for.assert_called()

    @patch('app.database.sql.connection.event')
    @patch('app.database.sql.connection.create_async_engine')
    @patch('app.database.sql.connection.get_settings')
    def test_mysql_event_listener_registered(self, mock_get_settings, mock_create_engine, mock_event):
        """Test MySQL event listener is registered"""
        # Mock settings for MySQL
        mock_settings = MagicMock()
        mock_settings.database_url = "mysql+aiomysql://user:pass@host/db"
        mock_settings.get_database_engine_args.return_value = {"echo": False}
        mock_settings.is_sqlite = False
        mock_settings.is_mysql = True
        mock_get_settings.return_value = mock_settings

        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        create_database_engine()

        # Verify event listener was registered for MySQL
        mock_event.listens_for.assert_called()


class TestDatabaseIntegration:
    """Test database integration scenarios"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()
        # Clear globals
        import app.database.sql.connection
        app.database.sql.connection._engine = None
        app.database.sql.connection._async_session_maker = None

    @patch('app.database.sql.connection.create_async_engine')
    @patch('app.database.sql.connection.async_sessionmaker')
    @patch('app.database.sql.connection.get_settings')
    def test_full_database_setup_flow(self, mock_get_settings, mock_sessionmaker, mock_create_engine):
        """Test complete database setup flow"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.database_url = "sqlite+aiosqlite:///./test.db"
        mock_settings.get_database_engine_args.return_value = {"echo": False}
        mock_settings.is_sqlite = True
        mock_settings.is_mysql = False
        mock_get_settings.return_value = mock_settings

        # Mock engine and session maker
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_maker = MagicMock()
        mock_sessionmaker.return_value = mock_session_maker

        # Test engine creation and caching
        engine1 = get_engine()
        engine2 = get_engine()
        assert engine1 is engine2

        # Test session maker creation and caching
        sm1 = get_session_maker()
        sm2 = get_session_maker()
        assert sm1 is sm2

        # Verify engine creation was called once
        mock_create_engine.assert_called_once()
        # Verify session maker creation was called once
        from sqlalchemy.ext.asyncio import AsyncSession
        mock_sessionmaker.assert_called_once_with(
            mock_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )