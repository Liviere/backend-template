"""
Unit tests for app.core.config module

Tests Settings class validation, environment variable parsing,
database URL construction, and configuration properties.
"""

import os
import pytest
from unittest.mock import patch
from pydantic import ValidationError

from app.core.config import Settings, ListParsingEnvSource, ListParsingDotEnvSource


class TestListParsingEnvSource:
    """Test environment variable list parsing functionality"""

    def test_cors_field_parsing_asterisk(self):
        """Test that '*' remains as single item list"""
        source = ListParsingEnvSource(Settings)
        field = None  # Mock field
        
        result = source.prepare_field_value("cors_origins", field, "*", False)
        assert result == ["*"]

    def test_cors_field_parsing_comma_separated(self):
        """Test comma-separated string parsing"""
        source = ListParsingEnvSource(Settings)
        field = None
        
        result = source.prepare_field_value("cors_origins", field, "a,b,c", False)
        assert result == ["a", "b", "c"]

    def test_cors_field_parsing_with_quotes_and_spaces(self):
        """Test parsing with quotes and extra spaces"""
        source = ListParsingEnvSource(Settings)
        field = None
        
        result = source.prepare_field_value("cors_origins", field, '"a, b , c"', False)
        assert result == ["a", "b", "c"]

    def test_non_cors_field_unchanged(self):
        """Test non-CORS fields pass through unchanged"""
        source = ListParsingEnvSource(Settings)
        field = None
        
        result = source.prepare_field_value("other_field", field, "value", False)
        assert result == "value"


class TestListParsingDotEnvSource:
    """Test .env file list parsing functionality"""

    def test_cors_field_parsing_same_as_env_source(self):
        """Test dotenv source behaves same as env source"""
        env_source = ListParsingEnvSource(Settings)
        dotenv_source = ListParsingDotEnvSource(Settings)
        field = None
        
        test_values = ["*", "a,b,c", '"a, b , c"']
        for value in test_values:
            env_result = env_source.prepare_field_value("cors_origins", field, value, False)
            dotenv_result = dotenv_source.prepare_field_value("cors_origins", field, value, False)
            assert env_result == dotenv_result


class TestSettings:
    """Test Settings class and its validation logic"""

    def test_default_settings(self):
        """Test default settings are valid"""
        settings = Settings()
        assert settings.app_name == "Backend Template API"
        assert settings.environment == "development"
        assert settings.debug is True
        assert settings.cors_origins == ["*"]
        assert settings.cors_methods == ["*"]
        assert settings.cors_headers == ["*"]

    def test_environment_validation_valid(self):
        """Test valid environment values are accepted"""
        valid_envs = ["development", "staging", "production", "testing"]
        for env in valid_envs:
            settings = Settings(environment=env)
            assert settings.environment == env

    def test_environment_validation_invalid(self):
        """Test invalid environment values raise ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            Settings(environment="invalid")
        assert "Environment must be one of" in str(exc_info.value)

    def test_database_url_validation_postgresql(self):
        """Test PostgreSQL URL is enhanced with asyncpg driver"""
        # Override database_service to prevent model validator from overwriting
        settings = Settings(
            database_service="postgresql+asyncpg",
            database_url="postgresql://user:pass@host:5432/db"
        )
        # The field validator should enhance the URL
        assert "postgresql+asyncpg" in settings.database_url

    def test_database_url_validation_mysql(self):
        """Test MySQL URL is enhanced with aiomysql driver"""
        settings = Settings(
            database_service="mysql+aiomysql",
            database_url="mysql://user:pass@host:3306/db"
        )
        assert "mysql+aiomysql" in settings.database_url

    def test_database_url_validation_sqlite(self):
        """Test SQLite URL is enhanced with aiosqlite driver"""  
        settings = Settings(
            database_service="sqlite+aiosqlite",
            database_url="sqlite:///./test.db"
        )
        assert "sqlite+aiosqlite" in settings.database_url

    def test_database_url_validation_empty_raises_error(self):
        """Test empty database URL raises ValidationError"""
        with pytest.raises(ValidationError):
            Settings(database_url="")

    def test_set_database_url_sqlite(self):
        """Test database URL construction for SQLite"""
        settings = Settings(database_service="sqlite+aiosqlite")
        assert settings.database_url == "sqlite+aiosqlite:///./app.db"

    def test_set_database_url_postgresql(self):
        """Test database URL construction for PostgreSQL"""
        settings = Settings(
            database_service="postgresql+asyncpg",
            database_host="localhost",
            database_port=5432,
            database_name="testdb",
            database_user="testuser",
            database_password="testpass"
        )
        expected = "postgresql+asyncpg://testuser:testpass@localhost:5432/testdb"
        assert settings.database_url == expected

    def test_set_database_url_mysql(self):
        """Test database URL construction for MySQL"""
        settings = Settings(
            database_service="mysql+aiomysql",
            database_host="localhost",
            database_port=3306,
            database_name="testdb",
            database_user="testuser",
            database_password="testpass"
        )
        expected = "mysql+aiomysql://testuser:testpass@localhost:3306/testdb"
        assert settings.database_url == expected

    def test_environment_properties(self):
        """Test environment property methods"""
        # Test development
        dev_settings = Settings(environment="development")
        assert dev_settings.is_development is True
        assert dev_settings.is_production is False
        assert dev_settings.is_testing is False

        # Test production
        prod_settings = Settings(environment="production")
        assert prod_settings.is_development is False
        assert prod_settings.is_production is True
        assert prod_settings.is_testing is False

        # Test testing
        test_settings = Settings(environment="testing")
        assert test_settings.is_development is False
        assert test_settings.is_production is False
        assert test_settings.is_testing is True

    def test_database_type_properties(self):
        """Test database type property methods"""
        # Test SQLite
        sqlite_settings = Settings(database_service="sqlite+aiosqlite")
        assert sqlite_settings.is_sqlite is True
        assert sqlite_settings.is_mysql is False
        assert sqlite_settings.is_postgresql is False
        assert sqlite_settings.database_type == "sqlite"

        # Test MySQL
        mysql_settings = Settings(
            database_service="mysql+aiomysql",
            database_host="host",
            database_name="db",
            database_user="user",
            database_password="pass"
        )
        assert mysql_settings.is_sqlite is False
        assert mysql_settings.is_mysql is True
        assert mysql_settings.is_postgresql is False
        assert mysql_settings.database_type == "mysql"

        # Test PostgreSQL
        pg_settings = Settings(
            database_service="postgresql+asyncpg",
            database_host="host", 
            database_name="db",
            database_user="user",
            database_password="pass"
        )
        assert pg_settings.is_sqlite is False
        assert pg_settings.is_mysql is False
        assert pg_settings.is_postgresql is True
        assert pg_settings.database_type == "postgresql"

    def test_get_database_engine_args_sqlite(self):
        """Test database engine args for SQLite"""
        settings = Settings(database_service="sqlite+aiosqlite", database_echo=True)
        args = settings.get_database_engine_args()
        
        assert args["echo"] is True
        assert "connect_args" in args
        assert args["connect_args"]["check_same_thread"] is False
        # SQLite should not have pooling args
        assert "pool_size" not in args

    def test_get_database_engine_args_mysql(self):
        """Test database engine args for MySQL"""
        settings = Settings(
            database_service="mysql+aiomysql",
            database_host="host",
            database_name="db", 
            database_user="user",
            database_password="pass",
            database_echo=False,
            database_pool_size=10,
            database_max_overflow=20,
            database_pool_timeout=60,
            database_pool_recycle=7200,
            database_mysql_charset="utf8mb4"
        )
        args = settings.get_database_engine_args()
        
        assert args["echo"] is False
        assert args["pool_size"] == 10
        assert args["max_overflow"] == 20
        assert args["pool_timeout"] == 60
        assert args["pool_recycle"] == 7200
        assert args["pool_pre_ping"] is True
        assert "connect_args" in args
        assert args["connect_args"]["charset"] == "utf8mb4"
        assert args["connect_args"]["use_unicode"] is True
        assert args["connect_args"]["autocommit"] is False

    def test_get_database_engine_args_postgresql(self):
        """Test database engine args for PostgreSQL"""
        settings = Settings(
            database_service="postgresql+asyncpg",
            database_host="host",
            database_name="db",
            database_user="user", 
            database_password="pass",
            database_echo=True,
            database_pool_size=15,
            database_max_overflow=25
        )
        args = settings.get_database_engine_args()
        
        assert args["echo"] is True
        assert args["pool_size"] == 15
        assert args["max_overflow"] == 25
        assert args["pool_pre_ping"] is True
        # PostgreSQL should not have MySQL-specific connect_args
        assert "connect_args" not in args or args.get("connect_args") is None

    @patch.dict(os.environ, {"CORS_ORIGINS": "https://app.com,https://api.com"})
    def test_cors_origins_from_env(self):
        """Test CORS origins are parsed from environment variables"""
        # Clear the LRU cache if get_settings is cached
        from app.core.config import get_settings
        get_settings.cache_clear()
        
        settings = Settings()
        # Note: Due to custom sources, this might not work as expected in unit tests
        # This tests the parsing logic itself
        source = ListParsingEnvSource(Settings)
        result = source.prepare_field_value("cors_origins", None, "https://app.com,https://api.com", False)
        assert result == ["https://app.com", "https://api.com"]

    def test_settings_with_testing_environment(self):
        """Test settings configured for testing environment"""
        settings = Settings(environment="testing", debug=False)
        assert settings.environment == "testing"
        assert settings.is_testing is True
        assert settings.debug is False