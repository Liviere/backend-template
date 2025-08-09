"""
Unit tests for app.core.redis_client module

Tests Redis connection management and health check functionality.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import redis.asyncio as aioredis

from app.core.redis_client import get_redis, ensure_redis_connection


class TestGetRedis:
    """Test get_redis function"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()
        # Clear the LRU cache for get_redis
        get_redis.cache_clear()

    @patch('app.core.redis_client.get_settings')
    @patch('redis.asyncio.from_url')
    def test_get_redis_creates_connection(self, mock_from_url, mock_get_settings):
        """Test get_redis creates Redis connection with correct URL"""
        mock_redis = MagicMock()
        mock_from_url.return_value = mock_redis
        
        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_get_settings.return_value = mock_settings

        result = get_redis()

        mock_from_url.assert_called_once_with(
            "redis://localhost:6379/0", 
            decode_responses=True
        )
        assert result == mock_redis

    @patch('redis.asyncio.from_url')
    def test_get_redis_caching(self, mock_from_url):
        """Test get_redis caches connection (LRU cache)"""
        mock_redis = MagicMock()
        mock_from_url.return_value = mock_redis

        with patch('app.core.config.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.redis_url = "redis://localhost:6379/0"
            mock_get_settings.return_value = mock_settings

            # Call get_redis multiple times
            result1 = get_redis()
            result2 = get_redis()
            result3 = get_redis()

            # Should only call from_url once due to caching
            mock_from_url.assert_called_once()
            assert result1 == result2 == result3 == mock_redis

    @patch('app.core.redis_client.get_settings')
    @patch('redis.asyncio.from_url')
    def test_get_redis_with_custom_url(self, mock_from_url, mock_get_settings):
        """Test get_redis with custom Redis URL"""
        mock_redis = MagicMock()
        mock_from_url.return_value = mock_redis
        
        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://custom-host:6380/5"
        mock_get_settings.return_value = mock_settings

        result = get_redis()

        mock_from_url.assert_called_once_with(
            "redis://custom-host:6380/5",
            decode_responses=True
        )
        assert result == mock_redis


class TestEnsureRedisConnection:
    """Test ensure_redis_connection function"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()
        get_redis.cache_clear()

    @patch('app.core.redis_client.get_redis')
    @pytest.mark.asyncio
    async def test_ensure_redis_connection_success(self, mock_get_redis):
        """Test ensure_redis_connection returns True when ping succeeds"""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_get_redis.return_value = mock_redis

        result = await ensure_redis_connection()

        assert result is True
        mock_redis.ping.assert_called_once()

    @patch('app.core.redis_client.get_redis')
    @pytest.mark.asyncio
    async def test_ensure_redis_connection_ping_false(self, mock_get_redis):
        """Test ensure_redis_connection returns False when ping returns False"""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = False
        mock_get_redis.return_value = mock_redis

        result = await ensure_redis_connection()

        assert result is False
        mock_redis.ping.assert_called_once()

    @patch('app.core.redis_client.get_redis')
    @pytest.mark.asyncio
    async def test_ensure_redis_connection_ping_none(self, mock_get_redis):
        """Test ensure_redis_connection returns False when ping returns None"""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = None
        mock_get_redis.return_value = mock_redis

        result = await ensure_redis_connection()

        assert result is False
        mock_redis.ping.assert_called_once()

    @patch('app.core.redis_client.get_redis')
    @pytest.mark.asyncio
    async def test_ensure_redis_connection_exception(self, mock_get_redis):
        """Test ensure_redis_connection returns False when exception occurs"""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Connection failed")
        mock_get_redis.return_value = mock_redis

        result = await ensure_redis_connection()

        assert result is False
        mock_redis.ping.assert_called_once()

    @patch('app.core.redis_client.get_redis')
    @pytest.mark.asyncio
    async def test_ensure_redis_connection_redis_error(self, mock_get_redis):
        """Test ensure_redis_connection handles Redis-specific errors"""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = aioredis.ConnectionError("Redis unavailable")
        mock_get_redis.return_value = mock_redis

        result = await ensure_redis_connection()

        assert result is False
        mock_redis.ping.assert_called_once()

    @patch('app.core.redis_client.get_redis')
    @pytest.mark.asyncio
    async def test_ensure_redis_connection_timeout_error(self, mock_get_redis):
        """Test ensure_redis_connection handles timeout errors"""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = aioredis.TimeoutError("Request timeout")
        mock_get_redis.return_value = mock_redis

        result = await ensure_redis_connection()

        assert result is False
        mock_redis.ping.assert_called_once()

    @patch('app.core.redis_client.get_redis')
    @pytest.mark.asyncio
    async def test_ensure_redis_connection_async_timeout(self, mock_get_redis):
        """Test ensure_redis_connection handles asyncio timeout"""
        import asyncio
        
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = asyncio.TimeoutError()
        mock_get_redis.return_value = mock_redis

        result = await ensure_redis_connection()

        assert result is False
        mock_redis.ping.assert_called_once()


class TestRedisClientIntegration:
    """Test Redis client integration scenarios"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()
        get_redis.cache_clear()

    @patch('redis.asyncio.from_url')
    @pytest.mark.asyncio
    async def test_redis_client_integration(self, mock_from_url):
        """Test complete Redis client integration flow"""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_from_url.return_value = mock_redis

        with patch('app.core.config.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.redis_url = "redis://localhost:6379/10"
            mock_get_settings.return_value = mock_settings

            # Test get_redis creates connection
            redis_client = get_redis()
            assert redis_client == mock_redis

            # Test ensure_redis_connection works
            is_connected = await ensure_redis_connection()
            assert is_connected is True

            # Verify the connection was created with correct parameters
            mock_from_url.assert_called_once_with(
                "redis://localhost:6379/10",
                decode_responses=True
            )

    @patch('redis.asyncio.from_url')
    @pytest.mark.asyncio
    async def test_redis_client_connection_failure(self, mock_from_url):
        """Test Redis client behavior when connection fails"""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = aioredis.ConnectionError("Cannot connect")
        mock_from_url.return_value = mock_redis

        with patch('app.core.config.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.redis_url = "redis://nonexistent:6379/0"
            mock_get_settings.return_value = mock_settings

            # Test get_redis still returns client (doesn't validate connection)
            redis_client = get_redis()
            assert redis_client == mock_redis

            # Test ensure_redis_connection detects failure
            is_connected = await ensure_redis_connection()
            assert is_connected is False

    @patch('app.core.redis_client.get_redis')
    @pytest.mark.asyncio
    async def test_multiple_health_checks(self, mock_get_redis):
        """Test multiple health checks use same Redis instance"""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_get_redis.return_value = mock_redis

        # Perform multiple health checks
        result1 = await ensure_redis_connection()
        result2 = await ensure_redis_connection()
        result3 = await ensure_redis_connection()

        assert result1 is True
        assert result2 is True
        assert result3 is True

        # get_redis should be called for each health check
        # (since it's not cached at the ensure_redis_connection level)
        assert mock_get_redis.call_count == 3

        # ping should be called for each health check
        assert mock_redis.ping.call_count == 3