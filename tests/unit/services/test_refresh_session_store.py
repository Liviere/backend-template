"""
Unit tests for app.services.refresh_session_store module

Tests RefreshSessionStore Redis-based session management functionality.
"""

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from jose import jwt

from app.services.refresh_session_store import RefreshSessionStore


class TestRefreshSessionStore:
    """Test RefreshSessionStore class functionality"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    def test_init(self, mock_get_settings, mock_get_redis):
        """Test RefreshSessionStore initialization"""
        mock_settings = MagicMock()
        mock_settings.redis_refresh_prefix = "test:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        store = RefreshSessionStore()

        assert store.settings == mock_settings
        assert store.redis == mock_redis
        assert store.prefix == "test:refresh:"

    def test_key_method(self):
        """Test _key method generates correct Redis key"""
        with patch('app.services.refresh_session_store.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.redis_refresh_prefix = "auth:refresh:"
            mock_get_settings.return_value = mock_settings
            
            with patch('app.services.refresh_session_store.get_redis'):
                store = RefreshSessionStore()
                
                key = store._key("test-jti-123")
                assert key == "auth:refresh:test-jti-123"

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    @pytest.mark.asyncio
    async def test_add_success(self, mock_get_settings, mock_get_redis):
        """Test add method successfully stores session"""
        mock_settings = MagicMock()
        mock_settings.redis_refresh_prefix = "auth:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_redis = AsyncMock()
        mock_get_redis.return_value = mock_redis

        store = RefreshSessionStore()
        
        # Test with future expiry
        future_exp = int(datetime.now(timezone.utc).timestamp()) + 3600  # 1 hour from now
        
        await store.add("test-jti", "user123", future_exp)

        # Verify Redis operations
        mock_redis.hset.assert_called_once_with(
            "auth:refresh:test-jti",
            mapping={"sub": "user123", "exp": future_exp}
        )
        mock_redis.expire.assert_called_once()
        # TTL should be positive
        expire_call = mock_redis.expire.call_args
        assert expire_call[0][0] == "auth:refresh:test-jti"
        assert expire_call[0][1] > 0

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    @pytest.mark.asyncio
    async def test_add_expired_token(self, mock_get_settings, mock_get_redis):
        """Test add method with expired token (TTL = 0)"""
        mock_settings = MagicMock()
        mock_settings.redis_refresh_prefix = "auth:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_redis = AsyncMock()
        mock_get_redis.return_value = mock_redis

        store = RefreshSessionStore()
        
        # Test with past expiry
        past_exp = int(datetime.now(timezone.utc).timestamp()) - 3600  # 1 hour ago
        
        await store.add("test-jti", "user123", past_exp)

        # Should still set the hash but not set expiry (TTL <= 0)
        mock_redis.hset.assert_called_once()
        mock_redis.expire.assert_not_called()

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    @pytest.mark.asyncio
    async def test_add_redis_exception(self, mock_get_settings, mock_get_redis):
        """Test add method handles Redis exceptions gracefully"""
        mock_settings = MagicMock()
        mock_settings.redis_refresh_prefix = "auth:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_redis = AsyncMock()
        mock_redis.hset.side_effect = Exception("Redis unavailable")
        mock_get_redis.return_value = mock_redis

        store = RefreshSessionStore()
        
        # Should not raise exception
        await store.add("test-jti", "user123", 9999999999)

        mock_redis.hset.assert_called_once()

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    @pytest.mark.asyncio
    async def test_exists_true(self, mock_get_settings, mock_get_redis):
        """Test exists method returns True when key exists"""
        mock_settings = MagicMock()
        mock_settings.redis_refresh_prefix = "auth:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_redis = AsyncMock()
        mock_redis.exists.return_value = 1  # Redis EXISTS returns count
        mock_get_redis.return_value = mock_redis

        store = RefreshSessionStore()
        
        result = await store.exists("test-jti")

        assert result is True
        mock_redis.exists.assert_called_once_with("auth:refresh:test-jti")

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    @pytest.mark.asyncio
    async def test_exists_false(self, mock_get_settings, mock_get_redis):
        """Test exists method returns False when key doesn't exist"""
        mock_settings = MagicMock()
        mock_settings.redis_refresh_prefix = "auth:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_redis = AsyncMock()
        mock_redis.exists.return_value = 0  # Redis EXISTS returns 0 for non-existent
        mock_get_redis.return_value = mock_redis

        store = RefreshSessionStore()
        
        result = await store.exists("test-jti")

        assert result is False
        mock_redis.exists.assert_called_once_with("auth:refresh:test-jti")

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    @pytest.mark.asyncio
    async def test_exists_exception(self, mock_get_settings, mock_get_redis):
        """Test exists method returns False when Redis exception occurs"""
        mock_settings = MagicMock()
        mock_settings.redis_refresh_prefix = "auth:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_redis = AsyncMock()
        mock_redis.exists.side_effect = Exception("Redis unavailable")
        mock_get_redis.return_value = mock_redis

        store = RefreshSessionStore()
        
        result = await store.exists("test-jti")

        assert result is False
        mock_redis.exists.assert_called_once()

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    @pytest.mark.asyncio
    async def test_revoke_success(self, mock_get_settings, mock_get_redis):
        """Test revoke method successfully deletes session"""
        mock_settings = MagicMock()
        mock_settings.redis_refresh_prefix = "auth:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_redis = AsyncMock()
        mock_get_redis.return_value = mock_redis

        store = RefreshSessionStore()
        
        await store.revoke("test-jti")

        mock_redis.delete.assert_called_once_with("auth:refresh:test-jti")

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    @pytest.mark.asyncio
    async def test_revoke_exception(self, mock_get_settings, mock_get_redis):
        """Test revoke method handles Redis exceptions gracefully"""
        mock_settings = MagicMock()
        mock_settings.redis_refresh_prefix = "auth:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_redis = AsyncMock()
        mock_redis.delete.side_effect = Exception("Redis unavailable")
        mock_get_redis.return_value = mock_redis

        store = RefreshSessionStore()
        
        # Should not raise exception
        await store.revoke("test-jti")

        mock_redis.delete.assert_called_once()

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    @pytest.mark.asyncio
    async def test_get_subject_success(self, mock_get_settings, mock_get_redis):
        """Test get_subject method successfully retrieves subject"""
        mock_settings = MagicMock()
        mock_settings.redis_refresh_prefix = "auth:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_redis = AsyncMock()
        mock_redis.hgetall.return_value = {"sub": "user123", "exp": "1234567890"}
        mock_get_redis.return_value = mock_redis

        store = RefreshSessionStore()
        
        result = await store.get_subject("test-jti")

        assert result == "user123"
        mock_redis.hgetall.assert_called_once_with("auth:refresh:test-jti")

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    @pytest.mark.asyncio
    async def test_get_subject_not_found(self, mock_get_settings, mock_get_redis):
        """Test get_subject method returns None when session not found"""
        mock_settings = MagicMock()
        mock_settings.redis_refresh_prefix = "auth:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_redis = AsyncMock()
        mock_redis.hgetall.return_value = {}  # Empty dict when key doesn't exist
        mock_get_redis.return_value = mock_redis

        store = RefreshSessionStore()
        
        result = await store.get_subject("test-jti")

        assert result is None
        mock_redis.hgetall.assert_called_once_with("auth:refresh:test-jti")

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    @pytest.mark.asyncio
    async def test_get_subject_exception(self, mock_get_settings, mock_get_redis):
        """Test get_subject method returns None when Redis exception occurs"""
        mock_settings = MagicMock()
        mock_settings.redis_refresh_prefix = "auth:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_redis = AsyncMock()
        mock_redis.hgetall.side_effect = Exception("Redis unavailable")
        mock_get_redis.return_value = mock_redis

        store = RefreshSessionStore()
        
        result = await store.get_subject("test-jti")

        assert result is None
        mock_redis.hgetall.assert_called_once()

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    @pytest.mark.asyncio
    async def test_decode_and_validate_refresh_success(self, mock_get_settings, mock_get_redis):
        """Test decode_and_validate_refresh successfully validates token"""
        mock_settings = MagicMock()
        mock_settings.secret_key = "test-secret"
        mock_settings.algorithm = "HS256"
        mock_settings.redis_refresh_prefix = "auth:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_redis = AsyncMock()
        mock_redis.exists.return_value = 1  # Session exists in Redis
        mock_get_redis.return_value = mock_redis

        store = RefreshSessionStore()
        
        # Create a valid refresh token
        payload = {
            "sub": "user123",
            "type": "refresh",
            "jti": "test-jti",
            "exp": 9999999999
        }
        token = jwt.encode(payload, "test-secret", algorithm="HS256")
        
        result = await store.decode_and_validate_refresh(token)

        assert result["sub"] == "user123"
        assert result["type"] == "refresh"
        assert result["jti"] == "test-jti"
        mock_redis.exists.assert_called_once_with("auth:refresh:test-jti")

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    @pytest.mark.asyncio
    async def test_decode_and_validate_refresh_wrong_type(self, mock_get_settings, mock_get_redis):
        """Test decode_and_validate_refresh raises error for wrong token type"""
        mock_settings = MagicMock()
        mock_settings.secret_key = "test-secret"
        mock_settings.algorithm = "HS256"
        mock_settings.redis_refresh_prefix = "auth:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_get_redis.return_value = AsyncMock()

        store = RefreshSessionStore()
        
        # Create an access token instead of refresh token
        payload = {
            "sub": "user123",
            "type": "access",  # Wrong type
            "jti": "test-jti",
            "exp": 9999999999
        }
        token = jwt.encode(payload, "test-secret", algorithm="HS256")
        
        with pytest.raises(ValueError, match="Not a refresh token"):
            await store.decode_and_validate_refresh(token)

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    @pytest.mark.asyncio
    async def test_decode_and_validate_refresh_missing_fields(self, mock_get_settings, mock_get_redis):
        """Test decode_and_validate_refresh raises error for missing required fields"""
        mock_settings = MagicMock()
        mock_settings.secret_key = "test-secret"
        mock_settings.algorithm = "HS256"
        mock_settings.redis_refresh_prefix = "auth:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_get_redis.return_value = AsyncMock()

        store = RefreshSessionStore()
        
        # Test missing jti
        payload = {
            "sub": "user123",
            "type": "refresh",
            "exp": 9999999999
            # Missing jti
        }
        token = jwt.encode(payload, "test-secret", algorithm="HS256")
        
        with pytest.raises(ValueError, match="Malformed refresh token"):
            await store.decode_and_validate_refresh(token)

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    @pytest.mark.asyncio
    async def test_decode_and_validate_refresh_session_not_found(self, mock_get_settings, mock_get_redis):
        """Test decode_and_validate_refresh raises error when session not in Redis"""
        mock_settings = MagicMock()
        mock_settings.secret_key = "test-secret"
        mock_settings.algorithm = "HS256"
        mock_settings.redis_refresh_prefix = "auth:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_redis = AsyncMock()
        mock_redis.exists.return_value = 0  # Session doesn't exist in Redis
        mock_get_redis.return_value = mock_redis

        store = RefreshSessionStore()
        
        # Create a valid refresh token
        payload = {
            "sub": "user123",
            "type": "refresh",
            "jti": "test-jti",
            "exp": 9999999999
        }
        token = jwt.encode(payload, "test-secret", algorithm="HS256")
        
        with pytest.raises(ValueError, match="Refresh session not found or revoked"):
            await store.decode_and_validate_refresh(token)

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    @pytest.mark.asyncio
    async def test_decode_and_validate_refresh_invalid_token(self, mock_get_settings, mock_get_redis):
        """Test decode_and_validate_refresh handles invalid JWT tokens"""
        mock_settings = MagicMock()
        mock_settings.secret_key = "test-secret"
        mock_settings.algorithm = "HS256"
        mock_settings.redis_refresh_prefix = "auth:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_get_redis.return_value = AsyncMock()

        store = RefreshSessionStore()
        
        # Invalid token
        with pytest.raises((ValueError, jwt.JWTError)):
            await store.decode_and_validate_refresh("invalid.token.here")


class TestRefreshSessionStoreIntegration:
    """Test RefreshSessionStore integration scenarios"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @patch('app.services.refresh_session_store.get_redis')
    @patch('app.services.refresh_session_store.get_settings')
    @pytest.mark.asyncio
    async def test_full_session_lifecycle(self, mock_get_settings, mock_get_redis):
        """Test complete session lifecycle: add, exists, get_subject, revoke"""
        mock_settings = MagicMock()
        mock_settings.secret_key = "test-secret"
        mock_settings.algorithm = "HS256"
        mock_settings.redis_refresh_prefix = "auth:refresh:"
        mock_get_settings.return_value = mock_settings
        
        mock_redis = AsyncMock()
        mock_redis.exists.return_value = 1
        mock_redis.hgetall.return_value = {"sub": "user123", "exp": "9999999999"}
        mock_get_redis.return_value = mock_redis

        store = RefreshSessionStore()
        
        # Add session
        future_exp = int(datetime.now(timezone.utc).timestamp()) + 3600
        await store.add("session-jti", "user123", future_exp)
        
        # Check it exists
        exists = await store.exists("session-jti")
        assert exists is True
        
        # Get subject
        subject = await store.get_subject("session-jti")
        assert subject == "user123"
        
        # Revoke session
        await store.revoke("session-jti")
        
        # Verify all Redis operations were called
        assert mock_redis.hset.called
        assert mock_redis.exists.called
        assert mock_redis.hgetall.called
        assert mock_redis.delete.called