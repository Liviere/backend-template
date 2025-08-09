"""
Unit tests for app.core.security module

Tests SecurityManager class, password hashing, JWT token creation/verification,
and password reset functionality.
"""

import os
from datetime import datetime, timedelta, UTC
from unittest.mock import patch, MagicMock
import pytest
from jose import jwt
from pydantic import BaseModel

from app.core.security import (
    SecurityManager,
    security_manager,
    create_access_token,
    create_refresh_token,
    verify_password,
    get_password_hash,
)


class TestSecurityManager:
    """Test SecurityManager class functionality"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        # Clear any cached settings
        from app.core.config import get_settings
        get_settings.cache_clear()
        self.security_manager = SecurityManager()

    def test_password_hashing_and_verification(self):
        """Test password hashing and verification"""
        password = "test_password_123"
        
        # Test hashing
        hashed = self.security_manager.get_password_hash(password)
        assert hashed != password
        assert len(hashed) > 0
        assert hashed.startswith("$2b$")  # bcrypt format
        
        # Test verification with correct password
        assert self.security_manager.verify_password(password, hashed) is True
        
        # Test verification with incorrect password
        assert self.security_manager.verify_password("wrong_password", hashed) is False

    def test_create_access_token_default_expiry(self):
        """Test access token creation with default expiry"""
        data = {"sub": "user123"}
        token = self.security_manager.create_access_token(data)
        
        # Decode token to verify structure
        payload = jwt.decode(
            token, 
            self.security_manager.settings.secret_key, 
            algorithms=[self.security_manager.settings.algorithm]
        )
        
        assert payload["sub"] == "user123"
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "jti" in payload
        assert len(payload["jti"]) > 0

    def test_create_access_token_custom_expiry(self):
        """Test access token creation with custom expiry"""
        data = {"sub": "user123"}
        expires_delta = timedelta(minutes=60)
        token = self.security_manager.create_access_token(data, expires_delta)
        
        payload = jwt.decode(
            token,
            self.security_manager.settings.secret_key,
            algorithms=[self.security_manager.settings.algorithm]
        )
        
        # Verify expiry time is approximately correct (within 1 minute)
        expected_exp = datetime.now(UTC) + expires_delta
        actual_exp = datetime.fromtimestamp(payload["exp"], UTC)
        assert abs((expected_exp - actual_exp).total_seconds()) < 60

    def test_create_refresh_token(self):
        """Test refresh token creation"""
        data = {"sub": "user123"}
        token = self.security_manager.create_refresh_token(data)
        
        payload = jwt.decode(
            token,
            self.security_manager.settings.secret_key,
            algorithms=[self.security_manager.settings.algorithm]
        )
        
        assert payload["sub"] == "user123"
        assert payload["type"] == "refresh"
        assert "exp" in payload
        assert "jti" in payload
        assert len(payload["jti"]) > 0
        
        # Verify refresh token has longer expiry
        expected_exp = datetime.now(UTC) + timedelta(
            days=self.security_manager.settings.refresh_token_expire_days
        )
        actual_exp = datetime.fromtimestamp(payload["exp"], UTC)
        assert abs((expected_exp - actual_exp).total_seconds()) < 300  # Within 5 minutes

    def test_verify_token_valid_access_token(self):
        """Test token verification with valid access token"""
        data = {"sub": "user123"}
        token = self.security_manager.create_access_token(data)
        
        token_data = self.security_manager.verify_token(token)
        
        assert token_data is not None
        assert token_data.user_id == "user123"

    def test_verify_token_rejects_refresh_token(self):
        """Test that verify_token rejects refresh tokens"""
        data = {"sub": "user123"}
        refresh_token = self.security_manager.create_refresh_token(data)
        
        token_data = self.security_manager.verify_token(refresh_token)
        
        assert token_data is None

    def test_verify_token_malformed_token(self):
        """Test token verification with malformed token"""
        malformed_token = "invalid.token.here"
        
        token_data = self.security_manager.verify_token(malformed_token)
        
        assert token_data is None

    def test_verify_token_missing_sub(self):
        """Test token verification with missing sub field"""
        # Create token without sub field
        payload = {
            "exp": datetime.now(UTC) + timedelta(minutes=30),
            "type": "access",
            "jti": "test-jti"
        }
        token = jwt.encode(
            payload,
            self.security_manager.settings.secret_key,
            algorithm=self.security_manager.settings.algorithm
        )
        
        token_data = self.security_manager.verify_token(token)
        
        assert token_data is None

    def test_verify_token_missing_type(self):
        """Test token verification with missing type field"""
        # Create token without type field
        payload = {
            "sub": "user123",
            "exp": datetime.now(UTC) + timedelta(minutes=30),
            "jti": "test-jti"
        }
        token = jwt.encode(
            payload,
            self.security_manager.settings.secret_key,
            algorithm=self.security_manager.settings.algorithm
        )
        
        token_data = self.security_manager.verify_token(token)
        
        assert token_data is None

    def test_generate_password_reset_token(self):
        """Test password reset token generation"""
        email = "test@example.com"
        token = self.security_manager.generate_password_reset_token(email)
        
        payload = jwt.decode(
            token,
            self.security_manager.settings.secret_key,
            algorithms=[self.security_manager.settings.algorithm]
        )
        
        assert payload["sub"] == email
        assert payload["type"] == "reset"
        assert "exp" in payload
        assert "nbf" in payload

    def test_verify_password_reset_token_valid(self):
        """Test password reset token verification with valid token"""
        email = "test@example.com"
        token = self.security_manager.generate_password_reset_token(email)
        
        verified_email = self.security_manager.verify_password_reset_token(token)
        
        assert verified_email == email

    def test_verify_password_reset_token_wrong_type(self):
        """Test password reset token verification with wrong token type"""
        # Create access token instead of reset token
        data = {"sub": "test@example.com"}
        access_token = self.security_manager.create_access_token(data)
        
        verified_email = self.security_manager.verify_password_reset_token(access_token)
        
        assert verified_email is None

    def test_verify_password_reset_token_invalid(self):
        """Test password reset token verification with invalid token"""
        invalid_token = "invalid.token.here"
        
        verified_email = self.security_manager.verify_password_reset_token(invalid_token)
        
        assert verified_email is None

    def test_verify_password_reset_token_expired(self):
        """Test password reset token verification with expired token"""
        # Create expired token
        past_time = datetime.now(UTC) - timedelta(hours=2)
        payload = {
            "sub": "test@example.com",
            "type": "reset",
            "exp": int(past_time.timestamp()),
            "nbf": int((past_time - timedelta(hours=1)).timestamp())
        }
        expired_token = jwt.encode(
            payload,
            self.security_manager.settings.secret_key,
            algorithm=self.security_manager.settings.algorithm
        )
        
        verified_email = self.security_manager.verify_password_reset_token(expired_token)
        
        assert verified_email is None

    def test_generate_random_string_default_length(self):
        """Test random string generation with default length"""
        random_str = SecurityManager.generate_random_string()
        
        assert len(random_str) >= 32  # URL-safe base64 encoding makes it longer
        assert isinstance(random_str, str)

    def test_generate_random_string_custom_length(self):
        """Test random string generation with custom length"""
        length = 16
        random_str = SecurityManager.generate_random_string(length)
        
        assert len(random_str) >= length
        assert isinstance(random_str, str)

    def test_generate_random_string_uniqueness(self):
        """Test that generated random strings are unique"""
        str1 = SecurityManager.generate_random_string()
        str2 = SecurityManager.generate_random_string()
        
        assert str1 != str2


class TestSecurityFunctions:
    """Test module-level convenience functions"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    def test_create_access_token_function(self):
        """Test create_access_token convenience function"""
        data = {"sub": "user123"}
        token = create_access_token(data)
        
        # Verify it's a valid JWT token
        assert isinstance(token, str)
        assert len(token.split('.')) == 3  # JWT has 3 parts

    def test_create_refresh_token_function(self):
        """Test create_refresh_token convenience function"""
        data = {"sub": "user123"}
        token = create_refresh_token(data)
        
        # Verify it's a valid JWT token
        assert isinstance(token, str)
        assert len(token.split('.')) == 3

    def test_verify_password_function(self):
        """Test verify_password convenience function"""
        password = "test_password"
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed) is True
        assert verify_password("wrong_password", hashed) is False

    def test_get_password_hash_function(self):
        """Test get_password_hash convenience function"""
        password = "test_password"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert len(hashed) > 0
        assert hashed.startswith("$2b$")


class TestTokenDataImport:
    """Test TokenData handling when schemas might not be available"""

    def test_fallback_token_data(self):
        """Test that fallback TokenData works when schemas unavailable"""
        # This tests the fallback TokenData class defined in security.py
        from app.core.security import TokenData
        
        token_data = TokenData(user_id="test123")
        assert token_data.user_id == "test123"


class TestSecurityManagerWithMockedSettings:
    """Test SecurityManager with different settings configurations"""

    def test_security_manager_with_custom_settings(self):
        """Test SecurityManager with custom settings"""
        with patch('app.core.security.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.secret_key = "test-secret-key"
            mock_settings.algorithm = "HS256"
            mock_settings.access_token_expire_minutes = 15
            mock_settings.refresh_token_expire_days = 30
            mock_get_settings.return_value = mock_settings
            
            security_mgr = SecurityManager()
            
            assert security_mgr.settings == mock_settings
            
            # Test token creation with custom settings
            data = {"sub": "user123"}
            token = security_mgr.create_access_token(data)
            
            payload = jwt.decode(
                token,
                "test-secret-key",
                algorithms=["HS256"]
            )
            
            assert payload["sub"] == "user123"
            assert payload["type"] == "access"

    def test_jwt_error_handling(self):
        """Test proper handling of JWT errors"""
        security_mgr = SecurityManager()
        
        # Test with completely invalid token
        invalid_tokens = [
            "",
            "not.a.token",
            "invalid.jwt.token.with.too.many.parts",
            None
        ]
        
        for invalid_token in invalid_tokens:
            if invalid_token is not None:
                result = security_mgr.verify_token(invalid_token)
                assert result is None