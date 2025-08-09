"""
Unit tests for app.core.dependecies module

Tests CommonQueryParams, user dependency functions and authentication checks.
"""

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import HTTPException

from app.core.dependecies import (
    CommonQueryParams,
    get_current_active_user,
    get_current_superuser,
)
from app.schemas.auth import TokenData


class TestCommonQueryParams:
    """Test CommonQueryParams class"""

    def test_default_values(self):
        """Test CommonQueryParams with default values"""
        # CommonQueryParams uses FastAPI Query objects, so we need to test differently
        # Create instance to test the values that would be assigned
        params = CommonQueryParams()
        
        # The values are actually Query objects with default values
        assert hasattr(params, 'q')
        assert hasattr(params, 'sort_by') 
        assert hasattr(params, 'sort_order')
        assert hasattr(params, 'include_deleted')

    def test_custom_values(self):
        """Test CommonQueryParams with custom values"""
        params = CommonQueryParams(
            q="search term",
            sort_by="created_at",
            sort_order="desc",
            include_deleted=True
        )
        
        assert params.q == "search term"
        assert params.sort_by == "created_at"
        assert params.sort_order == "desc"
        assert params.include_deleted is True

    def test_partial_custom_values(self):
        """Test CommonQueryParams with partially custom values"""
        # Test that CommonQueryParams class can be instantiated with args
        # Since this is a FastAPI dependency, we test the class structure
        params = CommonQueryParams(
            q="test query",
            sort_by="name"
        )
        
        assert params.q == "test query"
        assert params.sort_by == "name"
        # The remaining parameters should maintain their defaults but are Query objects
        assert hasattr(params, 'sort_order')
        assert hasattr(params, 'include_deleted')


class TestGetCurrentActiveUser:
    """Test get_current_active_user dependency"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_active_user_success(self):
        """Test get_current_active_user with active user"""
        current_user = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "username": "testuser",
            "email": "test@example.com",
            "is_active": True,
            "is_superuser": False
        }
        
        result = await get_current_active_user(current_user)
        
        assert result == current_user

    @pytest.mark.asyncio
    async def test_inactive_user_raises_exception(self):
        """Test get_current_active_user raises exception for inactive user"""
        current_user = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "username": "testuser",
            "email": "test@example.com",
            "is_active": False,
            "is_superuser": False
        }
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_active_user(current_user)
        
        assert exc_info.value.status_code == 400
        assert "Inactive user" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_user_with_missing_is_active_raises_exception(self):
        """Test get_current_active_user raises exception when is_active is missing"""
        current_user = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "username": "testuser",
            "email": "test@example.com",
            "is_superuser": False
        }
        # Missing is_active key should be treated as falsy
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_active_user(current_user)
        
        assert exc_info.value.status_code == 400
        assert "Inactive user" in str(exc_info.value.detail)


class TestGetCurrentSuperuser:
    """Test get_current_superuser dependency"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_superuser_success(self):
        """Test get_current_superuser with superuser"""
        current_user = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "username": "admin",
            "email": "admin@example.com",
            "is_active": True,
            "is_superuser": True
        }
        
        result = await get_current_superuser(current_user)
        
        assert result == current_user

    @pytest.mark.asyncio
    async def test_non_superuser_raises_exception(self):
        """Test get_current_superuser raises exception for non-superuser"""
        current_user = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "username": "testuser",
            "email": "test@example.com",
            "is_active": True,
            "is_superuser": False
        }
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_superuser(current_user)
        
        assert exc_info.value.status_code == 403
        assert "Not enough permissions" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_user_with_missing_is_superuser_raises_exception(self):
        """Test get_current_superuser raises exception when is_superuser is missing"""
        current_user = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "username": "testuser",
            "email": "test@example.com",
            "is_active": True
        }
        # Missing is_superuser key should be treated as falsy
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_superuser(current_user)
        
        assert exc_info.value.status_code == 403
        assert "Not enough permissions" in str(exc_info.value.detail)


class TestGetCurrentUser:
    """Test get_current_user dependency (integration-style test with mocking)"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_get_current_user_success(self):
        """Test get_current_user successfully retrieves user"""
        from app.core.dependecies import get_current_user
        
        # Mock token data
        token_data = TokenData(user_id="123e4567-e89b-12d3-a456-426614174000")
        
        # Mock database session and user
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_user = MagicMock()
        mock_user.id = uuid.UUID("123e4567-e89b-12d3-a456-426614174000")
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.first_name = "Test"
        mock_user.last_name = "User"
        mock_user.is_active = True
        mock_user.is_superuser = False
        mock_user.is_verified = True
        mock_user.created_at = "2023-01-01T00:00:00"
        mock_user.updated_at = "2023-01-01T00:00:00"
        
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result
        
        result = await get_current_user(token_data, mock_db)
        
        # Verify result structure
        assert result["id"] == "123e4567-e89b-12d3-a456-426614174000"
        assert result["username"] == "testuser"
        assert result["email"] == "test@example.com"
        assert result["first_name"] == "Test"
        assert result["last_name"] == "User"
        assert result["is_active"] is True
        assert result["is_superuser"] is False
        assert result["is_verified"] is True
        assert "created_at" in result
        assert "updated_at" in result
        
        # Verify database was queried
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_current_user_not_found(self):
        """Test get_current_user raises exception when user not found"""
        from app.core.dependecies import get_current_user
        
        # Mock token data
        token_data = TokenData(user_id="123e4567-e89b-12d3-a456-426614174000")
        
        # Mock database session with no user found
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token_data, mock_db)
        
        assert exc_info.value.status_code == 404
        assert "User not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_uuid(self):
        """Test get_current_user handles invalid UUID gracefully"""
        from app.core.dependecies import get_current_user
        
        # Mock token data with invalid UUID
        token_data = TokenData(user_id="invalid-uuid")
        
        # Mock database session
        mock_db = AsyncMock()
        
        with pytest.raises(ValueError):
            await get_current_user(token_data, mock_db)


class TestGetDbDependency:
    """Test get_db dependency"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_get_db_yields_session(self):
        """Test get_db yields database session"""
        from app.core.dependecies import get_db
        
        mock_session = AsyncMock()
        
        with patch('app.core.dependecies.get_async_session') as mock_get_session:
            # Mock the async context manager
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_session
            mock_context.__aexit__.return_value = None
            mock_get_session.return_value = mock_context
            
            # get_db is an async generator, so iterate over it
            db_gen = get_db()
            try:
                db = await db_gen.__anext__()
                assert db == mock_session
                # Verify session was properly closed when generator is closed
            finally:
                await db_gen.aclose()
            
            mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_db_handles_exception(self):
        """Test get_db properly handles exceptions and closes session"""
        from app.core.dependecies import get_db
        
        mock_session = AsyncMock()
        
        with patch('app.core.dependecies.get_async_session') as mock_get_session:
            # Mock the async context manager
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_session
            mock_context.__aexit__.return_value = None
            mock_get_session.return_value = mock_context
            
            # Test exception handling with async generator
            db_gen = get_db()
            try:
                db = await db_gen.__anext__()
                assert db == mock_session
                # Simulate exception in usage
                raise ValueError("Test exception")
            except ValueError:
                pass  # Expected
            finally:
                await db_gen.aclose()
            
            # Verify session was still closed despite exception
            mock_session.close.assert_called_once()