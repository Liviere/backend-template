"""
Unit tests for app.services.auth_service module

Tests AuthService for user authentication and management with mocked database operations.
"""

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.services.auth_service import AuthService


class TestAuthService:
    """Test AuthService class functionality"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    def test_init(self):
        """Test AuthService initialization"""
        mock_db = AsyncMock()
        service = AuthService(mock_db)
        assert service.db == mock_db

    @pytest.mark.asyncio
    async def test_get_user_by_id_success(self):
        """Test get_user_by_id successfully retrieves user"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_user = MagicMock()
        mock_user.id = "123e4567-e89b-12d3-a456-426614174000"
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result

        service = AuthService(mock_db)
        user = await service.get_user_by_id("123e4567-e89b-12d3-a456-426614174000")

        assert user == mock_user
        mock_db.execute.assert_called_once()
        # Verify the query was constructed correctly
        call_args = mock_db.execute.call_args[0][0]
        assert "users.id" in str(call_args) or "User.id" in str(call_args)

    @pytest.mark.asyncio
    async def test_get_user_by_id_not_found(self):
        """Test get_user_by_id returns None when user not found"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        service = AuthService(mock_db)
        user = await service.get_user_by_id("nonexistent-id")

        assert user is None
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_by_email_success(self):
        """Test get_user_by_email successfully retrieves user"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_user = MagicMock()
        mock_user.email = "test@example.com"
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result

        service = AuthService(mock_db)
        user = await service.get_user_by_email("test@example.com")

        assert user == mock_user
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_by_email_not_found(self):
        """Test get_user_by_email returns None when user not found"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        service = AuthService(mock_db)
        user = await service.get_user_by_email("nonexistent@example.com")

        assert user is None
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_by_username_success(self):
        """Test get_user_by_username successfully retrieves user"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_user = MagicMock()
        mock_user.username = "testuser"
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result

        service = AuthService(mock_db)
        user = await service.get_user_by_username("testuser")

        assert user == mock_user
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_by_username_not_found(self):
        """Test get_user_by_username returns None when user not found"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        service = AuthService(mock_db)
        user = await service.get_user_by_username("nonexistent")

        assert user is None
        mock_db.execute.assert_called_once()

    @patch('app.services.auth_service.verify_password')
    @pytest.mark.asyncio
    async def test_authenticate_user_success_by_username(self, mock_verify_password):
        """Test authenticate_user with valid username and password"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_user = MagicMock()
        mock_user.username = "testuser"
        mock_user.hashed_password = "hashed_password_123"
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result
        mock_verify_password.return_value = True

        service = AuthService(mock_db)
        user = await service.authenticate_user("testuser", "correct_password")

        assert user == mock_user
        mock_verify_password.assert_called_once_with("correct_password", "hashed_password_123")

    @patch('app.services.auth_service.verify_password')
    @pytest.mark.asyncio
    async def test_authenticate_user_success_by_email(self, mock_verify_password):
        """Test authenticate_user with valid email and password"""
        mock_db = AsyncMock()
        
        # Mock username lookup (returns None)
        mock_result_username = MagicMock()
        mock_result_username.scalar_one_or_none.return_value = None
        
        # Mock email lookup (returns user)
        mock_result_email = MagicMock()
        mock_user = MagicMock()
        mock_user.email = "test@example.com"
        mock_user.hashed_password = "hashed_password_123"
        mock_result_email.scalar_one_or_none.return_value = mock_user
        
        # Configure mock to return different results for username vs email queries
        mock_db.execute.side_effect = [mock_result_username, mock_result_email]
        mock_verify_password.return_value = True

        service = AuthService(mock_db)
        user = await service.authenticate_user("test@example.com", "correct_password")

        assert user == mock_user
        assert mock_db.execute.call_count == 2  # Once for username, once for email
        mock_verify_password.assert_called_once_with("correct_password", "hashed_password_123")

    @patch('app.services.auth_service.verify_password')
    @pytest.mark.asyncio
    async def test_authenticate_user_wrong_password(self, mock_verify_password):
        """Test authenticate_user with wrong password"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_user = MagicMock()
        mock_user.username = "testuser"
        mock_user.hashed_password = "hashed_password_123"
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result
        mock_verify_password.return_value = False

        service = AuthService(mock_db)
        user = await service.authenticate_user("testuser", "wrong_password")

        assert user is None
        mock_verify_password.assert_called_once_with("wrong_password", "hashed_password_123")

    @pytest.mark.asyncio
    async def test_authenticate_user_not_found(self):
        """Test authenticate_user with non-existent user"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        service = AuthService(mock_db)
        user = await service.authenticate_user("nonexistent", "password")

        assert user is None
        # Should try both username and email lookups
        assert mock_db.execute.call_count == 2

    @patch('app.services.auth_service.get_password_hash')
    @pytest.mark.asyncio
    async def test_create_user_success(self, mock_get_password_hash):
        """Test create_user successfully creates user"""
        mock_db = AsyncMock()
        mock_get_password_hash.return_value = "hashed_password_123"
        
        # Mock the user that gets created
        mock_user = MagicMock()
        mock_user.id = uuid.uuid4()
        mock_user.username = "newuser"
        mock_user.email = "new@example.com"

        service = AuthService(mock_db)
        user = await service.create_user(
            username="newuser",
            email="new@example.com",
            password="password123",
            first_name="New",
            last_name="User",
            is_active=True,
            is_superuser=False,
            is_verified=False
        )

        # Verify password was hashed
        mock_get_password_hash.assert_called_once_with("password123")
        
        # Verify database operations
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()
        
        # Verify the user object that was added
        added_user = mock_db.add.call_args[0][0]
        assert added_user.username == "newuser"
        assert added_user.email == "new@example.com"
        assert added_user.hashed_password == "hashed_password_123"
        assert added_user.first_name == "New"
        assert added_user.last_name == "User"
        assert added_user.is_active is True
        assert added_user.is_superuser is False
        assert added_user.is_verified is False

    @patch('app.services.auth_service.get_password_hash')
    @pytest.mark.asyncio
    async def test_create_user_with_defaults(self, mock_get_password_hash):
        """Test create_user with default values"""
        mock_db = AsyncMock()
        mock_get_password_hash.return_value = "hashed_password_123"

        service = AuthService(mock_db)
        user = await service.create_user(
            username="basicuser",
            email="basic@example.com",
            password="password123"
        )

        # Verify the user object with defaults
        added_user = mock_db.add.call_args[0][0]
        assert added_user.username == "basicuser"
        assert added_user.email == "basic@example.com"
        assert added_user.first_name is None
        assert added_user.last_name is None
        assert added_user.is_active is True  # default
        assert added_user.is_superuser is False  # default
        assert added_user.is_verified is False  # default

    @pytest.mark.asyncio
    async def test_update_user_profile_success(self):
        """Test update_user_profile successfully updates user"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.first_name = "Old"
        mock_user.last_name = "Name"
        mock_user.email = "old@example.com"
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result

        service = AuthService(mock_db)
        updated_user = await service.update_user_profile(
            user_id="user-123",
            first_name="New",
            last_name="Name",
            email="new@example.com"
        )

        assert updated_user == mock_user
        # Verify fields were updated
        assert mock_user.first_name == "New"
        assert mock_user.last_name == "Name"
        assert mock_user.email == "new@example.com"
        
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once_with(mock_user)

    @pytest.mark.asyncio
    async def test_update_user_profile_partial_update(self):
        """Test update_user_profile with partial updates"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.first_name = "Old"
        mock_user.last_name = "Name"
        mock_user.email = "old@example.com"
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result

        service = AuthService(mock_db)
        updated_user = await service.update_user_profile(
            user_id="user-123",
            first_name="Updated"
            # Only updating first name
        )

        assert updated_user == mock_user
        # Only first_name should be updated
        assert mock_user.first_name == "Updated"
        # Others should remain unchanged
        assert mock_user.last_name == "Name"
        assert mock_user.email == "old@example.com"

    @pytest.mark.asyncio
    async def test_update_user_profile_user_not_found(self):
        """Test update_user_profile returns None when user not found"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        service = AuthService(mock_db)
        updated_user = await service.update_user_profile(
            user_id="nonexistent",
            first_name="New"
        )

        assert updated_user is None
        mock_db.commit.assert_not_called()

    @patch('app.services.auth_service.get_password_hash')
    @pytest.mark.asyncio
    async def test_update_user_password_success(self, mock_get_password_hash):
        """Test update_user_password successfully updates password"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result
        mock_get_password_hash.return_value = "new_hashed_password"

        service = AuthService(mock_db)
        result = await service.update_user_password("user-123", "new_password")

        assert result is True
        assert mock_user.hashed_password == "new_hashed_password"
        mock_get_password_hash.assert_called_once_with("new_password")
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_user_password_user_not_found(self):
        """Test update_user_password returns False when user not found"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        service = AuthService(mock_db)
        result = await service.update_user_password("nonexistent", "new_password")

        assert result is False
        mock_db.commit.assert_not_called()

    @patch('app.services.auth_service.security_manager')
    @pytest.mark.asyncio
    async def test_request_password_reset_user_exists(self, mock_security_manager):
        """Test request_password_reset when user exists"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_user = MagicMock()
        mock_user.email = "test@example.com"
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result
        mock_security_manager.generate_password_reset_token.return_value = "reset_token_123"

        service = AuthService(mock_db)
        result = await service.request_password_reset("test@example.com")

        assert result is True
        mock_security_manager.generate_password_reset_token.assert_called_once_with("test@example.com")

    @patch('app.services.auth_service.security_manager')
    @pytest.mark.asyncio
    async def test_request_password_reset_user_not_found(self, mock_security_manager):
        """Test request_password_reset when user doesn't exist (no info leak)"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        service = AuthService(mock_db)
        result = await service.request_password_reset("nonexistent@example.com")

        # Should still return True to not leak user existence
        assert result is True
        # Should not generate token for non-existent user
        mock_security_manager.generate_password_reset_token.assert_not_called()

    @patch('app.services.auth_service.get_password_hash')
    @patch('app.services.auth_service.security_manager')
    @pytest.mark.asyncio
    async def test_reset_password_success(self, mock_security_manager, mock_get_password_hash):
        """Test reset_password with valid token"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_user = MagicMock()
        mock_user.email = "test@example.com"
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result
        
        mock_security_manager.verify_password_reset_token.return_value = "test@example.com"
        mock_get_password_hash.return_value = "new_hashed_password"

        service = AuthService(mock_db)
        result = await service.reset_password("valid_token", "new_password")

        assert result is True
        assert mock_user.hashed_password == "new_hashed_password"
        mock_security_manager.verify_password_reset_token.assert_called_once_with("valid_token")
        mock_get_password_hash.assert_called_once_with("new_password")
        mock_db.commit.assert_called_once()

    @patch('app.services.auth_service.security_manager')
    @pytest.mark.asyncio
    async def test_reset_password_invalid_token(self, mock_security_manager):
        """Test reset_password with invalid token"""
        mock_db = AsyncMock()
        mock_security_manager.verify_password_reset_token.return_value = None

        service = AuthService(mock_db)
        result = await service.reset_password("invalid_token", "new_password")

        assert result is False
        mock_db.execute.assert_not_called()
        mock_db.commit.assert_not_called()

    @patch('app.services.auth_service.security_manager')
    @pytest.mark.asyncio
    async def test_reset_password_user_not_found(self, mock_security_manager):
        """Test reset_password when user doesn't exist"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result
        mock_security_manager.verify_password_reset_token.return_value = "test@example.com"

        service = AuthService(mock_db)
        result = await service.reset_password("valid_token", "new_password")

        assert result is False
        mock_db.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_activate_user_success(self):
        """Test activate_user successfully activates user"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.is_active = False
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result

        service = AuthService(mock_db)
        result = await service.activate_user("user-123")

        assert result is True
        assert mock_user.is_active is True
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_activate_user_not_found(self):
        """Test activate_user returns False when user not found"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        service = AuthService(mock_db)
        result = await service.activate_user("nonexistent")

        assert result is False
        mock_db.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_deactivate_user_success(self):
        """Test deactivate_user successfully deactivates user"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.is_active = True
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result

        service = AuthService(mock_db)
        result = await service.deactivate_user("user-123")

        assert result is True
        assert mock_user.is_active is False
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_deactivate_user_not_found(self):
        """Test deactivate_user returns False when user not found"""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        service = AuthService(mock_db)
        result = await service.deactivate_user("nonexistent")

        assert result is False
        mock_db.commit.assert_not_called()


class TestAuthServiceIntegration:
    """Test AuthService integration scenarios"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @patch('app.services.auth_service.get_password_hash')
    @patch('app.services.auth_service.verify_password')
    @pytest.mark.asyncio
    async def test_user_registration_and_authentication_flow(self, mock_verify_password, mock_get_password_hash):
        """Test complete user registration and authentication flow"""
        mock_db = AsyncMock()
        mock_get_password_hash.return_value = "hashed_password_123"
        
        # Mock user creation
        service = AuthService(mock_db)
        
        # Create user
        await service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        # Verify user was added to database
        mock_db.add.assert_called_once()
        created_user = mock_db.add.call_args[0][0]
        assert created_user.username == "testuser"
        assert created_user.email == "test@example.com"
        assert created_user.hashed_password == "hashed_password_123"
        
        # Mock authentication
        mock_result = MagicMock()
        mock_user = MagicMock()
        mock_user.username = "testuser"
        mock_user.hashed_password = "hashed_password_123"
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result
        mock_verify_password.return_value = True
        
        # Authenticate user
        authenticated_user = await service.authenticate_user("testuser", "password123")
        
        assert authenticated_user == mock_user
        mock_verify_password.assert_called_once_with("password123", "hashed_password_123")