"""
Authentication Service

Business logic for user authentication, registration, and session management.
"""

import logging
import threading
from pathlib import Path
from typing import Optional
from uuid import UUID

import jinja2
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from inference_core.core.config import get_settings
from inference_core.core.security import (
    get_password_hash,
    security_manager,
    verify_password,
)
from inference_core.database.sql.models.user import User
from inference_core.observability.sentry_scrubbing import redact_pii_text
from inference_core.schemas.auth import RegisterRequest

# Import email functionality with fallback
try:
    from inference_core.celery.tasks.email_tasks import (
        encode_attachment,
        send_email_async,
    )
    from inference_core.services.email_service import get_email_service

    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class AuthService:
    """Authentication service for user management"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_id(self, user_id) -> Optional[User]:
        """
        Get user by ID

        Args:
            user_id: User ID (string or UUID)

        Returns:
            User instance or None
        """
        # Convert to UUID if string (for SQLAlchemy compatibility)
        if isinstance(user_id, str):
            try:
                user_id = UUID(user_id)
            except ValueError:
                # If it's not a valid UUID string, still try the query to maintain backward compatibility
                # This handles cases like "nonexistent-id" which tests expect to be passed to the database
                pass

        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email

        Args:
            email: User email

        Returns:
            User instance or None
        """
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username

        Args:
            username: Username

        Returns:
            User instance or None
        """
        result = await self.db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username/email and password

        Args:
            username: Username or email
            password: Password

        Returns:
            User instance if authenticated, None otherwise
        """
        # Try to find user by username or email
        user = await self.get_user_by_username(username)
        if not user:
            user = await self.get_user_by_email(username)

        if not user:
            return None

        if not verify_password(password, user.hashed_password):
            return None

        return user

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        is_active: bool = True,
        is_superuser: bool = False,
        is_verified: bool = False,
    ) -> User:
        """
        Create new user

        Args:
            username: Username
            email: Email
            password: Password
            first_name: First name
            last_name: Last name
            is_active: Whether user is active
            is_superuser: Whether user is superuser
            is_verified: Whether email is verified

        Returns:
            Created user instance
        """
        hashed_password = get_password_hash(password)

        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            first_name=first_name,
            last_name=last_name,
            is_active=is_active,
            is_superuser=is_superuser,
            is_verified=is_verified,
        )

        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def update_user_profile(
        self,
        user_id: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        email: Optional[str] = None,
    ) -> Optional[User]:
        """
        Update user profile

        Args:
            user_id: User ID
            first_name: New first name
            last_name: New last name
            email: New email

        Returns:
            Updated user instance or None
        """
        user = await self.get_user_by_id(user_id)
        if not user:
            return None

        if first_name is not None:
            user.first_name = first_name
        if last_name is not None:
            user.last_name = last_name
        if email is not None:
            user.email = email

        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def update_user_password(self, user_id: str, new_password: str) -> bool:
        """
        Update user password

        Args:
            user_id: User ID
            new_password: New password

        Returns:
            True if updated successfully
        """
        user = await self.get_user_by_id(user_id)
        if not user:
            return False

        user.hashed_password = get_password_hash(new_password)
        await self.db.commit()

        return True

    async def request_password_reset(self, email: str) -> bool:
        """
        Request password reset

        Args:
            email: User email

        Returns:
            True if reset token generated
        """
        user = await self.get_user_by_email(email)
        if not user:
            # Don't reveal if email exists
            return True

        # Generate reset token
        reset_token = security_manager.generate_password_reset_token(email)

        # Send password reset email
        try:
            await self._send_password_reset_email(email, reset_token)
            # Log the internal id, never the address (PII in log sinks).
            logger.info("Password reset email sent (user_id=%s)", user.id)
        except Exception as e:
            logger.error(
                "Failed to send password reset email (user_id=%s): %s",
                user.id,
                redact_pii_text(str(e)),
            )
            # Still return True to not reveal if email exists
            # In production, you might want to retry or alert admins

        return True

    async def _send_password_reset_email(self, email: str, reset_token: str):
        """
        Send password reset email using email service

        Args:
            email: User email address
            reset_token: Password reset token
        """
        if not EMAIL_AVAILABLE:
            # Never emit the address or the token — a logged reset token is a
            # working account-takeover link for anyone who can read the logs.
            logger.warning(
                "Email functionality not available; password reset token "
                "generated but NOT delivered"
            )
            return

        # Build reset URL
        settings = get_settings()
        app_url = getattr(settings, "app_public_url", "http://localhost:8000")
        reset_url = f"{app_url}/reset-password?token={reset_token}"

        # Template variables
        template_vars = {
            "reset_url": reset_url,
            "expiry_hours": 24,  # Based on token expiration in security.py
            "email": email,
        }

        # Render email templates
        text_content = self._render_template("reset_password.txt", template_vars)
        html_content = self._render_template("reset_password.html", template_vars)

        # Try to send via Celery first (async), fallback to direct sending
        try:
            if EMAIL_AVAILABLE:
                # Send asynchronously via Celery
                task = send_email_async(
                    to=email,
                    subject="Password Reset Request - Inference Core",
                    text=text_content,
                    html=html_content,
                )
                logger.info(f"Password reset email task queued: {task.id}")
            else:
                raise ImportError("Email functionality not available")

        except Exception as e:
            logger.warning(f"Failed to queue email via Celery: {e}, trying direct send")

            # Fallback: send email directly in a thread to avoid blocking
            email_service = get_email_service()
            if email_service:

                def send_direct():
                    try:
                        message_id = email_service.send_email(
                            to=email,
                            subject="Password Reset Request - Inference Core",
                            text=text_content,
                            html=html_content,
                        )
                        logger.info(f"Password reset email sent directly: {message_id}")
                    except Exception as e:
                        logger.error(
                            "Failed to send password reset email directly: %s",
                            redact_pii_text(str(e)),
                        )

                thread = threading.Thread(target=send_direct, daemon=True)
                thread.start()
            else:
                logger.warning(
                    "Email service not available; password reset token "
                    "generated but NOT delivered"
                )

    def _render_template(self, template_name: str, variables: dict) -> str:
        """
        Render email template with Jinja2

        Args:
            template_name: Template file name
            variables: Template variables

        Returns:
            Rendered template content
        """
        try:
            # Get template directory
            template_dir = Path(__file__).parent.parent.parent / "templates" / "email"

            # Set up Jinja2 environment
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(template_dir),
                autoescape=jinja2.select_autoescape(["html", "xml"]),
            )

            # Load and render template
            template = env.get_template(template_name)
            return template.render(**variables)

        except Exception as e:
            logger.error(f"Failed to render email template {template_name}: {e}")
            # Fallback to basic text
            if "reset_url" in variables:
                return f"Password reset link: {variables['reset_url']}"
            return "Password reset requested. Please check with administrator."

    async def reset_password(self, token: str, new_password: str) -> bool:
        """
        Reset password with token

        Args:
            token: Reset token
            new_password: New password

        Returns:
            True if password reset successfully
        """
        email = security_manager.verify_password_reset_token(token)
        if not email:
            return False

        user = await self.get_user_by_email(email)
        if not user:
            return False

        user.hashed_password = get_password_hash(new_password)
        await self.db.commit()

        return True

    async def deactivate_user(self, user_id: str) -> bool:
        """
        Deactivate user

        Args:
            user_id: User ID

        Returns:
            True if deactivated successfully
        """
        user = await self.get_user_by_id(user_id)
        if not user:
            return False

        user.is_active = False
        await self.db.commit()

        return True

    async def activate_user(self, user_id: str) -> bool:
        """
        Activate user

        Args:
            user_id: User ID

        Returns:
            True if activated successfully
        """
        user = await self.get_user_by_id(user_id)
        if not user:
            return False

        user.is_active = True
        await self.db.commit()

        return True

    def create_email_verification_token(self, user_id: str) -> str:
        """
        Create email verification token for user

        Args:
            user_id: User ID

        Returns:
            Email verification token
        """
        return security_manager.generate_email_verification_token(user_id)

    async def verify_email_with_token(self, token: str) -> bool:
        """
        Verify user email with verification token

        Args:
            token: Email verification token

        Returns:
            True if verification successful, False otherwise
        """
        user_id_str = security_manager.verify_email_verification_token(token)
        if not user_id_str:
            return False

        user = await self.get_user_by_id(user_id_str)
        if not user:
            return False

        settings = get_settings()
        if settings.auth_email_verification_makes_active:
            # Activate user if setting enabled
            user.is_active = True

        # Set user as verified (idempotent - no error if already verified)
        user.is_verified = True
        await self.db.commit()

        return True

    async def send_verification_email(self, user, verification_token: str):
        """
        Send email verification email using email service

        Args:
            user: User object with email and name fields
            verification_token: Email verification token
        """
        if not EMAIL_AVAILABLE:
            # Never emit the address or the token itself — a logged token is a
            # working verification link for anyone who can read the logs.
            logger.warning(
                "Email functionality not available; verification token "
                "generated but NOT delivered (user_id=%s)",
                user.id,
            )
            return

        # Build verification URL
        settings = get_settings()
        if settings.auth_email_verification_url_base:
            # Use frontend URL if configured
            verify_url = f"{settings.auth_email_verification_url_base}?token={verification_token}"
        else:
            # Use backend endpoint if no frontend URL configured
            app_url = getattr(settings, "app_public_url", "http://localhost:8000")
            verify_url = (
                f"{app_url}/api/v1/auth/verify-email?token={verification_token}"
            )

        # Template variables
        template_vars = {
            "verify_url": verify_url,
            "user_name": user.full_name,
            "email": user.email,
            "expiry_minutes": settings.auth_email_verification_token_ttl_minutes,
        }

        # Render email templates
        text_content = self._render_template("verify_email.txt", template_vars)
        html_content = self._render_template("verify_email.html", template_vars)

        # Send email via Celery task
        send_email_async(
            to=user.email,
            subject="Verify your email address",
            text=text_content,
            html=html_content,
        )

    async def request_verification_email(self, email: str) -> bool:
        """
        Request verification email for user by email address

        Args:
            email: User email address

        Returns:
            True if verification email sent (always returns True to not reveal if email exists)
        """
        user = await self.get_user_by_email(email)
        if not user:
            # Don't reveal if email exists
            return True

        # Generate verification token
        verification_token = self.create_email_verification_token(str(user.id))

        # Send verification email
        try:
            await self.send_verification_email(user, verification_token)
            logger.info("Email verification email sent (user_id=%s)", user.id)
        except Exception as e:
            logger.error(
                "Failed to send verification email (user_id=%s): %s",
                user.id,
                redact_pii_text(str(e)),
            )
            # Still return True to not reveal if email exists

        return True
