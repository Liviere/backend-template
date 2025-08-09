"""
Unit tests for app.services.llm_service module

Tests LLMService for managing LLM operations with mocked chains and models.
"""

import os
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.services.llm_service import LLMService, LLMResponse, LLMMetadata, get_llm_service


class TestLLMMetadata:
    """Test LLMMetadata model"""

    def test_llm_metadata_creation(self):
        """Test LLMMetadata creation"""
        metadata = LLMMetadata(
            model_name="test-model",
            timestamp="2023-01-01T00:00:00Z"
        )
        assert metadata.model_name == "test-model"
        assert metadata.timestamp == "2023-01-01T00:00:00Z"


class TestLLMResponse:
    """Test LLMResponse model"""

    def test_llm_response_creation(self):
        """Test LLMResponse creation"""
        metadata = LLMMetadata(
            model_name="test-model",
            timestamp="2023-01-01T00:00:00Z"
        )
        response = LLMResponse(
            result={"answer": "Test answer"},
            metadata=metadata
        )
        assert response.result == {"answer": "Test answer"}
        assert response.metadata.model_name == "test-model"


class TestLLMService:
    """Test LLMService class functionality"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @patch('app.services.llm_service.get_model_factory')
    @patch('app.services.llm_service.llm_config')
    def test_init(self, mock_llm_config, mock_get_model_factory):
        """Test LLMService initialization"""
        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory

        service = LLMService()

        assert service.config == mock_llm_config  # Uses the mocked config directly
        assert service.model_factory == mock_factory
        assert service._usage_stats["requests_count"] == 0
        assert service._usage_stats["total_tokens"] == 0
        assert service._usage_stats["errors_count"] == 0
        assert service._usage_stats["last_request"] is None

    @patch('app.services.llm_service.create_explanation_chain')
    @patch('app.services.llm_service.get_model_factory')
    @patch('app.services.llm_service.llm_config')
    @pytest.mark.asyncio
    async def test_explain_success(self, mock_llm_config, mock_get_model_factory, mock_create_chain):
        """Test explain method successful execution"""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.enable_monitoring = False
        mock_llm_config = mock_config
        
        # Mock model factory
        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory
        
        # Mock chain
        mock_chain = AsyncMock()
        mock_chain.model_name = "test-model"
        mock_chain.generate_story.return_value = "Test explanation"
        mock_create_chain.return_value = mock_chain

        service = LLMService()
        result = await service.explain("What is AI?")

        assert isinstance(result, LLMResponse)
        assert result.result["answer"] == "Test explanation"
        assert result.metadata.model_name == "test-model"
        assert service._usage_stats["requests_count"] == 1
        
        mock_create_chain.assert_called_once_with(model_name=None)
        mock_chain.generate_story.assert_called_once_with(question="What is AI?")

    @patch('app.services.llm_service.create_explanation_chain')
    @patch('app.services.llm_service.get_model_factory')
    @patch('app.services.llm_service.llm_config')
    @pytest.mark.asyncio
    async def test_explain_with_model_params(self, mock_llm_config, mock_get_model_factory, mock_create_chain):
        """Test explain method with custom model parameters"""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.enable_monitoring = False
        mock_llm_config = mock_config
        
        # Mock model factory
        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory
        
        # Mock chain
        mock_chain = AsyncMock()
        mock_chain.model_name = "custom-model"
        mock_chain.generate_story.return_value = "Custom explanation"
        mock_create_chain.return_value = mock_chain

        service = LLMService()
        result = await service.explain(
            "What is AI?",
            model_name="custom-model",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )

        assert result.result["answer"] == "Custom explanation"
        assert result.metadata.model_name == "custom-model"
        
        mock_create_chain.assert_called_once_with(
            model_name="custom-model",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )

    @patch('app.services.llm_service.create_explanation_chain')
    @patch('app.services.llm_service.get_model_factory')
    @patch('app.services.llm_service.llm_config')
    @pytest.mark.asyncio
    async def test_explain_chain_error(self, mock_llm_config, mock_get_model_factory, mock_create_chain):
        """Test explain method when chain raises error"""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.enable_monitoring = False
        mock_llm_config = mock_config
        
        # Mock model factory
        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory
        
        # Mock chain that raises error
        mock_chain = AsyncMock()
        mock_chain.generate_story.side_effect = Exception("Model error")
        mock_create_chain.return_value = mock_chain

        service = LLMService()
        
        with pytest.raises(Exception, match="Model error"):
            await service.explain("What is AI?")
        
        # Should increment error count
        assert service._usage_stats["errors_count"] == 1
        assert service._usage_stats["requests_count"] == 1

    @patch('app.services.llm_service.create_conversation_chain')
    @patch('app.services.llm_service.get_model_factory')
    @patch('app.services.llm_service.llm_config')
    @pytest.mark.asyncio
    async def test_converse_success(self, mock_llm_config, mock_get_model_factory, mock_create_chain):
        """Test converse method successful execution"""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.enable_monitoring = False
        mock_llm_config = mock_config
        
        # Mock model factory
        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory
        
        # Mock chain
        mock_chain = AsyncMock()
        mock_chain.model_name = "conversation-model"
        mock_chain.chat.return_value = "Hello! How can I help you?"
        mock_create_chain.return_value = mock_chain

        service = LLMService()
        result = await service.converse("session-123", "Hello")

        assert isinstance(result, LLMResponse)
        assert result.result["reply"] == "Hello! How can I help you?"
        assert result.result["session_id"] == "session-123"
        assert result.metadata.model_name == "conversation-model"
        assert service._usage_stats["requests_count"] == 1
        
        mock_create_chain.assert_called_once_with(model_name=None)
        mock_chain.chat.assert_called_once_with(session_id="session-123", user_input="Hello")

    @patch('app.services.llm_service.create_conversation_chain')
    @patch('app.services.llm_service.get_model_factory')
    @patch('app.services.llm_service.llm_config')
    @pytest.mark.asyncio
    async def test_converse_with_model_params(self, mock_llm_config, mock_get_model_factory, mock_create_chain):
        """Test converse method with custom model parameters"""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.enable_monitoring = False
        mock_llm_config = mock_config
        
        # Mock model factory
        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory
        
        # Mock chain
        mock_chain = AsyncMock()
        mock_chain.model_name = "custom-chat-model"
        mock_chain.chat.return_value = "Custom response"
        mock_create_chain.return_value = mock_chain

        service = LLMService()
        result = await service.converse(
            "session-456",
            "Tell me a story",
            model_name="custom-chat-model",
            temperature=0.8,
            max_tokens=200,
            request_timeout=30
        )

        assert result.result["reply"] == "Custom response"
        assert result.result["session_id"] == "session-456"
        
        mock_create_chain.assert_called_once_with(
            model_name="custom-chat-model",
            temperature=0.8,
            max_tokens=200,
            timeout=30  # request_timeout maps to timeout
        )

    @patch('app.services.llm_service.create_conversation_chain')
    @patch('app.services.llm_service.get_model_factory')
    @patch('app.services.llm_service.llm_config')
    @pytest.mark.asyncio
    async def test_converse_chain_error(self, mock_llm_config, mock_get_model_factory, mock_create_chain):
        """Test converse method when chain raises error"""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.enable_monitoring = False
        mock_llm_config = mock_config
        
        # Mock model factory
        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory
        
        # Mock chain that raises error
        mock_chain = AsyncMock()
        mock_chain.chat.side_effect = Exception("Conversation error")
        mock_create_chain.return_value = mock_chain

        service = LLMService()
        
        with pytest.raises(Exception, match="Conversation error"):
            await service.converse("session-123", "Hello")
        
        # Should increment error count
        assert service._usage_stats["errors_count"] == 1

    @patch('app.services.llm_service.get_model_factory')
    @patch('app.services.llm_service.llm_config')
    def test_get_available_models(self, mock_llm_config, mock_get_model_factory):
        """Test get_available_models method"""
        # Mock configuration
        mock_config = MagicMock()
        mock_llm_config = mock_config
        
        # Mock model factory
        mock_factory = MagicMock()
        mock_factory.get_available_models.return_value = {
            "gpt-3.5-turbo": True,
            "gpt-4": False
        }
        mock_get_model_factory.return_value = mock_factory

        service = LLMService()
        models = service.get_available_models()

        assert models == {"gpt-3.5-turbo": True, "gpt-4": False}
        mock_factory.get_available_models.assert_called_once()

    @patch('app.services.llm_service.get_model_factory')
    @patch('app.services.llm_service.llm_config')
    def test_get_usage_stats(self, mock_llm_config, mock_get_model_factory):
        """Test get_usage_stats method"""
        # Mock configuration
        mock_config = MagicMock()
        mock_llm_config = mock_config
        
        # Mock model factory
        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory

        service = LLMService()
        
        # Modify stats to test
        service._usage_stats["requests_count"] = 5
        service._usage_stats["errors_count"] = 1
        service._usage_stats["last_request"] = "2023-01-01T00:00:00Z"
        
        stats = service.get_usage_stats()

        assert stats["requests_count"] == 5
        assert stats["errors_count"] == 1
        assert stats["last_request"] == "2023-01-01T00:00:00Z"
        
        # Should return a copy, not the original
        stats["requests_count"] = 100
        assert service._usage_stats["requests_count"] == 5

    @patch('app.services.llm_service.logger')
    @patch('app.services.llm_service.get_model_factory')
    @patch('app.services.llm_service.llm_config')
    def test_log_request_with_monitoring(self, mock_llm_config, mock_get_model_factory, mock_logger):
        """Test _log_request method with monitoring enabled"""
        # Mock configuration with monitoring enabled
        mock_config = MagicMock()
        mock_config.enable_monitoring = True
        mock_llm_config = mock_config
        
        # Mock model factory
        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory

        service = LLMService()
        service._log_request("test_operation", {"param1": "value1", "param2": "value2"})

        mock_logger.info.assert_called_once_with(
            "LLM operation: test_operation, params: {'param1': 'value1', 'param2': 'value2'}"
        )

    @patch('app.services.llm_service.logger')
    @patch('app.services.llm_service.get_model_factory')
    @patch('app.services.llm_service.llm_config')
    def test_log_request_without_monitoring(self, mock_llm_config, mock_get_model_factory, mock_logger):
        """Test _log_request method with monitoring disabled"""
        # Mock configuration with monitoring disabled
        mock_llm_config.enable_monitoring = False
        
        # Mock model factory
        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory

        service = LLMService()
        service._log_request("test_operation", {"param1": "value1"})

        # Should not log when monitoring is disabled
        mock_logger.info.assert_not_called()

    @patch('app.services.llm_service.get_model_factory')
    @patch('app.services.llm_service.llm_config')
    def test_update_usage_stats_success(self, mock_llm_config, mock_get_model_factory):
        """Test _update_usage_stats method for successful operation"""
        # Mock configuration
        mock_config = MagicMock()
        mock_llm_config = mock_config
        
        # Mock model factory
        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory

        service = LLMService()
        initial_requests = service._usage_stats["requests_count"]
        initial_errors = service._usage_stats["errors_count"]
        
        service._update_usage_stats(success=True)

        assert service._usage_stats["requests_count"] == initial_requests + 1
        assert service._usage_stats["errors_count"] == initial_errors
        assert service._usage_stats["last_request"] is not None

    @patch('app.services.llm_service.get_model_factory')
    @patch('app.services.llm_service.llm_config')
    def test_update_usage_stats_failure(self, mock_llm_config, mock_get_model_factory):
        """Test _update_usage_stats method for failed operation"""
        # Mock configuration
        mock_config = MagicMock()
        mock_llm_config = mock_config
        
        # Mock model factory
        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory

        service = LLMService()
        initial_requests = service._usage_stats["requests_count"]
        initial_errors = service._usage_stats["errors_count"]
        
        service._update_usage_stats(success=False)

        assert service._usage_stats["requests_count"] == initial_requests + 1
        assert service._usage_stats["errors_count"] == initial_errors + 1

    @patch('app.services.llm_service.logger')
    @patch('app.services.llm_service.get_model_factory')
    @patch('app.services.llm_service.llm_config')
    def test_handle_error(self, mock_llm_config, mock_get_model_factory, mock_logger):
        """Test _handle_error method"""
        # Mock configuration
        mock_config = MagicMock()
        mock_llm_config = mock_config
        
        # Mock model factory
        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory

        service = LLMService()
        error = Exception("Test error")
        initial_errors = service._usage_stats["errors_count"]
        
        service._handle_error("test_operation", error)

        # Should log error and update stats
        mock_logger.error.assert_called_once_with(
            "LLM operation 'test_operation' failed: Test error"
        )
        assert service._usage_stats["errors_count"] == initial_errors + 1


class TestGetLLMService:
    """Test get_llm_service function"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @patch('app.services.llm_service.llm_service')
    def test_get_llm_service_returns_global_instance(self, mock_llm_service):
        """Test get_llm_service returns global service instance"""
        result = get_llm_service()
        assert result == mock_llm_service


class TestLLMServiceIntegration:
    """Test LLMService integration scenarios"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @patch('app.services.llm_service.create_explanation_chain')
    @patch('app.services.llm_service.create_conversation_chain')
    @patch('app.services.llm_service.get_model_factory')
    @patch('app.services.llm_service.llm_config')
    @pytest.mark.asyncio
    async def test_multiple_operations_update_stats(
        self, mock_llm_config, mock_get_model_factory, mock_create_conv_chain, mock_create_exp_chain
    ):
        """Test that multiple operations update usage statistics correctly"""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.enable_monitoring = False
        mock_llm_config = mock_config
        
        # Mock model factory
        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory
        
        # Mock explanation chain
        mock_exp_chain = AsyncMock()
        mock_exp_chain.model_name = "exp-model"
        mock_exp_chain.generate_story.return_value = "Explanation"
        mock_create_exp_chain.return_value = mock_exp_chain
        
        # Mock conversation chain
        mock_conv_chain = AsyncMock()
        mock_conv_chain.model_name = "conv-model"
        mock_conv_chain.chat.return_value = "Chat response"
        mock_create_conv_chain.return_value = mock_conv_chain

        service = LLMService()
        
        # Perform multiple operations
        await service.explain("What is AI?")
        await service.converse("session-1", "Hello")
        
        try:
            # Simulate a failure
            mock_exp_chain.generate_story.side_effect = Exception("Error")
            await service.explain("Another question")
        except Exception:
            pass  # Expected
        
        # Check usage statistics
        stats = service.get_usage_stats()
        assert stats["requests_count"] == 3
        assert stats["errors_count"] == 1
        assert stats["last_request"] is not None