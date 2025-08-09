"""
Unit tests for app.services.task_service module

Tests TaskService for managing Celery tasks with proper mocking.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from app.services.task_service import TaskService, get_task_service


class TestTaskService:
    """Test TaskService class functionality"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @patch('app.services.task_service.celery_app')
    def test_init(self, mock_celery_app):
        """Test TaskService initialization"""
        service = TaskService()
        # The service should use the mocked celery_app
        assert service.celery_app == mock_celery_app

    @patch('app.services.task_service.AsyncResult')
    @patch('app.services.task_service.celery_app')
    def test_get_task_status(self, mock_celery_app, mock_async_result_class):
        """Test get_task_status returns complete status information"""
        # Mock AsyncResult instance
        mock_result = MagicMock()
        mock_result.status = "SUCCESS"
        mock_result.result = {"message": "Task completed"}
        mock_result.info = {"progress": 100}
        mock_result.traceback = None
        mock_result.ready.return_value = True
        mock_result.successful.return_value = True
        mock_result.failed.return_value = False
        
        mock_async_result_class.return_value = mock_result

        service = TaskService()
        status = service.get_task_status("test-task-id")

        assert status["task_id"] == "test-task-id"
        assert status["status"] == "SUCCESS"
        assert status["result"] == {"message": "Task completed"}
        assert status["info"] == {"progress": 100}
        assert status["traceback"] is None
        assert status["successful"] is True
        assert status["failed"] is False

        mock_async_result_class.assert_called_once_with("test-task-id", app=mock_celery_app)

    @patch('app.services.task_service.AsyncResult')
    @patch('app.services.task_service.celery_app')
    def test_get_task_status_pending(self, mock_celery_app, mock_async_result_class):
        """Test get_task_status with pending task"""
        mock_result = MagicMock()
        mock_result.status = "PENDING"
        mock_result.result = None
        mock_result.info = None
        mock_result.traceback = None
        mock_result.ready.return_value = False
        mock_result.successful.return_value = None  # Not ready
        mock_result.failed.return_value = None  # Not ready
        
        mock_async_result_class.return_value = mock_result

        service = TaskService()
        status = service.get_task_status("pending-task-id")

        assert status["task_id"] == "pending-task-id"
        assert status["status"] == "PENDING"
        assert status["result"] is None
        assert status["successful"] is None
        assert status["failed"] is None

    @patch('app.services.task_service.AsyncResult')
    @patch('app.services.task_service.celery_app')
    def test_get_task_status_failed(self, mock_celery_app, mock_async_result_class):
        """Test get_task_status with failed task"""
        mock_result = MagicMock()
        mock_result.status = "FAILURE"
        mock_result.result = Exception("Task failed")
        mock_result.info = {"error": "Something went wrong"}
        mock_result.traceback = "Traceback details..."
        mock_result.ready.return_value = True
        mock_result.successful.return_value = False
        mock_result.failed.return_value = True
        
        mock_async_result_class.return_value = mock_result

        service = TaskService()
        status = service.get_task_status("failed-task-id")

        assert status["task_id"] == "failed-task-id"
        assert status["status"] == "FAILURE"
        assert status["traceback"] == "Traceback details..."
        assert status["successful"] is False
        assert status["failed"] is True

    @patch('app.services.task_service.AsyncResult')
    @patch('app.services.task_service.celery_app')
    def test_get_task_result(self, mock_celery_app, mock_async_result_class):
        """Test get_task_result retrieves task result"""
        mock_result = MagicMock()
        mock_result.get.return_value = {"data": "task result"}
        
        mock_async_result_class.return_value = mock_result

        service = TaskService()
        result = service.get_task_result("test-task-id")

        assert result == {"data": "task result"}
        mock_result.get.assert_called_once_with(timeout=None)

    @patch('app.services.task_service.AsyncResult')
    @patch('app.services.task_service.celery_app')
    def test_get_task_result_with_timeout(self, mock_celery_app, mock_async_result_class):
        """Test get_task_result with custom timeout"""
        mock_result = MagicMock()
        mock_result.get.return_value = {"data": "task result"}
        
        mock_async_result_class.return_value = mock_result

        service = TaskService()
        result = service.get_task_result("test-task-id", timeout=30)

        assert result == {"data": "task result"}
        mock_result.get.assert_called_once_with(timeout=30)

    @patch('app.services.task_service.AsyncResult')
    @patch('app.services.task_service.celery_app')
    def test_cancel_task(self, mock_celery_app, mock_async_result_class):
        """Test cancel_task revokes task with termination"""
        mock_result = MagicMock()
        mock_async_result_class.return_value = mock_result

        service = TaskService()
        result = service.cancel_task("test-task-id")

        assert result is True
        mock_result.revoke.assert_called_once_with(terminate=True)

    @patch('app.services.task_service.celery_app')
    def test_get_active_tasks(self, mock_celery_app):
        """Test get_active_tasks returns worker task information"""
        mock_inspect = MagicMock()
        mock_inspect.active.return_value = {"worker1": [{"task": "task1"}]}
        mock_inspect.scheduled.return_value = {"worker1": [{"task": "task2"}]}
        mock_inspect.reserved.return_value = {"worker1": [{"task": "task3"}]}
        
        mock_celery_app.control.inspect.return_value = mock_inspect

        service = TaskService()
        tasks = service.get_active_tasks()

        assert tasks["active"] == {"worker1": [{"task": "task1"}]}
        assert tasks["scheduled"] == {"worker1": [{"task": "task2"}]}
        assert tasks["reserved"] == {"worker1": [{"task": "task3"}]}

    @patch('app.services.task_service.celery_app')
    def test_get_active_tasks_with_none_response(self, mock_celery_app):
        """Test get_active_tasks handles None responses from inspect"""
        mock_inspect = MagicMock()
        mock_inspect.active.return_value = None
        mock_inspect.scheduled.return_value = None
        mock_inspect.reserved.return_value = None
        
        mock_celery_app.control.inspect.return_value = mock_inspect

        service = TaskService()
        tasks = service.get_active_tasks()

        # Should still return a dict structure even with None responses
        assert tasks["active"] is None
        assert tasks["scheduled"] is None
        assert tasks["reserved"] is None

    @patch('app.services.task_service.celery_app')
    def test_get_worker_stats(self, mock_celery_app):
        """Test get_worker_stats returns worker statistics"""
        mock_inspect = MagicMock()
        mock_inspect.stats.return_value = {"worker1": {"total": 10}}
        mock_inspect.ping.return_value = {"worker1": "pong"}
        mock_inspect.registered.return_value = {"worker1": ["task.name"]}
        
        mock_celery_app.control.inspect.return_value = mock_inspect

        service = TaskService()
        stats = service.get_worker_stats()

        assert stats["stats"] == {"worker1": {"total": 10}}
        assert stats["ping"] == {"worker1": "pong"}
        assert stats["registered"] == {"worker1": ["task.name"]}

    @patch('app.services.task_service.celery_app')
    def test_get_worker_stats_with_none_response(self, mock_celery_app):
        """Test get_worker_stats handles None responses from inspect"""
        mock_inspect = MagicMock()
        mock_inspect.stats.return_value = None
        mock_inspect.ping.return_value = None
        mock_inspect.registered.return_value = None
        
        mock_celery_app.control.inspect.return_value = mock_inspect

        service = TaskService()
        stats = service.get_worker_stats()

        # Should still return a dict structure even with None responses
        assert stats["stats"] is None
        assert stats["ping"] is None
        assert stats["registered"] is None

    @patch('app.services.task_service.celery_app')
    def test_explain_async(self, mock_celery_app):
        """Test explain_async submits explanation task"""
        mock_task = MagicMock()
        mock_task.id = "explain-task-123"
        mock_celery_app.send_task.return_value = mock_task

        service = TaskService()
        task_id = service.explain_async(query="What is AI?", model="gpt-3.5")

        assert task_id == "explain-task-123"
        mock_celery_app.send_task.assert_called_once_with(
            "llm.explain",
            kwargs={"query": "What is AI?", "model": "gpt-3.5"}
        )

    @patch('app.services.task_service.celery_app')
    def test_explain_async_no_kwargs(self, mock_celery_app):
        """Test explain_async without arguments"""
        mock_task = MagicMock()
        mock_task.id = "explain-task-456"
        mock_celery_app.send_task.return_value = mock_task

        service = TaskService()
        task_id = service.explain_async()

        assert task_id == "explain-task-456"
        mock_celery_app.send_task.assert_called_once_with(
            "llm.explain",
            kwargs={}
        )

    @patch('app.services.task_service.celery_app')
    def test_conversation_async(self, mock_celery_app):
        """Test conversation_async submits conversation task"""
        mock_task = MagicMock()
        mock_task.id = "conversation-task-789"
        mock_celery_app.send_task.return_value = mock_task

        service = TaskService()
        task_id = service.conversation_async(message="Hello", session_id="session-123")

        assert task_id == "conversation-task-789"
        mock_celery_app.send_task.assert_called_once_with(
            "llm.conversation",
            kwargs={"message": "Hello", "session_id": "session-123"}
        )

    @patch('app.services.task_service.celery_app')
    def test_conversation_async_no_kwargs(self, mock_celery_app):
        """Test conversation_async without arguments"""
        mock_task = MagicMock()
        mock_task.id = "conversation-task-000"
        mock_celery_app.send_task.return_value = mock_task

        service = TaskService()
        task_id = service.conversation_async()

        assert task_id == "conversation-task-000"
        mock_celery_app.send_task.assert_called_once_with(
            "llm.conversation",
            kwargs={}
        )


class TestGetTaskService:
    """Test get_task_service function"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @patch('app.services.task_service.task_service')
    def test_get_task_service_returns_global_instance(self, mock_task_service):
        """Test get_task_service returns global task service instance"""
        result = get_task_service()
        assert result == mock_task_service


class TestTaskServiceErrorHandling:
    """Test TaskService error handling scenarios"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @patch('app.services.task_service.AsyncResult')
    @patch('app.services.task_service.celery_app')
    def test_get_task_result_timeout_error(self, mock_celery_app, mock_async_result_class):
        """Test get_task_result when timeout occurs"""
        from celery.exceptions import TimeoutError
        
        mock_result = MagicMock()
        mock_result.get.side_effect = TimeoutError("Task timed out")
        
        mock_async_result_class.return_value = mock_result

        service = TaskService()
        
        with pytest.raises(TimeoutError):
            service.get_task_result("test-task-id", timeout=1)

    @patch('app.services.task_service.AsyncResult')
    @patch('app.services.task_service.celery_app')
    def test_get_task_result_task_failure(self, mock_celery_app, mock_async_result_class):
        """Test get_task_result when task failed"""
        mock_result = MagicMock()
        mock_result.get.side_effect = Exception("Task execution failed")
        
        mock_async_result_class.return_value = mock_result

        service = TaskService()
        
        with pytest.raises(Exception, match="Task execution failed"):
            service.get_task_result("failed-task-id")

    @patch('app.services.task_service.celery_app')
    def test_get_active_tasks_inspect_exception(self, mock_celery_app):
        """Test get_active_tasks when inspect methods raise exceptions"""
        mock_inspect = MagicMock()
        mock_inspect.active.side_effect = Exception("Connection failed")
        mock_inspect.scheduled.return_value = {}
        mock_inspect.reserved.return_value = {}
        
        mock_celery_app.control.inspect.return_value = mock_inspect

        service = TaskService()
        
        # Should handle exception gracefully
        with pytest.raises(Exception):
            service.get_active_tasks()

    @patch('app.services.task_service.celery_app')
    def test_send_task_connection_error(self, mock_celery_app):
        """Test task submission when broker is unavailable"""
        mock_celery_app.send_task.side_effect = Exception("Broker connection failed")

        service = TaskService()
        
        with pytest.raises(Exception, match="Broker connection failed"):
            service.explain_async(query="test")


class TestTaskServiceIntegration:
    """Test TaskService integration scenarios"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @patch('app.services.task_service.AsyncResult')
    @patch('app.services.task_service.celery_app')
    def test_task_lifecycle(self, mock_celery_app, mock_async_result_class):
        """Test complete task lifecycle: submit, check status, get result, cancel"""
        # Mock task submission
        mock_submit_task = MagicMock()
        mock_submit_task.id = "lifecycle-task-123"
        mock_celery_app.send_task.return_value = mock_submit_task
        
        # Mock task result
        mock_result = MagicMock()
        mock_result.status = "SUCCESS"
        mock_result.result = {"answer": "AI is artificial intelligence"}
        mock_result.ready.return_value = True
        mock_result.successful.return_value = True
        mock_result.failed.return_value = False
        mock_result.get.return_value = {"answer": "AI is artificial intelligence"}
        
        mock_async_result_class.return_value = mock_result

        service = TaskService()
        
        # Submit task
        task_id = service.explain_async(query="What is AI?")
        assert task_id == "lifecycle-task-123"
        
        # Check status
        status = service.get_task_status(task_id)
        assert status["status"] == "SUCCESS"
        assert status["successful"] is True
        
        # Get result
        result = service.get_task_result(task_id)
        assert result == {"answer": "AI is artificial intelligence"}
        
        # Cancel (even though it's completed)
        cancelled = service.cancel_task(task_id)
        assert cancelled is True
        
        # Verify all operations were called
        mock_celery_app.send_task.assert_called_once()
        mock_async_result_class.assert_called()
        mock_result.revoke.assert_called_once_with(terminate=True)