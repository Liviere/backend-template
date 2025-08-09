"""
Unit tests for app.core.logging_config module

Tests logging configuration setup with different debug settings
and JSON formatter functionality.
"""

import os
import logging
from unittest.mock import patch, MagicMock, call
import pytest

from app.core.logging_config import JsonFormatter, setup_logging


class TestJsonFormatter:
    """Test JsonFormatter functionality"""

    def setup_method(self, method):
        """Setup test environment"""
        self.formatter = JsonFormatter()

    def test_add_fields_with_timestamp(self):
        """Test JsonFormatter adds timestamp if not present"""
        log_record = {}
        record = MagicMock()
        record.created = 1640995200.0  # 2022-01-01 00:00:00 UTC
        record.levelname = "INFO"
        message_dict = {}

        self.formatter.add_fields(log_record, record, message_dict)

        assert log_record["timestamp"] == 1640995200.0
        assert log_record["level"] == "INFO"

    def test_add_fields_with_existing_timestamp(self):
        """Test JsonFormatter doesn't override existing timestamp"""
        log_record = {"timestamp": 1234567890.0}
        record = MagicMock()
        record.created = 1640995200.0
        record.levelname = "DEBUG"
        message_dict = {}

        self.formatter.add_fields(log_record, record, message_dict)

        # Should keep existing timestamp
        assert log_record["timestamp"] == 1234567890.0
        assert log_record["level"] == "DEBUG"

    def test_add_fields_with_existing_level(self):
        """Test JsonFormatter uses existing level if present"""
        log_record = {"level": "warning"}
        record = MagicMock()
        record.created = 1640995200.0
        record.levelname = "DEBUG"
        message_dict = {}

        self.formatter.add_fields(log_record, record, message_dict)

        # Should uppercase existing level
        assert log_record["level"] == "WARNING"

    def test_add_fields_uses_record_levelname(self):
        """Test JsonFormatter uses record.levelname when level not in log_record"""
        log_record = {}
        record = MagicMock()
        record.created = 1640995200.0
        record.levelname = "ERROR"
        message_dict = {}

        self.formatter.add_fields(log_record, record, message_dict)

        assert log_record["level"] == "ERROR"


class TestSetupLogging:
    """Test setup_logging function"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from app.core.config import get_settings
        get_settings.cache_clear()

    @patch('app.core.logging_config.dictConfig')
    def test_setup_logging_debug_mode(self, mock_dict_config):
        """Test setup_logging with debug=True sets DEBUG level"""
        with patch('app.core.config.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.debug = True
            mock_get_settings.return_value = mock_settings

            setup_logging()

            # Verify dictConfig was called
            mock_dict_config.assert_called_once()
            config = mock_dict_config.call_args[0][0]

            # Check that log level is DEBUG
            assert config["loggers"]["uvicorn"]["level"] == "DEBUG"
            assert config["loggers"]["fastapi"]["level"] == "DEBUG"
            assert config["loggers"]["app"]["level"] == "DEBUG"
            assert config["root"]["level"] == "DEBUG"

    @patch('app.core.logging_config.dictConfig')  
    def test_setup_logging_production_mode(self, mock_dict_config):
        """Test setup_logging with debug=False sets INFO level"""
        # Since the logging config setup is already called, we need to isolate the test better
        with patch('app.core.logging_config.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.debug = False
            mock_get_settings.return_value = mock_settings

            setup_logging()

            # Verify dictConfig was called
            mock_dict_config.assert_called_once()
            config = mock_dict_config.call_args[0][0]

            # Check that log level is INFO
            assert config["loggers"]["uvicorn"]["level"] == "INFO"
            assert config["loggers"]["fastapi"]["level"] == "INFO"
            assert config["loggers"]["app"]["level"] == "INFO"
            assert config["root"]["level"] == "INFO"

    @patch('app.core.logging_config.dictConfig')
    def test_setup_logging_config_structure(self, mock_dict_config):
        """Test setup_logging creates expected configuration structure"""
        with patch('app.core.config.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.debug = True
            mock_get_settings.return_value = mock_settings

            setup_logging()

            config = mock_dict_config.call_args[0][0]

            # Verify configuration structure
            assert config["version"] == 1
            assert config["disable_existing_loggers"] is False

            # Check formatters
            assert "default" in config["formatters"]
            assert "json" in config["formatters"]
            assert config["formatters"]["json"]["()"] == JsonFormatter

            # Check handlers
            assert "console" in config["handlers"]
            assert "file" in config["handlers"]
            assert config["handlers"]["console"]["class"] == "logging.StreamHandler"
            assert config["handlers"]["file"]["class"] == "logging.handlers.TimedRotatingFileHandler"
            assert config["handlers"]["file"]["filename"] == "logs/app.log"

            # Check loggers
            required_loggers = ["uvicorn", "fastapi", "app"]
            for logger_name in required_loggers:
                assert logger_name in config["loggers"]
                assert "console" in config["loggers"][logger_name]["handlers"]
                assert "file" in config["loggers"][logger_name]["handlers"]
                assert config["loggers"][logger_name]["propagate"] is False

            # Check root logger
            assert "console" in config["root"]["handlers"]
            assert "file" in config["root"]["handlers"]

    @patch('app.core.logging_config.dictConfig')
    def test_setup_logging_file_handler_config(self, mock_dict_config):
        """Test file handler configuration in setup_logging"""
        with patch('app.core.config.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.debug = False
            mock_get_settings.return_value = mock_settings

            setup_logging()

            config = mock_dict_config.call_args[0][0]
            file_handler = config["handlers"]["file"]

            assert file_handler["formatter"] == "json"
            assert file_handler["filename"] == "logs/app.log"
            assert file_handler["when"] == "midnight"
            assert file_handler["interval"] == 1
            assert file_handler["backupCount"] == 30
            assert file_handler["encoding"] == "utf-8"

    @patch('app.core.logging_config.dictConfig')
    def test_setup_logging_console_handler_config(self, mock_dict_config):
        """Test console handler configuration in setup_logging"""
        import sys
        
        with patch('app.core.config.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.debug = True
            mock_get_settings.return_value = mock_settings

            setup_logging()

            config = mock_dict_config.call_args[0][0]
            console_handler = config["handlers"]["console"]

            assert console_handler["class"] == "logging.StreamHandler"
            assert console_handler["formatter"] == "default"
            assert console_handler["stream"] == sys.stdout

    def test_integration_with_actual_logger(self, caplog):
        """Test integration with actual logging to verify handler works"""
        # Test that logging works - caplog captures at pytest level, not handler level
        test_logger = logging.getLogger("app.test")
        
        # Test that logging works
        with caplog.at_level(logging.DEBUG):
            test_logger.debug("Test debug message")
            test_logger.info("Test info message")
            test_logger.error("Test error message")

        # Verify messages were captured by caplog
        assert any("Test debug message" in record.message for record in caplog.records)
        assert any("Test info message" in record.message for record in caplog.records)
        assert any("Test error message" in record.message for record in caplog.records)

    def test_json_formatter_integration(self):
        """Test JsonFormatter integration in actual logging"""
        formatter = JsonFormatter()
        
        # Create a test log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Verify it's JSON-like format (contains expected fields)
        assert "timestamp" in formatted or str(record.created) in formatted
        assert "INFO" in formatted
        assert "Test message" in formatted
        # Note: logger name might not be in formatted output depending on format string