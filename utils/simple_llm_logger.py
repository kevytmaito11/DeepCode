#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-simplified LLM response logger
Focuses on logging the core content of LLM replies, simple and easy to configure
"""

import json
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class SimpleLLMLogger:
    """Ultra-simplified LLM response logger"""

    def __init__(self, config_path: str = "mcp_agent.config.yaml"):
        """
        Initialize the logger

        Args:
            config_path: Path to config file
        """
        self.config = self._load_config(config_path)
        self.llm_config = self.config.get("llm_logger", {})

        # Return immediately if disabled
        if not self.llm_config.get("enabled", True):
            self.enabled = False
            return

        self.enabled = True
        self._setup_logger()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load config file"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ Failed to load config file: {e}, using default config")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config"""
        return {
            "llm_logger": {
                "enabled": True,
                "output_format": "json",
                "log_level": "basic",
                "log_directory": "logs/llm_responses",
                "filename_pattern": "llm_responses_{timestamp}.jsonl",
                "include_models": ["claude-sonnet-4", "gpt-4", "o3-mini"],
                "min_response_length": 50,
            }
        }

    def _setup_logger(self):
        """Set up logger"""
        log_dir = self.llm_config.get("log_directory", "logs/llm_responses")

        # Create log directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Generate log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_pattern = self.llm_config.get(
            "filename_pattern", "llm_responses_{timestamp}.jsonl"
        )
        self.log_file = os.path.join(
            log_dir, filename_pattern.format(timestamp=timestamp)
        )

        print(f"📝 LLM response log: {self.log_file}")

    def log_response(self, content: str, model: str = "", agent: str = "", **kwargs):
        """
        Log LLM response - simplified version

        Args:
            content: LLM response content
            model: Model name
            agent: Agent name
            **kwargs: Other optional info
        """
        if not self.enabled:
            return

        # Check if should log
        if not self._should_log(content, model):
            return

        # Build log entry
        log_entry = self._build_entry(content, model, agent, kwargs)

        # Write log
        self._write_log(log_entry)

        # Console display
        self._console_log(content, model, agent)

    def _should_log(self, content: str, model: str) -> bool:
        """Check if should log"""
        # Check length
        min_length = self.llm_config.get("min_response_length", 50)
        if len(content) < min_length:
            return False

        # Check model
        include_models = self.llm_config.get("include_models", [])
        if include_models and not any(m in model for m in include_models):
            return False

        return True

    def _build_entry(self, content: str, model: str, agent: str, extra: Dict) -> Dict:
        """Build log entry"""
        log_level = self.llm_config.get("log_level", "basic")

        if log_level == "basic":
            # Basic level: only log core content
            return {
                "timestamp": datetime.now().isoformat(),
                "content": content,
                "model": model,
            }
        else:
            # Detailed level: include more info
            entry = {
                "timestamp": datetime.now().isoformat(),
                "content": content,
                "model": model,
                "agent": agent,
            }
            # Add extra info
            if "token_usage" in extra:
                entry["tokens"] = extra["token_usage"]
            if "session_id" in extra:
                entry["session"] = extra["session_id"]
            return entry

    def _write_log(self, entry: Dict):
        """Write to log file"""
        output_format = self.llm_config.get("output_format", "json")

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                if output_format == "json":
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                elif output_format == "text":
                    timestamp = entry.get("timestamp", "")
                    model = entry.get("model", "")
                    content = entry.get("content", "")
                    f.write(f"[{timestamp}] {model}: {content}\n\n")
                elif output_format == "markdown":
                    timestamp = entry.get("timestamp", "")
                    model = entry.get("model", "")
                    content = entry.get("content", "")
                    f.write(f"**{timestamp}** | {model}\n\n{content}\n\n---\n\n")
        except Exception as e:
            print(f"⚠️ Failed to write log: {e}")

    def _console_log(self, content: str, model: str, agent: str):
        """Brief console display"""
        preview = content[:80] + "..." if len(content) > 80 else content
        print(f"🤖 {model} ({agent}): {preview}")


# Global instance
_global_logger = None


def get_llm_logger() -> SimpleLLMLogger:
    """Get global LLM logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = SimpleLLMLogger()
    return _global_logger


def log_llm_response(content: str, model: str = "", agent: str = "", **kwargs):
    """Convenience function: log LLM response"""
    logger = get_llm_logger()
    logger.log_response(content, model, agent, **kwargs)


# Example usage
if __name__ == "__main__":
    # Test logging
    log_llm_response(
        content="This is a test LLM response content to verify if the simplified logger works correctly.",
        model="claude-sonnet-4-20250514",
        agent="TestAgent",
    )

    print("✅ Simplified LLM log test completed")
