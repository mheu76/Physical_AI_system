"""
Physical AI System

발달적 학습과 체화된 지능을 구현하는 Physical AI 시스템
"""

__version__ = "1.0.0"
__author__ = "Physical AI Team"
__email__ = "team@physical-ai.com"

from .foundation_model.slm_foundation import SLMFoundation
from .developmental_learning.dev_engine import DevelopmentalEngine
from .ai_agent_execution.agent_executor import AgentExecutor
from .hardware_abstraction.hal_manager import HardwareManager

__all__ = [
    "SLMFoundation",
    "DevelopmentalEngine", 
    "AgentExecutor",
    "HardwareManager"
]
