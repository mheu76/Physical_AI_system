# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands for Development

### Installation & Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install development dependencies  
pip install -r requirements-dev.txt

# Install HuggingFace requirements for PHI-3.5 models
pip install -r requirements-hf.txt

# Install interactive interface requirements
pip install -r requirements-interactive.txt

# Install with specific feature sets
pip install -e .[dev,simulation,visualization]

# Setup via setup.py
python setup.py install
```

### Running the System
```bash
# Run main system with specific mission
python main.py --mission "Pick up the red cup and place it on the table"

# Run with custom config
python main.py --config configs/modular.yaml

# Run continuous learning mode (no mission specified)
python main.py

# Run modular system
python modular_main.py
```

### Interactive Learning Interfaces
```bash
# Desktop GUI interface
python interactive_learning_interface.py

# Web-based interface (http://localhost:5001)
python enhanced_web_learning_interface.py

# Behavior model definition GUI
python behavior_model_gui.py

# Realtime feedback system
python realtime_feedback_system.py
```

### Examples and Testing
```bash
# Run basic example
python examples/basic_example.py

# PHI-3.5 specific demos
python examples/phi35_demo_example.py

# Developmental learning example  
python examples/developmental_learning_example.py

# SLM training example
python examples/slm_training_example.py

# Run integration tests
python -m pytest tests/test_integration.py

# Run modular system tests
python -m pytest tests/test_modular_system.py

# Run all tests with coverage
pytest --cov=. tests/
```

### Code Quality & Development Tools
```bash
# Code formatting
black .

# Linting
flake8 .

# Type checking
mypy .

# Run all tests with coverage
pytest --cov=. tests/

# Run specific foundation model tests
python test_foundation.py
python test_phi35_foundation.py
python test_phi35_simple.py

# Run cache fix test
python test_cache_fix.py
```

### Training and Model Operations
```bash
# Train SLM foundation model
python train_slm_foundation.py

# Test foundation model
python test_foundation.py

# Physics simulation
python simulation/physics_sim.py
```

## System Architecture

### Core Design Pattern
This is a **4-layer Physical AI system** implementing **Developmental Robotics** and **Embodied AI** principles:

1. **Foundation Model Layer** (`foundation_model/`)
   - PHI-3.5 small language model integration (`phi35_integration.py`)
   - SLM foundation with learning capabilities (`slm_foundation.py`) 
   - LLM learning module for continuous improvement (`llm_learning_module.py`)

2. **Developmental Learning Layer** (`developmental_learning/`)
   - Developmental engine for autonomous skill acquisition (`dev_engine.py`)
   - Curriculum-based learning progression
   - Memory management and experience replay

3. **AI Agent Execution Layer** (`ai_agent_execution/`)
   - Real-time motion control and safety monitoring
   - Physics-based action execution (`agent_executor.py`)

4. **Hardware Abstraction Layer** (`hardware_abstraction/`)
   - Unified interface for sensors and actuators (`hal_manager.py`)
   - Platform-agnostic hardware control

### Key System Components

#### Configuration Management
- **Centralized config system** via `core/config_manager.py`
- **YAML-based configuration** in `configs/` directory
- **Runtime validation** and hot-reloading support
- **Default config**: `configs/default.yaml` (production settings)
- **Modular config**: `configs/modular.yaml` (plugin-based setup)

#### Plugin Architecture  
- **Plugin system** in `plugins/` directory organized by function:
  - `plugins/foundation/` - Foundation model plugins
  - `plugins/learning/` - Learning algorithm plugins  
  - `plugins/execution/` - Execution strategy plugins
  - `plugins/hal/` - Hardware abstraction plugins
- **Plugin manager** (`core/plugin_manager.py`) handles discovery and lifecycle

#### Event-Driven Architecture
- **Event bus** (`core/event_bus.py`) for loose coupling between components
- **Mission broker** (`core/mission_broker.py`) orchestrates complex tasks
- **Asynchronous execution** throughout the system

#### Web Interface
- **Flask-based web interface** in `web_interface/`
- **Real-time learning dashboard** with progress visualization
- **Mission submission and monitoring**
- **System health and metrics display**

### Learning and AI Components

#### PHI-3.5 Integration
- **Primary model**: `microsoft/Phi-3.5-mini-instruct`
- **Device flexibility**: Auto-detection (CUDA/CPU/MPS)
- **Memory optimization**: FP16, quantization support
- **Learning capabilities**: Fine-tuning and continuous adaptation

#### Developmental Learning
- **Skill progression**: Simple → Complex skill acquisition
- **Memory systems**: Episodic, semantic, and working memory
- **Curriculum learning**: Structured learning stages defined in config
- **Meta-learning**: Learning how to learn new skills faster

#### Training Infrastructure
- **Distributed learning**: Ray cluster support for scaling
- **Model checkpointing**: Automatic saving/loading of trained models  
- **Performance tracking**: Comprehensive metrics and logging

### Data Flow Pattern

1. **Mission Input** → Foundation Model interprets natural language
2. **Task Planning** → Breaks down into executable subtasks  
3. **Skill Analysis** → Developmental Engine checks required skills
4. **Execution** → Agent Executor performs physical actions
5. **Learning** → Experience feeds back to improve future performance

### Important Implementation Details

#### Async/Await Pattern
The entire system is built on **asyncio** for:
- Non-blocking I/O operations
- Concurrent execution of learning and control loops
- Real-time responsiveness during physical execution

#### Safety and Validation
- **Config validation** with schemas and runtime checks
- **Hardware safety limits** enforced at multiple levels
- **Emergency stop** capabilities throughout execution chain
- **Mock hardware mode** for safe development/testing

#### Memory and Performance
- **Memory-efficient model loading** with device auto-selection
- **Caching systems** for frequently accessed data
- **Background processing** for learning tasks
- **Resource monitoring** and automatic cleanup

## Development Workflow

### Making Changes
1. **Configuration changes**: Edit files in `configs/` - auto-reloaded
2. **Plugin development**: Add to appropriate `plugins/` subdirectory  
3. **Core system changes**: Modify files in respective layer directories
4. **Testing**: Always run integration tests after significant changes

### Key Files to Understand
- `main.py` - System entry point and orchestration
- `modular_main.py` - Alternative modular system entry point
- `core/config_manager.py` - Configuration system
- `foundation_model/slm_foundation.py` - Core AI model with PHI-3.5 integration
- `foundation_model/phi35_integration.py` - PHI-3.5 specific implementation
- `developmental_learning/dev_engine.py` - Learning algorithms
- `configs/default.yaml` - System configuration reference
- `configs/modular.yaml` - Modular system configuration

### Plugin Development
Follow the plugin interface patterns in existing plugins. Each plugin type has specific base classes and hooks defined in the core system.

### Model Training
Use the provided training scripts as templates. The system supports both supervised fine-tuning and reinforcement learning approaches for different components.

### Important Language Support
This system includes Korean language support and localization:
- System language can be configured in `configs/default.yaml` with the `language` setting
- Localization is handled through `core/localization.py`
- Support for `ko` (Korean) and `en` (English) languages

### Working with Git
The repository uses Git for version control. Key branches and workflow patterns:
- Main development happens on the `main` branch
- Modified files are tracked, including `developmental_learning/dev_engine.py`
- Always check `git status` before making significant changes

### Entry Points and Console Scripts
The system provides multiple entry points configured in `setup.py`:
- `physical-ai` - Main system entry point
- `physical-ai-test` - Integration test runner
- `physical-ai-example` - Basic example runner

### Physical AI Code - Unified Interface
A new Claude Code-style unified interface has been implemented:

```bash
# Run the unified interface (Claude Code style)
python physical_ai_code.py

# Single mission execution
python physical_ai_code.py --mission "Pick up the red cup"

# With custom config
python physical_ai_code.py --config configs/modular.yaml --debug
```

**Key Features:**
- **Natural Language Interface**: Talk to the robot in Korean/English
- **Tool System**: All Physical AI functions as callable tools
- **Session Management**: Persistent conversation history
- **Command Processing**: Slash commands (/mission, /learn, /hardware, etc.)
- **Rich Terminal UI**: Beautiful terminal interface with Rich library

**Available Commands:**
- `/mission <task>` - Execute physical missions
- `/learn <skill>` - Start skill learning  
- `/hardware` - Check hardware status
- `/simulate <scenario>` - Run physics simulation
- `/vision` - Computer vision tasks
- `/status` - System status
- `/tools` - List available tools
- `/help` - Get help

**Natural Language Examples:**
- "로봇아, 빨간 컵을 테이블로 옮겨줘"
- "새로운 잡기 동작을 학습해줘"
- "시뮬레이션에서 테스트해보자"
- "하드웨어 상태를 확인해줘"