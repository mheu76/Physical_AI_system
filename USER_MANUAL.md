# Physical AI Code System User Manual

## 🤖 System Overview

Physical AI Code is a PHI-3.5-based Physical AI system providing a Claude Code-style unified interface. It enables natural language robot control, dynamic AI agent creation, and developmental learning.

## 🚀 Getting Started

### System Requirements
- Python 3.8 or higher
- Windows/Linux/macOS support
- Memory: Minimum 4GB RAM recommended

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/mheu76/Physical_AI_system.git
cd Physical_AI_system
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

3. **Test System**
```bash
python test_basic_functionality.py
```

## 📋 Basic Usage

### 1. Interactive Mode
```bash
python physical_ai_code.py
```
Starts interactive interface similar to Claude Code.

### 2. Single Mission Execution
```bash
python physical_ai_code.py --mission "Pick up the red cup and place it on the table"
```

### 3. Custom Configuration
```bash
python physical_ai_code.py --config configs/custom.yaml
```

## 🛠️ Core Features

### Slash Commands

#### `/status` - System Status Check
```
/status
```
Shows current system status and number of active tools.

#### `/tools` - Available Tools List
```
/tools
```
Displays 6 core tools with descriptions:
- **mission_executor**: Natural language mission execution
- **learning_system**: Learning and development system
- **hardware_status**: Hardware status monitoring
- **physics_simulation**: Physics simulation
- **vision_system**: Computer vision
- **agent_manager**: Dynamic agent management

#### `/agent` - Dynamic Agent Management

**Create Agent**
```
/agent create cleaning robot assistant
```

**List Agents**
```
/agent list
```

**Agent Information**
```
/agent info [agent_name]
```

**Update Agent**
```
/agent update [agent_name] make it work faster
```

**Delete Agent**
```
/agent delete [agent_name]
```

**Execute Agent**
```
/agent execute [agent_name] clean the desk
```

### Natural Language Mission Execution

The system understands and executes complex natural language commands:

```
Pick up the red cup and place it on the table
```

```
Look around with the camera and identify objects
```

```
Check the current position of the robot arm
```

## 🎯 Advanced Usage

### 1. Configuration Customization

Edit `configs/default.yaml` to modify system settings:

```yaml
foundation_model:
  phi35:
    model_name: "microsoft/Phi-3.5-mini-instruct"
    device: "auto"  # auto, cpu, cuda, mps
    temperature: 0.7

system:
  language: "en"  # ko, en
  mock_hardware: true  # Hardware simulation mode
```

### 2. Developer Mode

```bash
python physical_ai_code.py --debug --verbose
```

### 3. Web Interface

```bash
python enhanced_web_learning_interface.py
```
Access http://localhost:5001 in browser

## 📁 Directory Structure

```
Physical_AI_system/
├── physical_ai_code/           # Unified interface system
│   ├── core/                  # Core components
│   │   ├── interface_manager.py    # Main interface
│   │   ├── tool_system.py         # Tool system
│   │   ├── agent_system.py        # Agent management
│   │   └── command_processor.py   # Command processing
│   └── ui/                    # User interface
│       └── cli_interface.py       # CLI interface
├── agents/                    # Created agents storage
├── configs/                   # Configuration files
├── foundation_model/          # PHI-3.5 model system
├── developmental_learning/    # Developmental learning system
├── ai_agent_execution/       # Agent execution system
└── hardware_abstraction/     # Hardware abstraction
```

## 🔧 Troubleshooting

### Common Issues

**Q: System initializes in "limited mode"**
A: Normal behavior when hardware is not connected. All features work in simulation mode.

**Q: Unicode characters display incorrectly**
A: Terminal encoding issue. Use UTF-8 compatible terminal or set language to "en" in config.

**Q: PHI-3.5 model loading fails**
A: May be GPU memory shortage. Try setting `device: "cpu"` in configuration.

### Error Message Solutions

**TactileSensor initialization error**
```
ERROR: Can't instantiate abstract class TactileSensor
```
→ Warning in hardware simulation mode. Doesn't affect functionality.

**Model loading error**
```
ERROR: Failed to load PHI-3.5 model
```
→ Check internet connection and Hugging Face Hub accessibility.

## 📊 Performance Optimization

### 1. GPU Acceleration
```yaml
foundation_model:
  phi35:
    device: "cuda"  # Use NVIDIA GPU
```

### 2. Model Caching
Models are cached in `models/cache/` directory on first run.

### 3. Memory Optimization
```yaml
foundation_model:
  phi35:
    max_length: 1024  # Use smaller context length
```

## 🧪 Testing and Development

### Run Tests
```bash
# Basic functionality test
python test_basic_functionality.py

# Full test suite
pytest tests/ -v

# Test with coverage
pytest --cov=. tests/
```

### Development Tools
```bash
# Code formatting
black .

# Linting
flake8 .

# Type checking
mypy .
```

## 🔒 Security and Precautions

- System runs locally by default
- Check firewall settings for remote access
- Don't store sensitive data in `configs/` directory
- Avoid including personal information when creating agents

## 📞 Support and Contribution

- **GitHub Issues**: https://github.com/mheu76/Physical_AI_system/issues
- **Documentation**: See `CLAUDE.md` file
- **Development Guide**: See `README_UNIFIED_INTERFACE.md`

## 🚀 Extended Features

### Plugin Development
Add new features in the `plugins/` directory.

### ROS2 Integration
```bash
python modular_main.py --ros2
```

### Distributed Execution
```bash
python main.py --distributed --workers 4
```

---

## 📈 Version Information

**Current Version**: 2.0.0-phi35
**Last Updated**: 2025-08-25
**Compatibility**: Python 3.8+

For more detailed information, refer to the `CLAUDE.md` file.