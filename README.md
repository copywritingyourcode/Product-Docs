# AI Assistant Swarm

A powerful, extensible AI assistant system that uses a swarm of specialized agents to provide high-quality responses across different domains.

## Overview

The AI Assistant Swarm is a local-first AI assistant platform that leverages multiple specialized agents to handle different types of queries. The system uses an orchestrator to route queries to the most appropriate specialist agent, ensuring high-quality, domain-specific responses.

Key features:
- 🧠 Multiple specialized agents (Medical, Python, General)
- 🔄 Intelligent query routing via an orchestrator
- 💾 Persistent memory with vector storage
- 📄 Document upload and retrieval
- 🔍 Memory search capabilities
- 🖥️ Multiple interfaces (CLI and Web)
- 🚀 Local-first architecture using Ollama

## Requirements

- Python 3.9+
- [Ollama](https://ollama.ai/) installed with the following models:
  - `gemma:27b` (Medical Specialist)
  - `deepseek-coder:33b` (Python Developer)
  - `qwen:14b` (General Assistant & Orchestrator)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-assistant-swarm.git
cd ai-assistant-swarm
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Make sure Ollama is running and the required models are installed:
```bash
ollama pull gemma:27b
ollama pull deepseek-coder:33b
ollama pull qwen:14b
```

## Usage

### Command Line Interface

To start the AI Assistant Swarm with the CLI interface:

```bash
    python -m swarm --interface cli
```

Available commands in the CLI:
- `/help` - Display help information
- `/exit` - Exit the application
- `/clear` - Clear the current conversation
- `/new` - Start a new conversation
- `/list` - List all saved conversations
- `/load <id>` - Load a conversation by ID
- `/agent <type>` - Switch to a specific agent (medical, python, fallback)
- `/upload <file_path>` - Upload a file to the assistant's memory
- `/memory <query>` - Search the assistant's memory
- `/models` - Check the status of required models

### Web Interface

To start the AI Assistant Swarm with the web interface:

```bash
python -m swarm --interface web
```

Optional parameters:
- `--host` - Host address (default: 127.0.0.1)
- `--port` - Port number (default: 7860)
- `--share` - Create a shareable link
- `--no-streaming` - Disable response streaming
- `--hide-agent` - Hide agent names in responses
- `--data-dir` - Custom data directory path
- `--log-level` - Logging level (default: info)

## Architecture

The AI Assistant Swarm consists of several key components:

1. **SwarmManager**: The central coordinator that manages agents and memory.

2. **Agents**:
   - **OrchestratorAgent**: Routes queries to the appropriate specialist agent.
   - **MedicalAgent**: Handles health and medical queries.
   - **PythonAgent**: Specializes in Python programming questions.
   - **FallbackAgent**: Handles general queries that don't fit other specialists.

3. **Memory System**:
   - **VectorStore**: Stores and retrieves documents using semantic search.
   - **RetentionManager**: Manages document retention policies.

4. **Interfaces**:
   - **CommandLineInterface**: Terminal-based interface.
   - **WebInterface**: Browser-based interface using Gradio.

## Development

### Running Tests

To run the test suite:

```bash
python -m unittest discover tests
```

### Adding a New Agent

To add a new specialist agent:

1. Create a new agent class in `swarm/agents/` that inherits from `BaseAgent`.
2. Implement the `generate_response` method.
3. Register the agent in `SwarmManager._initialize_agents`.

Example:
```python
from swarm.agents.base import BaseAgent

class MyNewAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="My New Specialist",
            model_name="model-name:version",
            system_prompt="You are a specialist in..."
        )
    
    async def generate_response(self, query, retrieve_from_memory=True):
        # Implementation here
        return response
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project uses [Ollama](https://ollama.ai/) for running local LLMs.
- The web interface is built with [Gradio](https://gradio.app/).
- Vector storage is implemented using [FAISS](https://github.com/facebookresearch/faiss).
