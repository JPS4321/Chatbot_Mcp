# Chatbot MCP Client

This repository contains a chatbot client that connects to an **MCP (Model Context Protocol) server** over stdio.
It uses OpenAI’s API to interpret user messages, select tools exposed by the MCP server, and provide structured, user-friendly responses.

---

## Features

* Connects to a remote MCP server defined in `servers.yaml`.
* Loads tools dynamically and routes user queries to the most relevant tool.
* Keeps **conversation context** (`chat_context.json`) across interactions:

  * Stores previous tool arguments.
  * Stores the last list output to support follow-ups like *“the first one”*.
* Maintains a **chat log** (`chatbot_log.txt`) for auditing.
* Supports **ordinal references** (e.g., *first, second, #3*).
* Provides fallback answers if no tool matches.

---

## Requirements

* Python 3.10+
* OpenAI API key

Dependencies are listed in `requirements.txt`, and include:

* `openai`
* `python-dotenv`
* `prompt_toolkit`
* `pyyaml`
* `mcp`

---

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # Linux/Mac
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment variables**

   * Create a `.env` file in the project root:

     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   * You can get your API key from [OpenAI](https://platform.openai.com/).

5. **Configure servers**

   * Edit `servers.yaml` and define your MCP server, for example:

     ```yaml
     servers:
       remote_echo:
         command: "python"
         args: ["path/to/your/server.py"]
         env: {}
     ```

---

## Usage

Run the chatbot:

```bash
python host.py
```

You should see output similar to:

```
Connected to MCP server: remote_echo
🛠️ Available tools: tool1, tool2, tool3
Commands: tools | context | exit
```

Then you can interact:

```
> what is the top game
```

The bot will:

1. Parse your message.
2. Decide which tool to call.
3. Print a step-by-step trace.
4. Return a final user-friendly answer.

---

## Context & Logs

* **Context file:** `chat_context.json`
  Stores tool history, last arguments, and last list of entities.

* **Log file:** `chatbot_log.txt`
  Contains a raw log of all interactions.

To reset context, simply delete `chat_context.json` or restart the client (it overwrites the file on startup).

---

## Exiting

At any prompt, type:

```
exit
```

to close the session.

---
