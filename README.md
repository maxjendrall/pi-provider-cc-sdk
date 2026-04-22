# pi-provider-cc-sdk

Use your Claude subscription with pi using Anthropic's official [Claude Code SDK](https://github.com/anthropics/claude-agent-sdk-typescript).

> Based on [pi-claude-bridge](https://github.com/elidickinson/pi-claude-bridge) by Eli Dickinson.

## Setup

```bash
pi install npm:pi-provider-cc-sdk
```

Requires `claude` CLI installed and logged in. Then `/reload` in pi.

## Usage

Use `/model` to select one of:

- `claude-sdk/claude-opus-4-7`
- `claude-sdk/claude-opus-4-6`
- `claude-sdk/claude-sonnet-4-6`
- `claude-sdk/claude-haiku-4-5`

## Prompt strategy

This provider now defaults to a safer prompt composition path for Claude subscription auth:

- keep the Claude Code preset
- strip Claude dynamic sections from the system prompt
- append a safe subset of Pi's prompt
- append nearest `AGENTS.md`
- append Pi skills block
- keep Pi documentation out of the always-on system prompt

The Pi documentation block is intentionally excluded because it can trigger Anthropic's subscription `extra usage` error on the Claude SDK path.

## Prompt modes

Control the append strategy with `PI_CLAUDE_SDK_PROMPT_MODE`:

- `safe-pi` default: safe Pi base + nearest `AGENTS.md` + skills block
- `safe-pi-base`: safe Pi base only
- `safe-pi-no-skills`: safe Pi base + nearest `AGENTS.md`
- `safe-pi-no-agents`: safe Pi base + skills block
- `preset-only`: Claude Code preset only

## Recommended docs strategy

Instead of appending large documentation blocks into the Claude SDK system prompt, load docs on demand through Pi skills or by reading the installed Pi docs from disk when the user asks Pi-specific questions.
