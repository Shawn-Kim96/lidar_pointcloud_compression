# Gemini Agent Setup for This Research Repo (2026-02-27)

## What files matter
- `GEMINI.md`
  - project memory and default collaboration behavior.
- `.gemini/settings.json`
  - project-level Gemini settings (context loading behavior).
- `.gemini/prompts/*.md`
  - reusable role/context prompt blocks.
- `.gemini/commands/**/*.toml`
  - custom slash commands for role switching.
- `~/.gemini/settings.json`
  - user-level MCP server registration.

## Added in this repo
- Project memory:
  - `GEMINI.md`
- Prompt library:
  - `.gemini/prompts/context_lidar_compression_research.md`
  - `.gemini/prompts/persona_senior_research_scientist.md`
  - `.gemini/prompts/role_experiment_director.md`
  - `.gemini/prompts/role_paper_reviewer_response.md`
  - `.gemini/prompts/role_result_interpretation_debugger.md`
- Slash commands:
  - `/role:senior_scientist`
  - `/role:experiment_director`
  - `/role:paper_reviewer_response`
  - `/role:result_debugger`

## How to use in Gemini CLI
1. Restart Gemini CLI after config edits.
2. Run `/memory refresh`.
3. Run `/tools` to check MCP tool availability.
4. Use role command, then ask your task:
   - `/role:senior_scientist "design a fair Stage2 ablation"`

## MCP plan applied
- Scholar access:
  - Use browser MCP to access `https://scholar.google.com`.
- PDF reading:
  - Use PDF MCP server tools for PDF inspection/extraction.

## Global MCP config applied
- File:
  - `~/.gemini/settings.json`
- Registered servers:
  - `scholar_browser`:
    - `npx -y @playwright/mcp --headless --caps pdf,vision`
  - `pdf_reader`:
    - `npx -y mcp-pdf`
- Quick verification in Gemini CLI:
  1. restart Gemini CLI
  2. run `/tools`
  3. confirm both `scholar_browser` and `pdf_reader` tools are visible

## Notes
- Keep Track-A and Track-B claims separated in any generated writing.
- For Track-B, always require non-zero original AP sanity before reconstructed comparisons.
