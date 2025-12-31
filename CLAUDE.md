# Claude Code Rules for abide

## Commit Messages

**NEVER include Claude attribution in commit messages.** Do not add:
- `Co-Authored-By: Claude <noreply@anthropic.com>`
- `🤖 Generated with [Claude Code](https://claude.com/claude-code)`
- Any other reference to Claude or AI generation

Commit messages should be clean and professional, describing only what was changed and why.

## Git Hooks

A `commit-msg` hook is installed in `.git/hooks/` that will reject commits containing Claude attribution.
