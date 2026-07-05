# CLAUDE.md

Always make changes in a new worktree unless explicitly asked to do it outside of a worktree. Never add yourself as a co-author of a commit.

## Python Development

After modifying any Python files, always run both scripts in order:

```bash
./scripts/lint.sh
./scripts/typecheck.sh
```

Fix any errors reported before considering the task complete.
