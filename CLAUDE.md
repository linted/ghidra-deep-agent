# CLAUDE.md

Always make changes in a new worktree unless explicitly asked to do it outside of a worktree. Never add yourself as a co-author of a commit. Don't overwrite the default `user.name` when making a commit.

## Python Development

After modifying any Python files, always run all three scripts in order:

```bash
./scripts/lint.sh
./scripts/typecheck.sh
./scripts/test.sh
```

Fix any errors reported before considering the task complete. These are the same
checks CI runs (`.github/workflows/ci.yml`), except CI uses `./scripts/lint.sh --check`
so it reports rather than rewrites.

`./scripts/test.sh` skips the `integration` tests, which need a live MongoDB with
vector search plus an embedding model. Run those explicitly when they are relevant:

```bash
uv run pytest -m integration
```
