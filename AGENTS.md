# AGENTS.md

## Project overview
- This repository provides sequence-length balancing algorithms in Rust with a Python ZeroMQ server/client wrapper.
- Main Rust code lives in `src/`; Python service code lives in `zmq_server/`.

## Build and test commands
- Rust build: `cargo build`
- Rust tests: `cargo test`
- Rust lint: `cargo clippy --all-targets --all-features`
- Python sanity check: `python -m compileall zmq_server`

## Code style guidelines
- Keep changes small and focused.
- Prefer explicit types and clear names.
- For Python code, add type hints and docstrings to functions so behavior is easy to verify.

## Testing instructions
- Run the smallest relevant tests first, then broader checks if needed.
- Include command output status in summaries.

## Security considerations
- Do not hardcode secrets or credentials.
- Validate and sanitize external input (especially JSON and network payloads).

## Extra instructions
- Commit messages should be concise and imperative (e.g., `Add AGENTS guidance`).
- PR descriptions should summarize what changed, why, and what checks were run.

## Nested AGENTS.md for monorepos
- Add nested `AGENTS.md` files in subprojects when rules need to differ.
- The closest `AGENTS.md` in the directory tree takes precedence.
