## Code Style

- Write clean, simple, readable code
- Small functions and classes — each does one thing
- No comments or docstrings unless the logic is genuinely non-obvious
- All imports at the top of the file (lazy imports only when justified by performance)
- Prefer explicit over implicit

## Principles

- **OOP**: Use classes when modeling state/behavior; prefer composition over inheritance
- **SOLID**: Single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion
- **KISS**: Simplest solution that works — no premature abstraction
- **DRY**: Extract only when duplication is real (rule of three), not speculative
- **YAGNI**: Don't build what isn't needed yet

## Structure

- Keep files short and focused — one class/concern per file
- Group by feature/domain, not by type (avoid generic `utils/`, `helpers/`)
- Flat is better than nested — minimize directory depth

## Practices

- Fail fast: validate inputs early, raise clear errors
- Use meaningful names — if a name needs a comment, rename it
- Prefer pure functions where possible
- Type hints / type annotations always (Python, TS, etc.)
- Return early to avoid deep nesting
- No magic numbers — use named constants

## Testing

- Test behavior, not implementation
- One assertion per logical concept
- Arrange-Act-Assert structure
- No test should depend on another test

## When Generating Code

- Don't wrap in unnecessary try/catch unless handling a specific failure
- Don't add logging unless asked
- Don't over-engineer: no factories, builders, or abstractions without clear need
- Prefer stdlib/builtins before reaching for third-party libraries
