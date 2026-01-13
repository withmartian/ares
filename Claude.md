# Claude Coding Guidelines for ARES

This document provides coding guidelines for Claude when working on the ARES codebase.

## Commenting Philosophy

### Comments Should Explain WHY, Not WHAT

The most valuable comments explain the reasoning behind a decision, not what the code is doing. Code should be self-explanatory through clear variable names, function names, and structure.

**Good Examples:**
```python
# Group by temperature to avoid mixing generation params across actors
grouped_requests = group_by_temperature(requests)

# SWE-Bench requires a fresh container per instance to ensure isolation
if not self._test_spec.is_remote_image:
    raise NotImplementedError("Need to implement local image support.")

# Queue mediates requests to allow linear code flow while exposing LLM interactions as observations/actions
llm_client = QueueMediatedLLMClient(q=asyncio.Queue())
```

**Bad Examples:**
```python
# Build an LLMResponse using the helper function
response = build_llm_response(data)

# Call the helper function
result = helper_function(input)

# Create a list
my_list = []
```

### When to Comment

- **WHY**: Explain architectural decisions, non-obvious design choices, or business logic
- **HOW**: For complex algorithms or non-trivial operations, explain the approach
- **NOT WHAT**: Avoid describing what the code obviously does

### Complex Code

For complex operations, explain HOW it works:

```python
# Parse test results by applying repo-specific log parser, then evaluate against
# FAIL_TO_PASS and PASS_TO_PASS criteria to determine resolution status
log_parser = MAP_REPO_TO_PARSER[repo]
test_status_map = log_parser(test_output, specs)
report = get_eval_tests_report(test_status_map, eval_instance)
```

## General Coding Guidelines

### Type Hints

- Use type hints for function parameters and return values
- Use `collections.abc` types for abstract collections (`Sequence`, `Mapping`, etc.)
- Prefer specific types over generic ones when possible

### Async/Await

- This codebase uses async extensively
- Always use `async def` and `await` for I/O operations
- Use `asyncio.create_task()` for concurrent operations

### Documentation Strings

- Use docstrings for classes and public functions
- Follow Google-style docstring format with Args, Returns, and Raises sections
- Include usage examples in class docstrings when helpful

### Code Organization

- Keep related functionality together in modules
- Use dataclasses for data structures (`@dataclasses.dataclass`)
- Use Pydantic models for external data validation (`pydantic.BaseModel`)

### Logging

- Use Python's `logging` module with module-level loggers
- Include contextual IDs (`id(self)`, `id(container)`) for tracking in concurrent operations
- Use appropriate log levels: DEBUG for detailed flow, INFO for key operations, WARNING/ERROR for issues

### Error Handling

- Raise descriptive exceptions with clear messages
- Use runtime checks for state validation (e.g., "Container has not been created before...")
- Add assertion messages for invariants: `assert condition, "explanation"`

### Naming Conventions

- Use descriptive names that make the code self-documenting
- Private methods and attributes: prefix with `_`
- Constants: `UPPER_SNAKE_CASE`
- Classes: `PascalCase`
- Functions/variables: `snake_case`

## Project-Specific Patterns

### Container Management

- Containers are async context managers
- Always start containers before use
- Use factory patterns for container creation

### Code Agents

- Code agents use the dm_env async spec
- LLM interactions are mediated through queues
- Agents run as asyncio tasks

### Testing

- Test specs come from SWE-bench harness
- Each SWE-bench instance requires a separate container
- Use repo-specific parsers for test output

## TODOs

Use TODO comments for:
- Missing implementations
- Future improvements
- Known issues that need addressing

Format: `# TODO: Description of what needs to be done.`
