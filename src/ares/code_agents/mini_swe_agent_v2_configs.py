"""Configuration presets for MiniSWECodeAgentV2.

This module provides pre-configured settings for different benchmarks and use cases.
"""

import dataclasses

# Default environment variables for text-based mode
_DEFAULT_ENV_VARS = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}


# Text-based templates (generic workflow)
_TEXTBASED_SYSTEM_TEMPLATE = """
You are a helpful assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
Your reasoning and analysis here. Explain why you want to perform the action.

```mswea_bash_command
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected.
""".strip()


_TEXTBASED_INSTANCE_TEMPLATE = """
Please solve this issue: {{task}}

You can execute bash commands and edit files to implement the necessary changes.

## Recommended Workflow

This workflows should be done step-by-step so that you can iterate on your changes and any possible problems.

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust
6. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
   Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

## Important Rules

1. Every response must contain exactly one action
2. The action must be enclosed in triple backticks
3. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
   However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

<system_information>
{{system}} {{release}} {{version}} {{machine}}
</system_information>

## Formatting your response

Here is an example of a correct response:

<example_response>
THOUGHT: I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

```mswea_bash_command
ls -la
```
</example_response>

## Useful command examples

### Create a new file:

```mswea_bash_command
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:

{%- if system == "Darwin" -%}
<important>
You are on MacOS. For all the below examples, you need to use `sed -i ''` instead of `sed -i`.
</important>
{%- endif -%}

```mswea_bash_command
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:

```mswea_bash_command
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```mswea_bash_command
anything
```
""".strip()  # noqa: E501


_SHARED_OBSERVATION_TEMPLATE = """
{% if output.exception_info -%}
<exception>{{output.exception_info}}</exception>
{% endif -%}
<returncode>{{output.returncode}}</returncode>
{% if output.output | length < 10000 -%}
<output>
{{ output.output -}}
</output>
{%- else -%}
<warning>
The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
If you're using grep or find and it produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.
</warning>
{%- set elided_chars = output.output | length - 10000 -%}
<output_head>
{{ output.output[:5000] }}
</output_head>
<elided_chars>
{{ elided_chars }} characters elided
</elided_chars>
<output_tail>
{{ output.output[-5000:] }}
</output_tail>
{%- endif -%}
""".strip()  # noqa: E501


_TEXTBASED_FORMAT_ERROR_TEMPLATE = """
Format error:

<error>
{{error}}
</error>

Here is general guidance on how to format your response:

Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions.

Please format your action in triple backticks as shown in <response_example>.

<response_example>
Here are some thoughts about why you want to perform the action.

```mswea_bash_command
<action>
```
</response_example>

If you have completed your assignment, please consult the first message about how to
submit your solution (you will not be able to continue working on this task after that).
""".strip()


# SWE-bench Verified templates
_SWEBENCH_SYSTEM_TEMPLATE = """
You are a helpful assistant that can interact with a computer shell to solve programming tasks.
""".strip()


_SWEBENCH_INSTANCE_TEMPLATE = """
<pr_description>
Consider the following PR description:
{{task}}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.
<IMPORTANT>This is an interactive process where you will think and issue AT LEAST ONE command, see the result, then think and issue your next command(s).</important>

For each response:

1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide one or more bash code blocks to execute

## Important Boundaries

- MODIFY: Regular source code files in /testbed (this is the working directory for all your subsequent commands)
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Command Execution Rules

You are operating in an environment where

1. You issue at least one command
3. The system executes the command(s) in a subshell
4. You see the result(s)
5. You write your next command(s)

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. At least one code block with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE bash code block. You can make MULTIPLE code blocks in a single response when the commands are independent (e.g., searching multiple files, reading different parts of the codebase).
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

Example of a CORRECT response:
<example_response>
I need to understand the Builder-related code. Let me find relevant files and check the project structure.

```mswea_bash_command
ls -la
```

```mswea_bash_command
find src -name '*.java' | grep -i builder
```

```mswea_bash_command
cat README.md | head -50
```
</example_response>

## Environment Details

- You have a full Linux shell environment
- Always use non-interactive flags (-y, -f) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- You can use bash commands or invoke any tool that is available in the environment
- You can also create new tools or scripts to help you with the task
- If a tool isn't available, you can also install it

## Submission

When you've completed your work, you MUST submit your changes as a git patch.
Follow these steps IN ORDER, with SEPARATE commands:

Step 1: Create the patch file
Run `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only the source files you modified.
Do NOT commit your changes.

<IMPORTANT>
The patch must only contain changes to the specific source files you modified to fix the issue.
Do not submit file creations or changes to any of the following files:

- test and reproduction files
- helper scripts, tests, or tools that you created
- installation, build, packaging, configuration, or setup scripts unless they are directly part of the issue you were fixing (you can assume that the environment is already set up for your client)
- binary or compiled files
</IMPORTANT>

Step 2: Verify your patch
Inspect patch.txt to confirm it only contains your intended changes and headers show `--- a/` and `+++ b/` paths.

Step 3: Submit (EXACT command required)
You MUST use this EXACT command to submit:

```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

If the command fails (nonzero exit status), it will not submit.

<CRITICAL>
- Creating/viewing the patch and submitting it MUST be separate commands (not combined with &&).
- If you modify patch.txt after verifying, you SHOULD verify again before submitting.
- You CANNOT continue working (reading, editing, testing) in any way on this task after submitting.
</CRITICAL>
</instructions>
""".strip()  # noqa: E501


@dataclasses.dataclass(frozen=True)
class MiniSWEAgentV2Config:
    """Configuration for MiniSWECodeAgentV2.

    This dataclass holds all configurable parameters for the agent.
    Use the factory methods to create preset configurations.
    """

    # Limits and behavior
    step_limit: int = 0
    cost_limit: float = 3.0
    timeout_s: int | None = None
    temperature: float = 0.0

    # Templates
    system_template: str = _TEXTBASED_SYSTEM_TEMPLATE
    instance_template: str = _TEXTBASED_INSTANCE_TEMPLATE
    observation_template: str = _SHARED_OBSERVATION_TEMPLATE
    format_error_template: str = _TEXTBASED_FORMAT_ERROR_TEMPLATE

    # Environment and execution
    env_vars: dict[str, str] = dataclasses.field(default_factory=lambda: _DEFAULT_ENV_VARS.copy())
    working_directory: str | None = None

    # Action parsing and submission
    submission_sentinel: str = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    action_patterns: tuple[str, ...] = (
        r"```mswea_bash_command\s*\n(.*?)\n```",
        r"```bash\s*\n(.*?)\n```",
    )

    @classmethod
    def text_based(cls) -> "MiniSWEAgentV2Config":
        """Create config for generic text-based agent.

        This is the default configuration matching mini_textbased.yaml from
        the reference implementation. Suitable for general task-solving.
        """
        return cls()

    @classmethod
    def swe_bench_verified(cls) -> "MiniSWEAgentV2Config":
        """Create config for SWE-bench Verified benchmark.

        This configuration matches the swebench.yaml from the reference
        implementation with adaptations for SWE-bench Verified:
        - Step limit: 250 (hard limit)
        - Timeout: 60 seconds per command
        - Working directory: /testbed
        - PR-focused prompts with git patch submission
        """
        return cls(
            step_limit=250,
            timeout_s=60,
            system_template=_SWEBENCH_SYSTEM_TEMPLATE,
            instance_template=_SWEBENCH_INSTANCE_TEMPLATE,
            working_directory="/testbed",
        )
