#!/bin/bash

# commands.sh
# This script helps manage common development processes for the project.

# Usage:
# commands.sh [action]

# [action]:
# - format: Format all Python files using yapf
# - docs_serve: Start MkDocs server and Tailwind watcher
# - docs_build: Build the static site to the dist/ directory

set -e # Exit immediately if a command exits with a non-zero status.

DRY_RUN=false

# Parse options
while getopts "n" opt; do
  case $opt in
    n)
      DRY_RUN=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

# Define a function to execute commands or print them in dry-run mode
function execute_cmd() {
    if "$DRY_RUN"; then
        echo "DRY RUN: $@"
    else
        echo "> $@"
        "$@"
    fi
}

# Execute actions based on arguments
while (( "$#" )); do
    ACTION="$1"
    shift # Consume the action

    # Validate action
    case "$ACTION" in
        format)
            execute_cmd yapf --style "{based_on_style: google, column_limit: 80, indent_width: 2}" --in-place *.py
            ;;
        docs_serve)
            # Run tailwind and mkdocs in parallel.
            # Kill existing tailwind process if running to avoid duplicates.
            pkill -f "@tailwindcss/cli" || true
            pnpm dlx @tailwindcss/cli -i ./src/input.css -o ./docs/stylesheets/tailwind.css --watch &
            # Use polling and explicit livereload flag to work around click/mac issues.
            export WATCHDOG_USE_POLLING=1
            execute_cmd .venv/bin/mkdocs serve --livereload
            ;;
        docs_build)
            pnpm dlx @tailwindcss/cli -i ./src/input.css -o ./docs/stylesheets/tailwind.css --minify
            execute_cmd .venv/bin/mkdocs build
            ;;
        *)
            echo "Error: Invalid action '$ACTION'." >&2
            echo "Usage: commands.sh [action] ([action2]...)" >&2
            exit 1
            ;;
    esac
done
