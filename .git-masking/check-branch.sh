#!/bin/bash

# check-branch.sh - Look up the current branch in the mapping and display task info
# Usage: ./check-branch.sh [BRANCH_NAME]
#
# This script displays the real task information for a masked branch.
# If no branch name is provided, uses the current git branch.

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAPPING_FILE="${SCRIPT_DIR}/branch-mapping.json"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is required but not installed.${NC}"
    echo "Install with: brew install jq (macOS) or apt-get install jq (Linux)"
    exit 1
fi

# Check if mapping file exists
if [ ! -f "$MAPPING_FILE" ]; then
    echo -e "${RED}Error: branch-mapping.json not found at $MAPPING_FILE${NC}"
    echo "The branch masking system may not be initialized."
    exit 1
fi

# Get branch name from argument or current git branch
if [ -n "$1" ]; then
    BRANCH_NAME="$1"
else
    # Get current branch name
    if ! BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD 2>/dev/null); then
        echo -e "${RED}Error: Not in a git repository${NC}"
        exit 1
    fi
fi

echo -e "${BLUE}=== Branch Information ===${NC}"
echo -e "Branch: ${GREEN}${BRANCH_NAME}${NC}"
echo ""

# Look up branch in mapping
BRANCH_INFO=$(jq -r --arg branch "$BRANCH_NAME" '.branches[$branch]' "$MAPPING_FILE")

if [ "$BRANCH_INFO" = "null" ] || [ -z "$BRANCH_INFO" ]; then
    echo -e "${YELLOW}⚠ This branch is not in the masked branch mapping.${NC}"
    echo ""
    echo "This could mean:"
    echo "  • It's an unmasked branch (e.g., 'main', 'develop')"
    echo "  • It was created before the masking system was implemented"
    echo "  • It's a legacy branch"
    echo ""
    echo "To create a new masked branch, run:"
    echo -e "  ${GREEN}./new-branch.sh${NC}"
    exit 0
fi

# Extract and display information
SESSION_ID=$(echo "$BRANCH_INFO" | jq -r '.sessionId')
TASK_ID=$(echo "$BRANCH_INFO" | jq -r '.taskId')
DESCRIPTION=$(echo "$BRANCH_INFO" | jq -r '.description')
CREATED=$(echo "$BRANCH_INFO" | jq -r '.created')
STATUS=$(echo "$BRANCH_INFO" | jq -r '.status')

echo -e "${CYAN}Session ID:${NC}    ${GREEN}${SESSION_ID}${NC}"
echo -e "${CYAN}Task ID:${NC}       ${TASK_ID}"
echo -e "${CYAN}Description:${NC}   ${DESCRIPTION}"
echo -e "${CYAN}Created:${NC}       ${CREATED}"
echo -e "${CYAN}Status:${NC}        ${STATUS}"
echo ""

# Show commit message examples
echo -e "${BLUE}=== Commit Message Examples ===${NC}"
echo -e "${GREEN}git commit -m 'feat: session-${SESSION_ID} - add new feature'${NC}"
echo -e "${GREEN}git commit -m 'fix: session-${SESSION_ID} - resolve bug'${NC}"
echo -e "${GREEN}git commit -m 'docs: session-${SESSION_ID} - update documentation'${NC}"
echo ""

# Show PR title example
echo -e "${BLUE}=== PR Title Example ===${NC}"
echo -e "${GREEN}gh pr create --title 'feat: session-${SESSION_ID} - feature implementation'${NC}"
echo ""

# Warning about security
echo -e "${YELLOW}⚠ SECURITY REMINDER:${NC}"
echo -e "Never reference '${RED}${TASK_ID}${NC}' in commit messages or PR titles!"
echo -e "Always use '${GREEN}session-${SESSION_ID}${NC}' for all public git operations."
