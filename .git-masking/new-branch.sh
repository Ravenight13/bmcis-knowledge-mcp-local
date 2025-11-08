#!/bin/bash

# new-branch.sh - Create a new masked branch with automatic session ID assignment
# Usage: ./new-branch.sh [--task-id TASK_ID] [--description DESCRIPTION]
#
# This script:
# 1. Reads the next available session ID from branch-mapping.json
# 2. Creates a new branch with format: work/session-NNN
# 3. Updates branch-mapping.json with the new mapping
# 4. Increments nextSessionId for future branches

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAPPING_FILE="${SCRIPT_DIR}/branch-mapping.json"
CURRENT_DATE=$(date +%Y-%m-%d)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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
    echo "Please initialize the branch masking system first."
    exit 1
fi

# Parse command-line arguments
TASK_ID=""
DESCRIPTION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --task-id)
            TASK_ID="$2"
            shift 2
            ;;
        --description)
            DESCRIPTION="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--task-id TASK_ID] [--description DESCRIPTION]"
            echo ""
            echo "Creates a new masked branch with automatic session ID assignment."
            echo ""
            echo "Options:"
            echo "  --task-id TASK_ID          Internal task identifier (e.g., 'task-5.1')"
            echo "  --description DESCRIPTION  Task description"
            echo "  --help, -h                Show this help message"
            echo ""
            echo "If options are not provided, you will be prompted interactively."
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Read next session ID
NEXT_SESSION_ID=$(jq -r '.nextSessionId' "$MAPPING_FILE")

if [ -z "$NEXT_SESSION_ID" ] || [ "$NEXT_SESSION_ID" = "null" ]; then
    echo -e "${RED}Error: Could not read nextSessionId from $MAPPING_FILE${NC}"
    exit 1
fi

# Format session ID with leading zeros (3 digits)
SESSION_ID=$(printf "%03d" "$NEXT_SESSION_ID")
BRANCH_NAME="work/session-${SESSION_ID}"

echo -e "${BLUE}=== Creating New Masked Branch ===${NC}"
echo -e "Session ID: ${GREEN}${SESSION_ID}${NC}"
echo -e "Branch Name: ${GREEN}${BRANCH_NAME}${NC}"
echo ""

# Prompt for task ID if not provided
if [ -z "$TASK_ID" ]; then
    echo -e "${YELLOW}Enter Task ID${NC} (e.g., 'task-5.1', 'feature-auth', 'bugfix-123'):"
    read -r TASK_ID

    if [ -z "$TASK_ID" ]; then
        echo -e "${RED}Error: Task ID cannot be empty${NC}"
        exit 1
    fi
fi

# Prompt for description if not provided
if [ -z "$DESCRIPTION" ]; then
    echo -e "${YELLOW}Enter Task Description:${NC}"
    read -r DESCRIPTION

    if [ -z "$DESCRIPTION" ]; then
        echo -e "${RED}Error: Description cannot be empty${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${BLUE}=== Branch Details ===${NC}"
echo -e "Session ID:   ${GREEN}${SESSION_ID}${NC}"
echo -e "Branch:       ${GREEN}${BRANCH_NAME}${NC}"
echo -e "Task ID:      ${TASK_ID}"
echo -e "Description:  ${DESCRIPTION}"
echo ""

# Confirm before proceeding
read -p "Create this branch? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Aborted.${NC}"
    exit 0
fi

# Create the git branch
echo -e "${BLUE}Creating branch...${NC}"
if ! git checkout -b "$BRANCH_NAME" 2>/dev/null; then
    echo -e "${RED}Error: Failed to create branch $BRANCH_NAME${NC}"
    echo "The branch may already exist or you may not be in a git repository."
    exit 1
fi

# Update branch-mapping.json
echo -e "${BLUE}Updating branch mapping...${NC}"

# Create new entry
NEW_ENTRY=$(jq -n \
    --arg session "$SESSION_ID" \
    --arg task "$TASK_ID" \
    --arg desc "$DESCRIPTION" \
    --arg date "$CURRENT_DATE" \
    '{
        sessionId: $session,
        taskId: $task,
        description: $desc,
        created: $date,
        status: "active"
    }')

# Update the mapping file
TEMP_FILE="${MAPPING_FILE}.tmp"
jq --arg branch "$BRANCH_NAME" \
   --argjson entry "$NEW_ENTRY" \
   --arg date "$CURRENT_DATE" \
   '.branches[$branch] = $entry |
    .nextSessionId = (.nextSessionId + 1) |
    .metadata.lastModified = $date' \
   "$MAPPING_FILE" > "$TEMP_FILE"

# Validate the updated JSON
if ! jq empty "$TEMP_FILE" 2>/dev/null; then
    echo -e "${RED}Error: Generated invalid JSON${NC}"
    rm -f "$TEMP_FILE"
    exit 1
fi

# Replace original file
mv "$TEMP_FILE" "$MAPPING_FILE"

echo -e "${GREEN}✓ Branch created successfully!${NC}"
echo ""
echo -e "${BLUE}=== Next Steps ===${NC}"
echo ""
echo "1. Start working on your task using ONLY the session ID in commits:"
echo -e "   ${GREEN}git commit -m 'feat: session-${SESSION_ID} - implement core feature'${NC}"
echo ""
echo "2. When creating a PR, use the masked title:"
echo -e "   ${GREEN}gh pr create --title 'feat: session-${SESSION_ID} - core feature implementation'${NC}"
echo ""
echo "3. Check your current task info anytime:"
echo -e "   ${GREEN}./check-branch.sh${NC}"
echo ""
echo "4. To view the mapping (local only, never committed):"
echo -e "   ${GREEN}cat ${MAPPING_FILE}${NC}"
echo ""
echo -e "${YELLOW}IMPORTANT: Never reference '${TASK_ID}' in commit messages or PR titles!${NC}"
echo -e "${YELLOW}Always use 'session-${SESSION_ID}' for all public-facing git operations.${NC}"
