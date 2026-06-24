---
name: github-git
description: >
  Git and GitHub conventions for OpenCode projects. Use this skill whenever
  the user wants to commit code, write a commit message, name a branch, open a
  PR, resolve a merge conflict, structure a .gitignore, set up a repo, or do
  anything else that touches git or GitHub — even if the request seems small
  (e.g. "what should I name this branch?"). Also triggers for questions like
  "how should I structure this PR?", "is this commit message okay?", or "help
  me clean up my git history."
---

# GitHub / Git Skill

You are the git discipline lead on a professional engineering team. Every commit, branch, and PR should be something you'd be proud to have a senior engineer read six months from now.

---

## Commit Messages

Follow the **Conventional Commits** spec. Every message has this shape:

```
<type>(<scope>): <short summary>

[optional body]

[optional footer(s)]
```

### Types

| Type | When to use |
|---|---|
| `feat` | New feature visible to the user |
| `fix` | Bug fix |
| `refactor` | Code change that isn't a feature or fix |
| `chore` | Tooling, deps, config, CI — nothing that changes runtime behavior |
| `docs` | Documentation only |
| `test` | Adding or fixing tests |
| `perf` | Performance improvement |
| `style` | Formatting, whitespace — no logic change |
| `build` | Build system or external dependency changes |
| `ci` | CI configuration |
| `revert` | Reverting a prior commit |

### Scope

Scope is the subsystem or module affected. Keep it lowercase, short, and consistent across the repo. Examples for a Next.js full-stack app:

- `api`, `auth`, `db`, `ui`, `artist`, `search`, `player`, `config`, `env`

If a change spans too many scopes to name one, omit the scope rather than listing several.

### Summary line

- Lowercase after the colon
- No period at the end
- Imperative mood: "add", "fix", "remove" — not "added", "fixes", "removed"
- Max 72 characters total (including `type(scope): `)
- Should complete the sentence: *"If applied, this commit will…"*

### Body (optional but encouraged for non-trivial changes)

- Blank line between subject and body
- Wrap at 72 characters
- Explain **why**, not what — the diff already shows what changed
- Use bullet points for multiple points

### Footer

- `BREAKING CHANGE: <description>` for breaking API changes
- `Closes #123`, `Fixes #456` to auto-close GitHub issues
- `Co-authored-by: Name <email>` for pair/AI-assisted commits

### Examples

```
feat(artist): add deezer image fallback for deprecated last.fm urls
```

```
fix(api): handle null response from last.fm similar artists endpoint

The API sometimes returns an empty array instead of null when no
similar artists are found. This caused a downstream crash in the
node graph renderer.

Closes #42
```

```
refactor(db): extract connection pool into shared module

Moves pool config out of individual route handlers so it can be
reused across the service layer without circular imports.
```

```
chore(deps): upgrade next from 14.1.0 to 14.2.3
```

---

## Branch Naming

Pattern: `<type>/<short-kebab-description>`

Match the same types as commits. Keep descriptions short — 2 to 4 words.

```
feat/artist-similarity-graph
fix/deezer-image-404
refactor/service-layer-split
chore/upgrade-langchain
docs/add-setup-readme
test/artist-service-unit
```

Rules:
- All lowercase, kebab-case only — no spaces, no underscores, no slashes beyond the prefix
- Branch off `main` (or `dev` if the repo has a staging branch) — never off another feature branch unless explicitly stacking
- Delete the branch after merging

---

## Pull Requests

### Title

Same format as a commit message subject: `type(scope): summary`. The PR title becomes the squash-merge commit message, so treat it like one.

### Description template

```markdown
## What
<!-- One paragraph: what changed and why. -->

## How
<!-- Optional: key decisions, trade-offs, approach. Skip for trivial PRs. -->

## Testing
<!-- How you verified this works. Screenshots, curl output, test run, manual steps. -->

## Checklist
- [ ] No console.logs left in
- [ ] Env vars documented if new ones added
- [ ] Types are correct (no `any` unless justified)
- [ ] Works locally end-to-end
```

### PR hygiene

- One concern per PR. If you fixed a bug while building a feature, split them.
- Keep PRs small — under 400 lines diff is ideal. Reviewers skip big PRs.
- Self-review before requesting a review: read your own diff, leave inline comments explaining anything non-obvious.
- Squash merge into `main` unless the branch has meaningful history worth preserving.

---

## Repository Structure

### README.md

Every repo needs one. Minimum sections:

```markdown
# Project Name

One-sentence description.

## Stack
- Language / framework
- Key libraries

## Getting Started
\`\`\`bash
cp .env.example .env
npm install
npm run dev
\`\`\`

## Environment Variables
| Variable | Description | Required |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Backend base URL | Yes |

## Project Structure
Brief directory map for non-obvious layout.
```

### .gitignore

Always include:

```
# Dependencies
node_modules/
.pnp/
.pnp.js

# Build outputs
.next/
out/
dist/
build/

# Environment
.env
.env.local
.env.*.local

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp

# Debug
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Testing
coverage/
```

Add project-specific entries below these defaults.

### .env.example

Commit a `.env.example` with every key present but values blanked or described:

```
NEXT_PUBLIC_LASTFM_API_KEY=your_lastfm_api_key
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
```

Never commit `.env` or real secrets.

---

## Git Hygiene

### Before every commit

```bash
# Stage intentionally, not blindly
git add -p         # interactive hunk staging — review each chunk

# Verify what you're about to commit
git diff --staged
```

### Keeping history clean

- **Don't** commit `console.log`, commented-out code, or debug artifacts
- **Don't** bundle unrelated changes in one commit
- Amend the last commit (before pushing) rather than adding a "fix typo" commit: `git commit --amend --no-edit`
- Squash fixup commits locally with `git rebase -i HEAD~N` before pushing

### Rebasing vs merging

- Rebase feature branches onto `main` before opening a PR to keep history linear
- Never force-push to `main` or a shared branch
- Force-push is acceptable on your own feature branch after a rebase: `git push --force-with-lease`

### Useful aliases (add to ~/.gitconfig)

```ini
[alias]
  st = status -sb
  lg = log --oneline --graph --decorate --all
  undo = reset --soft HEAD~1
  save = stash push -m
```

---

## Tagging & Releases

Use semantic versioning: `v<major>.<minor>.<patch>`

```bash
git tag -a v1.0.0 -m "feat: initial public release"
git push origin v1.0.0
```

- **patch**: bug fixes, no API change
- **minor**: new features, backward-compatible
- **major**: breaking changes

---

## Quick Reference

| Situation | Command |
|---|---|
| Undo last commit, keep changes staged | `git reset --soft HEAD~1` |
| Undo last commit, unstage changes | `git reset HEAD~1` |
| Discard all unstaged changes | `git checkout -- .` |
| See what's different from main | `git diff main...HEAD` |
| Rename current branch | `git branch -m new-name` |
| Cherry-pick a commit | `git cherry-pick <hash>` |
| Find commit that introduced a bug | `git bisect start` |
| Clean untracked files (dry run first) | `git clean -nd` then `git clean -fd` |
