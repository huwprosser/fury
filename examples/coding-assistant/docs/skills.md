# How to Create a Skill for Pi

Skills let you extend Pi with new capabilities, custom workflows, and documentation. You can design a skill for anything Pi can help automate—such as document processing, implementing coding procedures, or interacting with external APIs.

This guide shows you how to make your own skill.

---

## 1. Create the Skill Directory

Every skill lives in its own directory inside the skills directory. Name your directory using lowercase letters, numbers, and hyphens only. For example:

```
skills/my-skill/
```

## 2. Add a `SKILL.md` File

Inside your skill directory, create a file named `SKILL.md`. This markdown file holds the configuration and documentation for your skill.

At a minimum, `SKILL.md` needs:

- **Frontmatter**: YAML block at the top describing the skill.
- **Instructions**: Explanation of what the skill does, how to set it up, and how to use it.

### Example structure

```
my-skill/
├── SKILL.md
├── scripts/
│   └── my-helper.sh
├── references/
│   └── guide.md
└── assets/
    └── data.json
```

## 3. Write the Frontmatter

Frontmatter is a YAML block at the very top of your `SKILL.md` file, delimited by `---`. The fields below are required or recommended:

```markdown
---
name: my-skill
description: A short summary of what the skill does and when to use it.
license: MIT                      # Optional
compatibility: Linux, Mac, NodeJS # Optional
metadata:                         # Optional, for custom fields
  author: you@example.com
---
```

**Rules for `name`:**
- 1-64 characters, lowercase letters/numbers/hyphens
- No leading/trailing or double hyphens
- Must match the directory name

**Rules for `description`:**
- Max 1024 characters
- Be specific about what the skill does and when to use it

**Optional fields:**
- `license`: Name or file reference
- `compatibility`: Specify platforms, dependencies, etc.
- `metadata`: Any extra info (key-value pairs)
- `allowed-tools`: Space-separated list of CLI tools the skill can use (advanced)
- `disable-model-invocation`: Set to `true` to disable auto-loading

## 4. Add Instructions, Setup, and Usage

Below the frontmatter, use Markdown to document your skill. At a minimum, include:

- What your skill does
- Setup steps (if any)
- How to use it (examples)
- Any helper scripts or references

### Example SKILL.md

```markdown
---
name: pdf-tools
description: Extracts text/tables from PDFs, fills forms, merges files. Use for PDF workflows.
---

# PDF Tools

## Setup

Install dependencies before first use:

```bash
cd /path/to/pdf-tools && npm install
```

## Usage

Extract text from a PDF:

```bash
./scripts/extract.sh input.pdf
```

Combine multiple PDFs:

```bash
./scripts/merge.sh file1.pdf file2.pdf -o combined.pdf
```

See reference docs:

[Read the guide](references/guide.md)
```

Use relative paths for all files referenced in instructions.

## 5. Add Any Supporting Files

You can include scripts, data files, documentation, or other assets as needed. Use sensible organization such as:

- `scripts/`: Shell/Node/Python scripts used by the skill
- `references/`: Extended docs or API guides
- `assets/`: Data, templates, etc.

## 6. Validate Your Skill

Follow these guidelines so your skill loads correctly:

- The directory name and `name` frontmatter must match
- No uppercase letters or spaces in the skill name
- Include a clear `description`
- Check that instructions are clear and reference only included files

Common mistakes which usually just issue warnings:
- Name too long or contains invalid characters
- Missing or vague description
- Unknown frontmatter fields

**Note:** Skills **without a `description`** will not be loaded.

## Example Directory

```
example-skill/
├── SKILL.md
├── scripts/
│   └── run.sh
└── references/
    └── README.md
```

## Example SKILL.md

```markdown
---
name: example-skill
description: Shows how to build and structure a basic Pi skill.
---

# Example Skill

## Setup

No setup required.

## Usage

Run the helper script:

```bash
./scripts/run.sh
```
```

---

For more information, see the [Agent Skills specification](https://agentskills.io/specification).
