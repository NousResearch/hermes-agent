# Skill Candidate Card

Use this when a tenant wants to nominate a private skill or workflow for shared review. This card is a summary, not the raw private skill.

## Candidate

- Name: `<skill-or-workflow-name>`
- Type: `skill | workflow | cron-template | tool-pattern`
- Source tenant: `<tenant-name>`
- Proposed shared name: `<optional>`
- Status: `draft | submitted | reviewed | promoted | rejected`

## Purpose

One sentence describing what the candidate does.

## Intended users

Who could reuse this if it became shared?

## What should be shared

Describe the reusable method, template, checklist, or code pattern.

## What must remain private

List anything removed or excluded from the candidate bundle.

- Personal preferences
- Customer data
- Internal project names
- Raw conversation excerpts
- Memory/session content
- Credentials, tokens, Sheet IDs, chat IDs, or private paths

## Dependencies

List tools, APIs, skills, or environment assumptions.

## Risk level

`low | medium | high`

Explain why.

## Sanitization checklist

- [ ] No secrets or credentials
- [ ] No raw memory or session transcript
- [ ] No customer/private data
- [ ] No private file paths or chat IDs
- [ ] Examples are generic or synthetic
- [ ] External side effects are documented

## Evidence of usefulness

Summarize usage count, success examples, or why this pattern is broadly reusable. Do not include private source data.

## Review request

What decision is requested from the operator/admin?

- [ ] Review only
- [ ] Convert to shared skill
- [ ] Convert to shared template
- [ ] Reject / keep private
