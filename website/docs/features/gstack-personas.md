# gstack Personas: Specialized Review Agents

Hermes Agent includes **7 specialized personas** inspired by [Garry Tan's gstack](https://github.com/garrytan/gstack) framework. Each persona is a subagent with a curated system prompt, restricted toolset, and specific focus. Together, they enable comprehensive reviews across product strategy, architecture, security, quality, design, and deployment.

## What Are Personas?

Personas are specialized AI agents that adopt specific roles within your development workflow. Each persona:

- **Adopts a specific mindset** (e.g., CEO, Architect, Security Officer)
- **Focuses on key concerns** (e.g., strategic fit, tech debt, OWASP vulnerabilities)
- **Has access to relevant tools** (e.g., Designer has browser for visual inspection)
- **Follows structured output format** (e.g., markdown sections with decision criteria)
- **Spawns as a subagent** (isolated context, no access to parent history or dangerous tools)

## The 7 Personas

### 👔 CEO
**Chief Product Officer** — Strategic fit, user value, market timing

**Focus:**
- Does this align with the product vision?
- Will users actually want this?
- Can we build and maintain it with current team?
- What risks could derail this?

**Tools:** Terminal, File, Web

**Output format:**
- Strategic Assessment (vision fit, competitive advantage)
- User Value (who benefits, what pain solved)
- Resource Reality (feasibility, constraints)
- Key Risks (failure modes, dependencies)
- Recommendation (proceed / refine scope / reconsider)
- Top Questions (3-5 critical questions to answer first)

**Usage:** `/ceo-review <target>`

---

### 🏗️ Eng Manager
**Engineering Architecture Lead** — Architecture, tech debt, maintainability

**Focus:**
- Is the design scalable and following our patterns?
- Is this introducing tech debt or paying it down?
- What's the coupling? Is it testable?
- Will the next person understand this?

**Tools:** Terminal, File

**Output format:**
- Architecture Review (patterns, scalability, coupling)
- Tech Debt (new debt, paydown, tradeoffs)
- Maintainability (clarity, test coverage, docs)
- Performance (efficiency, bottlenecks)
- Standards Alignment (convention mismatches)
- Blocking Issues (must fix before merge)
- Nice-to-Have (good-to-have improvements)

**Usage:** `/eng-review <target>`

---

### 🎨 Designer
**Head of Design & UX** — User flow, visual design, accessibility

**Focus:**
- Is the flow intuitive? Can users figure it out?
- Does it look professional and on-brand?
- Is it accessible (WCAG, keyboard, screen reader)?
- How does it handle errors and edge cases?

**Tools:** Browser, Web

**Output format:**
- Flow & Intuition (UX clarity)
- Visual Design (brand alignment, consistency)
- Accessibility (WCAG, keyboard nav, screen readers)
- Responsive Design (mobile, tablet, desktop)
- Micro-interactions (feedback, animations)
- Edge Cases (errors, loading, empty states)
- Specific Suggestions (3-5 concrete improvements)
- Assessment (shipping-ready / needs fixes / needs major work)

**Usage:** `/design-review <target>`

---

### 🔍 Reviewer
**Code Quality & Production Safety Lead** — Code quality, testing, production safety

**Focus:**
- Is the code readable and following best practices?
- Are there tests? Do they cover edge cases?
- What happens when things go wrong?
- Can we debug this in production?
- Will this break existing functionality?

**Tools:** Terminal, File

**Output format:**
- Code Quality (style, readability, best practices)
- Test Coverage (happy path, edge cases)
- Error Handling (failure modes, user-friendliness)
- Production Safety (regression risks, rollback)
- Logging & Observability (debuggability)
- Security Check (injection, auth, validation)
- Performance (N+1s, memory, blocking ops)
- Must-Fix Issues (critical blockers)
- Nice-to-Have (improvements)
- Approval (approved / request changes / hold)

**Usage:** `/reviewer <target>`

---

### 🧪 QA Lead
**Quality Assurance & Testing Lead** — Functional testing, edge cases, user flows

**Focus:**
- Does the core feature work as described?
- What happens at boundaries and error conditions?
- Can real users complete their task?
- Do existing features still work?
- Does it work on different browsers/devices?

**Tools:** Browser

**Output format:**
- Happy Path Testing (core features working)
- Edge Cases Tested (boundaries, errors, empty states)
- User Flow Walk-through (real user task completion)
- Regression Testing (existing features still work)
- Cross-Browser/Device (desktop, mobile, different browsers)
- Bugs Found (critical / major / minor with steps to reproduce)
- Accessibility (keyboard, screen reader, WCAG)
- Performance (load times, responsiveness, hangs)
- Test Cases to Add (what should be automated)
- Ready to Ship? (yes/no with confidence)

**Usage:** `/qa-audit <target>`

---

### 🔐 CSO
**Chief Security Officer** — Security, compliance, data protection

**Focus:**
- OWASP Top 10 vulnerabilities (injection, XSS, CSRF, auth)?
- How is PII handled? Is it encrypted?
- Are secrets hardcoded or exposed in configs?
- Are there known vulns in dependencies?
- Does it meet compliance requirements (GDPR, SOC 2)?

**Tools:** Terminal, File, Web

**Output format:**
- OWASP Assessment (injection, XSS, CSRF, auth, access control)
- Data Protection (PII handling, encryption, retention)
- Secrets & Credentials (hardcoded keys, exposed configs)
- Dependency Audit (known vulnerabilities)
- API Security (rate limiting, input validation)
- Authentication & Authorization (tokens, sessions, RBAC)
- Compliance (GDPR, SOC 2, industry requirements)
- High-Risk Issues (must fix before production)
- Medium-Risk Issues (should fix before GA)
- Low-Risk Issues (nice to fix, document as known)
- Security Sign-off (approved / with exceptions / blocked)

**Usage:** `/cso <target>`

---

### 🚀 Release Engineer
**Deployment & Release Manager** — Deployment strategy, rollback, monitoring

**Focus:**
- How do we ship this safely (canary, blue-green)?
- If it breaks, can we revert quickly?
- What metrics matter? Are we watching them?
- Are database migrations backward-compatible?
- Do we need feature flags?

**Tools:** Terminal

**Output format:**
- Deployment Strategy (canary / blue-green / rolling / direct)
- Rollback Plan (revert procedure, estimated time)
- Database Migrations (backward compat, testing, rollback)
- Feature Flags (toggles needed, config changes)
- Monitoring & Alerts (key metrics, alert thresholds)
- Communication Plan (notifications, runbooks)
- Pre-deployment Checklist (tests, security, docs)
- Go/No-Go Criteria (deployment readiness)
- Release Steps (detailed step-by-step)
- Post-deployment Verification (success validation)
- Ready to Release? (go / hold with reasons)

**Usage:** `/release-check <target>`

---

## How to Use: Example Workflows

### Single Persona Review

Review a Python file for code quality:
```
/reviewer ~/myproject/src/auth.py
```

Output: Detailed code review report saved to `~/.hermes/reviews/reviewer_YYYYMMDD_HHMMSS.md`

### Add Context

Provide background information:
```
/reviewer ~/myproject/src/auth.py
This is a critical authentication module that handles token verification for our API.
```

### Chained Reviews (Workflow)

Review a feature end-to-end, starting with design and ending with deployment sign-off:

**1. Design Review**
```
/design-review ~/myproject/docs/feature_mockups.md
```
Review UX, visual consistency, accessibility.

**2. Code Review**
```
/reviewer ~/myproject/src/feature.py
```
Review code quality, testing, production safety.

**3. Architecture Review**
```
/eng-review ~/myproject/src/feature.py
```
Review design patterns, tech debt, maintainability.

**4. Security Audit**
```
/cso ~/myproject/src/feature.py
```
Audit for OWASP vulnerabilities, compliance.

**5. QA Testing**
```
/qa-audit ~/myproject/tests/feature_test.py
```
Verify testing coverage and edge cases.

**6. Release Planning**
```
/release-check ~/myproject/RELEASE_PLAN.md
```
Plan deployment strategy, rollback, monitoring.

**7. CEO Sign-Off**
```
/ceo-review ~/myproject/FEATURE.md
```
Strategic assessment and final approval.

## Command Reference

| Command | Alias | Description |
|---------|-------|-------------|
| `/reviewer <target>` | `/review` | Code quality, production safety |
| `/ceo-review <target>` | `/ceo` | Strategic fit, user value |
| `/design-review <target>` | `/design` | UX, visual, accessibility |
| `/eng-review <target>` | `/eng` | Architecture, tech debt |
| `/qa-audit <target>` | `/qa` | Testing, edge cases |
| `/cso <target>` | `/security` | Security, OWASP, compliance |
| `/release-check <target>` | `/release` | Deployment, rollback, monitoring |
| `/gstack` | — | List all personas and usage |

## Output Format

All persona reviews follow consistent markdown structure:

```markdown
# [Persona Name] Review
**Target:** path/to/file
**Date:** 2026-03-31 10:30 AM

## Section 1: Focus Area
- Bullet points with findings
- Clear assessment of status

## Section 2: Another Focus Area
- Additional findings
- Recommendations

## Decision / Next Steps
- Clear recommendation
- Actions or follow-up items
```

Reports are automatically saved to: `~/.hermes/reviews/<persona>_YYYYMMDD_HHMMSS.md`

## Troubleshooting

### "No active agent context" Error
This means you need to start a Hermes session first:
```
/new
/reviewer ~/myproject/main.py
```

### Empty or Unhelpful Review
- **Problem:** The review is too generic
- **Solution:** Provide context for the reviewer:
  ```
  /reviewer ~/myproject/src/critical_path.py
  This is the core payment processing module. Focus on error handling and rollback safety.
  ```

### Review Takes Too Long
Persona subagents have iteration limits (15-30 per persona):
- **CEO:** 20 max iterations (strategic thinking is complex)
- **Eng Manager:** 25 max iterations (architecture requires depth)
- **Reviewer:** 30 max iterations (code review is thorough)
- **QA Lead:** 20 max iterations (testing is iterative)
- **CSO:** 25 max iterations (security is comprehensive)
- **Designer:** 15 max iterations (UX clarity is focused)
- **Release Engineer:** 20 max iterations (deployment planning)

If a review is cut off, you can:
1. **Retry with narrower scope:** `/reviewer src/auth.py` (specific file instead of whole module)
2. **Add explicit boundaries:** "Focus on security concerns only"

### How to Interpret Results

**Must-Fix vs. Nice-to-Have:**
- **Must-Fix/Blocking:** Fix before merge or deployment
- **Nice-to-Have:** Good improvements but not critical

**Confidence Levels:**
- Designer: "Shipping-ready" = high confidence
- QA: "Yes" (with X% confidence) = testing thoroughness
- CEO: "Proceed as-is" = strategic alignment

**Risk Levels (Security):**
- **High-Risk:** Blocks production deployment
- **Medium-Risk:** Must fix before general availability
- **Low-Risk:** Document as known, fix when possible

## Advanced: Custom Context

All personas accept optional context:

```
/reviewer ~/src/payment.py
This was flagged by the infosec team last week. Focus on input validation and transaction logging.
```

The context is injected into the persona's task description, helping them focus on key concerns.

## Performance Notes

- **Subagent Spawning:** Each persona spawns as an isolated subprocess (safe, no cross-contamination)
- **Toolset Isolation:** Dangerous tools (delegate_task, send_message, execute_code) are always blocked
- **Context Limit:** Child agents see only their task + context, not parent history
- **Iteration Budget:** Each persona has a max iteration count (see above)

## When to Use Each Persona

| Persona | Use When | Don't Use For |
|---------|----------|---------------|
| **CEO** | Evaluating new features, big architectural changes, major scope decisions | Code-level reviews, detailed testing |
| **Eng Manager** | Reviewing module structure, API design, tech debt assessment | Visual design, security audits |
| **Designer** | New UI components, UX flows, visual polish, accessibility | Backend code, deployment |
| **Reviewer** | Pull requests, code changes, test coverage, error handling | Architecture decisions, UI design |
| **QA Lead** | Feature branches, end-to-end flows, regression testing | Code review, architecture |
| **CSO** | Authentication, encryption, sensitive data handling, compliance | UI design, performance |
| **Release Engineer** | Shipping to production, monitoring setup, rollback planning | Code review, UX |

## See Also

- [gstack by Garry Tan](https://github.com/garrytan/gstack) — The inspiration behind this feature
- [Delegation Architecture](../integration/delegation.md) — How subagents work
- [Hermes CLI Reference](../cli-reference.md) — All available commands
