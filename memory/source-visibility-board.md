# Source Visibility Board

最后更新：2026-05-18

## VIS-001：Root project memory and planning docs

- Surface：root control project
- Path：`.`
- Visibility tier：agent-private
- Audience：用户和本地 agents
- Sync target：none
- Allowed paths：
  - `memory/`
  - `docs/`
  - `reference/.agent/`
  - `reference/cards/`
  - `reference/project-use/`
- Forbidden paths：
  - credentials
  - private tokens
  - 长篇 copyrighted paper excerpts
  - 未脱敏 private collaborator feedback
- Cleanup gate：任何 public release 或 collaborator-visible push 之前
- Audit status：needs-verification

## VIS-002：未来 paper/report 组件

- Surface：paper
- Path：`paper/`
- Visibility tier：agent-private
- Audience：用户和本地 agents
- Sync target：none
- Allowed paths：[]
- Forbidden paths：
  - raw experiment outputs
  - private reviewer strategy
  - unverified result claims
- Cleanup gate：submission 或 sharing 前
- Audit status：not-started
