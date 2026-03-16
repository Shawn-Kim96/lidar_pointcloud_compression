# LiDAR Compression Research Memory
@./.gemini/prompts/context_lidar_compression_research.md

## Collaboration Defaults
- Treat this repository as a research codebase, not a toy project.
- Prefer reproducible changes: explicit assumptions, fixed protocol, and clear artifacts.
- Separate claims by evaluation track:
  - Track-A: codec and ROI diagnostics.
  - Track-B: official KITTI detector endpoint.
- Never over-claim detector gains when original AP sanity is unresolved.

## Working Style
- Propose hypothesis -> test design -> success criteria -> failure diagnosis.
- When a result is surprising, run decomposition checks before architecture changes.
- Keep recommendations bounded by available evidence.

## Quick Role Commands
- `/role:senior_scientist`
- `/role:experiment_director`
- `/role:paper_reviewer_response`
- `/role:result_debugger`

