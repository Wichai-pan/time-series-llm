# Slides Status

最后更新：2026-06-08

## Active Deck

- Deck id：`2026-06-08-advisor-postmeeting-update`
- Source：`slides/decks/2026-06-08-advisor-postmeeting-update.md`
- PPTX：`slides/decks/2026-06-08-advisor-postmeeting-update.pptx`
- Preview：`http://localhost:3030/` when `npm run dev:postmeeting -- --port 3030` is running
- Audience：advisor progress update
- Purpose：两页说明会后新增 verification experiments：normalization、multi-subject、unseen subject、latent validity / clip。

## Build / Template Status

- `slides/package.json` now configures a minimal Slidev environment.
- Dependencies installed with `npm install`.
- Current active deck builds with `npm run build:postmeeting`.
- Editable PPTX generated with `npm run build:pptx:postmeeting`.
- Local preview verified with `curl -I http://localhost:3030/` returning HTTP 200.
- PDF exported to `slides/decks/2026-06-08-advisor-postmeeting-update.pdf`.
- Static build output currently appears under `slides/decks/dist/2026-06-08-advisor-postmeeting-update/` due to Slidev entry-relative output behavior.

## Visual Validation

- Current active deck is text/table only; no external figure assets required.
- Build passed. Browser visual inspection still recommended before final presentation.
- PPTX package inspected with `unzip -l`; it contains 2 slides and speaker notes.

## Known Risks

- Full slide-by-slide visual inspection has not been performed inside the browser beyond successful HTTP preview and PDF export.
- PPTX has not been manually opened in PowerPoint/Keynote in this session.
- Shell startup prints unrelated sandbox permission warnings from local zsh plugins; Slidev build itself succeeds.
- Metric terminology must be explained orally as diagnostic TSGBench-style metrics, not final benchmark or HAR utility claims.
