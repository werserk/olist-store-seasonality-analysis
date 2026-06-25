# Olist seasonality Slidev presentation

Source of truth: `slides.md`.

Planning helpers for restructuring:

- `assignment-questions.md` — exact main and additional questions from the assignment;
- `current-structure.md` — current slide-by-slide structure before the planned rewrite;
- `proposed-question-driven-structure.md` — draft of the new question-driven presentation flow.

## Commands

```bash
npm install
npm run dev
npm run build
npm run export:pdf
```

Speaker text is duplicated in two places:

- inline Slidev presenter notes inside `slides.md` as HTML comments;
- standalone reading copy in `speaker-notes.md`.

Assets:

- `assets/figures/` — project figures (`results/final_figure_set/` + pres exports);
- `assets/pptx/` — legacy PPTX extracts (not used in `slides.md`).

Refresh figures before deploy:

```bash
python scripts/sync_slidev_figures.py
```

## Deployment

Production URL: `https://olist-store.edu.werserk.com/`.

Deployment target:

```text
GitHub Actions on main
  -> Slidev build from presentation/slides.md
  -> Yandex Object Storage bucket olist-store.edu.werserk.com
  -> Certificate Manager + bucket HTTPS
  -> Yandex Cloud DNS A olist-store.edu.werserk.com -> 213.180.193.247
```

The workflow is `.github/workflows/deploy-olist-store-presentation.yml`.
