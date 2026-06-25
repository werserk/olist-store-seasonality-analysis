# Olist seasonality Slidev presentation

Source of truth: `slides.md`.

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

- `assets/pptx/` — images extracted from the original PPTX;
- `assets/figures/` — cleaned project figures copied from `results/final_figure_set/`.

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
