# Microimputation dashboard

This [Next.js](https://nextjs.org) app charts microimpute's imputation and benchmarking results from a CSV file.

## Features

- **Sample data**: imputation results that ship with the dashboard, so you can try it without a file of your own
- **File upload**: Drag & drop CSV file support with comprehensive validation
- **URL loading**: Load CSV files from remote URLs with error handling
- **No GitHub loading**: PolicyEngine archived microimpute, so the dashboard no longer loads GitHub Actions artifacts or reads a GitHub token. Anyone who opens a link from the old loader sees a message asking for a CSV instead.
- **Interactive visualizations**: benchmark loss charts, per-variable charts, distribution overlays, Wasserstein-distance and KL-divergence bar charts, a predictor correlation matrix, a matrix of the mutual information between each predictor and each imputed variable, and predictor-ordering robustness charts

## Getting started

Install Node.js, then run the development server from the root of the `microimpute` repository:

```bash
make dashboard
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

The dashboard loads the Inter and JetBrains Mono fonts through [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts).
