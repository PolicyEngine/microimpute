# Microimputation dashboard

This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Features

- **Sample data**: Pre-loaded imputation results for immediate testing
- **File upload**: Drag & drop CSV file support with comprehensive validation
- **URL loading**: Load CSV files from remote URLs with error handling
- **GitHub loading (retired)**: PolicyEngine archived microimpute, so the dashboard no longer loads GitHub Actions artifacts or reads a GitHub token. Anyone who opens a link from the old loader sees a message asking for a CSV instead
- **Interactive visualizations**: benchmark loss charts, per-variable charts, distribution overlays, Wasserstein-distance and KL-divergence bar charts, a predictor correlation matrix and predictor-ordering robustness charts

## Getting started

Install Node.js, then run the development server from the root of the `microimpute` repository:

```bash
make dashboard
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

The dashboard loads the Inter and JetBrains Mono fonts through [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts).

## Learn more

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!
