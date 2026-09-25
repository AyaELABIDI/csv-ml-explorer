# CSV ML Explorer

Upload a CSV file and instantly get an overview of your data, descriptive
statistics, column distributions, a correlation matrix, and a quick
machine-learning baseline (regression or classification) — all running
locally in your browser. No server, no upload, no data leaves your computer.

<img width="931" height="389" alt="image" src="https://github.com/user-attachments/assets/264c57d2-94b0-44ff-beda-31944c91e446" />
<img width="947" height="391" alt="image" src="https://github.com/user-attachments/assets/5dd38082-71d0-401f-a2e9-b3935c72de68" />

## Features

- **Drag-and-drop CSV upload**, with automatic delimiter detection
- **Overview**: row/column counts, missing values, duplicates, column health
- **Data table**: searchable, sortable, with a header-row toggle
- **Statistics**: count, missing, unique, mean, std, min/max, quartiles, most
  common value per column — downloadable as CSV
- **Charts**: histograms for numeric columns, bar charts for categorical
  ones, a correlation heatmap, and a scatter plot to compare any two columns
- **Model training**: pick a target column, auto-detect regression vs.
  classification, and train Linear Regression, K-Nearest Neighbors, or
  Nearest Centroid — with R² / RMSE / MAE or accuracy + confusion matrix
- **Light/dark mode**

## Tech stack

- React 18 + Vite
- PapaParse for CSV parsing
- Plain CSS (no framework)
- All machine learning code is custom-written in `src/ml.js` (no external
  ML library)

## Getting started

```bash
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`).

## Build

```bash
npm run build
npm run preview
```

## Deploy to GitHub Pages

```bash
npm run deploy
```

Then enable GitHub Pages in the repo settings: **Settings → Pages → Source:
Deploy from a branch → Branch: `gh-pages` → folder `/ (root)`**.
