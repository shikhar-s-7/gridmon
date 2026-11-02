# Publishing the website (GitHub Pages)

The site is static and already built: `web/dist/` is committed (built by `python scripts/build_site.py`). Publishing
means telling GitHub Pages to serve that folder. Expected address: **https://shikhar-s-7.github.io/gridmon/**

## Preview locally first
```
python -m http.server -d web/dist 8000
```
then open http://localhost:8000 (opening the HTML files directly from disk does not work: the browser blocks the
data and WebAssembly downloads for `file://` pages).

## One-time setup on GitHub (about 3 minutes)
1. The workflow file `.github/workflows/pages.yml` must be on the `main` branch. If it is already there (it was pushed
   with the M8 commits), skip to step 3.
2. If it is missing (a push of workflow files can be refused when the access token lacks the "workflow" permission):
   on github.com open the repository, click **Add file -> Create new file**, type the name
   `.github/workflows/pages.yml`, paste the text below, and click **Commit changes** (directly to `main`).
3. Open **Settings -> Pages**. Under **Build and deployment -> Source**, choose **GitHub Actions**.
4. Open the **Actions** tab, select **Deploy website to GitHub Pages**, click **Run workflow** (branch `main`).
   After about a minute the run shows the site address; later pushes to `main` that change `web/dist/` redeploy
   automatically.

## The workflow text
```yaml
# Deploy the committed static site (web/dist) to GitHub Pages. Nothing is built here: scripts/build_site.py
# produces web/dist locally (it is committed), so this workflow needs no toolchain.
name: Deploy website to GitHub Pages
on:
  push:
    branches: [main]
    paths: ["web/dist/**", ".github/workflows/pages.yml"]
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: false
jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: web/dist
      - id: deployment
        uses: actions/deploy-pages@v4
```

## Updating the site
After changing reports, pages or the C core: `python scripts/build_site.py --check` (rebuilds data, the WebAssembly
module and the pages, and re-checks WebAssembly against the native C core), optionally
`node web/tools/check_site.mjs http://localhost:8000 reports/m8_site_check.json reports/figures` with the preview
server running, then commit `web/` and push to `main`.

## Notes
- Private repositories need a paid GitHub plan for Pages; public repositories do not.
- The site makes no requests to other servers and has no tracking; GitHub Pages itself logs visits as any web host does.
