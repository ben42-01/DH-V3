# GitHub Pages Deployment Guide

## Prerequisites

- GitHub CLI (`gh`) installed and authenticated
- Docusaurus built and ready to deploy
- Repository created on GitHub

## Setup (One-time)

### 1. Create Repository (if not already created)

Using GitHub CLI:
```bash
gh repo create qr-x-wiki --public --source=. --remote=origin --push
```

Or manually:
- Go to [github.com/new](https://github.com/new)
- Create `qr-x-wiki` as public repo
- Clone locally and push this code

### 2. Configure Repository Settings (via UI or CLI)

**Via UI (easier for first time):**
1. Go to repo → Settings → Pages
2. Under "Source", select `Deploy from a branch`
3. Branch: `gh-pages`, Folder: `/ (root)`
4. Click Save

**Via GitHub CLI:**
```bash
gh repo edit --enable-pages
```

### 3. Update docusaurus.config.js

Before deploying, ensure these values match your GitHub setup:

```javascript
// In docusaurus.config.js
const config = {
  url: 'https://YOUR_USERNAME.github.io',
  baseUrl: '/qr-x-wiki/',
  organizationName: 'YOUR_USERNAME',  // or org name
  projectName: 'qr-x-wiki',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,
  // ... rest of config
};
```

## Deploy Workflow

### Option A: Using GitHub CLI (Recommended)

```bash
# 1. Build the site
npm run build

# 2. Deploy to gh-pages branch
npm run deploy

# 3. Verify deployment
gh repo view --web
```

### Option B: Manual Deployment

```bash
# 1. Install gh-pages package (optional, for easier deployments)
npm install --save-dev gh-pages

# 2. Build
npm run build

# 3. Deploy to gh-pages branch manually
git branch -D gh-pages 2>/dev/null || true
git subtree push --prefix build origin gh-pages
```

## Verify Deployment

1. **Check Actions (CI/CD):**
   ```bash
   gh run list
   ```

2. **View deployed site:**
   ```bash
   gh repo view --web
   ```

3. **Manual check:**
   - Go to `https://YOUR_USERNAME.github.io/qr-x-wiki/`
   - Should show your Docusaurus wiki

## Troubleshooting

### Site not showing
- Wait 2-3 minutes for GitHub Pages to rebuild
- Check Actions tab for deployment errors
- Verify `gh-pages` branch exists: `git branch -a`

### Wrong URL/baseUrl
- Update `docusaurus.config.js` and rebuild
- Re-run `npm run deploy`

### Permission denied
- Ensure GitHub CLI is authenticated: `gh auth login`
- Verify you have push access to the repo

## Continuous Deployment (Optional)

To auto-deploy on push, create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm install
      - run: npm run build
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./build
```

Then just push to main and it auto-deploys!
