# Deploy MindScore AI on Hugging Face Spaces

Your repo is configured for a **Streamlit Space**. The app entry point is `app/streamlit_app.py` (set in `README.md` YAML frontmatter).

## Prerequisites

- [Hugging Face account](https://huggingface.co/join)
- GitHub repo pushed: `https://github.com/adityatagra/Mindscore-AI`
- Trained models in `models/` (already in your repo)

## Option A — Create Space from GitHub (recommended)

1. Open [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. **Space name:** e.g. `mindscore-ai`
3. **License:** MIT (or your choice)
4. **Space SDK:** Streamlit
5. **Space hardware:** CPU basic (free)
6. Under **Create from**, choose **GitHub** and connect your account
7. Select repository **`adityatagra/Mindscore-AI`**
8. Branch: `main` → **Create Space**

Hugging Face will clone the repo and build automatically (first build may take 5–15 minutes).

## Option B — Push with Git CLI

```bash
# One-time: install CLI and login
pip install huggingface_hub
huggingface-cli login

# Create empty Space on the website first (Streamlit SDK), then:
cd "/Users/adityatagra/Downloads/depression detection using text and speech"
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/mindscore-ai
git push hf main
```

Replace `YOUR_USERNAME` and `mindscore-ai` with your Space owner and name.

## After deploy

- Public URL: `https://huggingface.co/spaces/YOUR_USERNAME/mindscore-ai`
- **Building** → wait until status is **Running**
- If build fails, open **Logs** on the Space page

## Push updates

```bash
git add .
git commit -m "Update app"
git push origin main
```

If the Space is linked to GitHub, it rebuilds on each push to `main`.

## Troubleshooting

| Issue | Fix |
|--------|-----|
| Build fails on `librosa` | `packages.txt` installs `libsndfile1` and `ffmpeg` — commit and push |
| Model not found | Ensure `models/*.joblib` are committed (not in `.gitignore`) |
| Large file rejected | Enable [Git LFS](https://git-lfs.github.com/) for `models/*.joblib` or use HF Hub model upload |
| App blank / errors | Check **Logs** tab; verify `app/streamlit_app.py` path in README YAML |
| Slow cold start | Normal on free CPU; models (~72 MB) load on first request |

## Notes

- **SQLite history** (`outputs/user_scores.db`) resets when the Space restarts (ephemeral disk on free tier).
- This is a **demo / academic project**, not a clinical tool.
- For always-on hosting without your laptop, prefer Hugging Face over ngrok.
