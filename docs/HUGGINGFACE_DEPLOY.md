# Deploy MindScore AI on Hugging Face Spaces

Hugging Face **removed Streamlit from the “New Space” form**. Use the **Docker** SDK instead — this repo includes a `Dockerfile` that runs Streamlit on port **8501**.

## Prerequisites

- [Hugging Face account](https://huggingface.co/join)
- GitHub repo pushed: `https://github.com/adityatagra/Mindscore-AI`
- Trained models in `models/` (already in your repo)
- Files: `Dockerfile`, `README.md` (YAML: `sdk: docker`, `app_port: 8501`)

## Option A — Create Space from GitHub (recommended)

1. Open [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. **Space name:** e.g. `mindscore-ai`
3. **Space SDK:** **Docker** (not Gradio)
4. **Space hardware:** CPU basic (free)
5. **Visibility:** Public (or Private)
6. After creation: **Settings → Repository** → connect **`adityatagra/Mindscore-AI`**, branch `main`

   *Or* pick **Docker** + blank Space, then push this repo to the Space remote.

Hugging Face builds the `Dockerfile` automatically (first build may take 10–20 minutes).

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
