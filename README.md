# Skin Lesion Classifier — Streamlit deployment

Best model: MobileNetV2 (DL)

## Deploy on Streamlit Community Cloud
1. Create a new GitHub repository and upload everything inside this folder (app.py, requirements.txt,
   model_info.json, best_model.keras).
   - GitHub blocks files > 100 MB. If the model is larger, use Git LFS or pick a smaller model.
2. Go to https://share.streamlit.io -> "New app" -> select the repo, branch and `app.py` as the main file.
3. In "Advanced settings" choose a Python version compatible with the pinned packages
   (this notebook ran on Python 3.13.15; 3.11 or 3.12 are safe choices).
4. Deploy.

Run locally:  pip install -r requirements.txt  &&  streamlit run app.py
