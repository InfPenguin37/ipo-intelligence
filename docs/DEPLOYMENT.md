# Free hosting with Streamlit Community Cloud

The app can run on Streamlit Community Cloud's free tier with the pretrained model bundled in the repository. No
local training database or model download is required for inference.

## Deploy

1. Push the repository to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) and choose **Create app**.
3. Select this repository, the `main` branch, and `src/ipo_intel/app.py` as the app file.
4. In **Advanced settings**, choose Python 3.12, the version used to create the bundled model.
5. In the **Secrets** field, optionally add the setting below to disable public live scraping:

   ```toml
   IPO_ENABLE_SCRAPERS = "0"
   ```

   With this setting, the app uses bundled seed data and user-entered overrides, and disables the live-fetch checkbox.
   Omit the setting or use `"1"` only if you want visitors to trigger public IPOWatch and Chittorgarh requests. Check
   those sites' terms and consider the request rate before enabling live fetching on a public app. IPO name suggestions
   are loaded from Chittorgarh's active and upcoming mainboard IPO list while live fetching is enabled.
6. Deploy. Streamlit installs the root `requirements.txt` and provides a `*.streamlit.app` URL.

The root requirements file installs this project in editable mode, so the package includes the tracked pretrained
model. Python 3.12 is recommended because the model was serialized with Python 3.12.15, scikit-learn 1.9.1,
NumPy 2.5.4, pandas 3.0.6 and joblib 1.6.0. scikit-learn does not guarantee model persistence compatibility across
versions; if the deployed runtime upgrades dependencies and emits compatibility warnings or fails to load the model,
pin the versions used to create the artifact or retrain it under the deployment environment.

## Persistence and limits

The bundled model and seed data are part of the GitHub source and are available after every restart. The SQLite
database, live GMP time series, saved overrides and cached fetch responses are written under `data/`, which is
runtime-local storage on Community Cloud and must not be treated as persistent. A restart or rebuild can remove that
history. Persistent user data would require an external database or storage service.

Community Cloud apps with no traffic for 12 hours go to sleep; a visitor can wake the app. CPU, memory and storage
limits apply and may change. The free host is suitable for a personal/demo app, not a guarantee of always-on service
or durable data.

## Local clone

The same tracked model works after a normal clone and project setup:

```bash
conda env create -f environment.yml
conda activate ipo-intel
ipo app
```

Training data is only required if you want to retrain or evaluate the model with `ipo train`.

## Official deployment references

- [Deploy a Streamlit app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [File organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [Dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
- [Limits and sleeping](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
