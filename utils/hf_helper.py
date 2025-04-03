from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

def hf_login():
    user_secrets = UserSecretsClient()
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    login(token=HF_TOKEN)
