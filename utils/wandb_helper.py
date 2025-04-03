import wandb
from kaggle_secrets import UserSecretsClient

def wandb_init():
    user_secrets = UserSecretsClient()
    WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project="Qwen2.5-7B-Instruct-Medical-Chatbot", name="1-epoch")
