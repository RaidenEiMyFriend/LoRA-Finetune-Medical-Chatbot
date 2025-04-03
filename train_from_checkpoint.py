import torch
from utils.hf_helper import hf_login
from utils.wandb_helper import wandb_init
from model.model_loader import load_model_and_tokenizer
from data.dataset_loader import load_medical_dataset
from utils.prompt_formatter import formatting_prompts_func
from configs.lora_config import lora_config
from configs.train_config import train_config
from trainer.trainer import get_trainer

def train_from_checkpoint(checkpoint_path):
    hf_login()
    wandb_init()

    print(f"Loading model from checkpoint: {checkpoint_path}")
    model, tokenizer = load_model_and_tokenizer(checkpoint_path)

    dataset = load_medical_dataset()
    dataset = dataset.map(lambda x: formatting_prompts_func(x, tokenizer), batched=True)


    trainer = get_trainer(model, tokenizer, dataset, train_config, lora_config)

    print("Resuming training from checkpoint...")
    trainer.train(resume_from_checkpoint=True)

    trainer.save_model(f"{checkpoint_path}/final")

if __name__ == "__main__":
    checkpoint_path = "Medical/final"  # Đường dẫn đến checkpoint cũ
    train_from_checkpoint(checkpoint_path)
