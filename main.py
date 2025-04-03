from utils.hf_helper import hf_login
from utils.wandb_helper import wandb_init
from model.model_loader import load_model_and_tokenizer
from data.dataset_loader import load_medical_dataset
from utils.prompt_formatter import formatting_prompts_func
from configs.lora_config import lora_config
from configs.train_config import train_config
from trainer.trainer import get_trainer

def main():
    hf_login()
    wandb_init()
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    dataset = load_medical_dataset()
    dataset = dataset.map(lambda x: formatting_prompts_func(x, tokenizer), batched=True)
    
    trainer = get_trainer(model, tokenizer, dataset, train_config, lora_config)
    trainer.train()
    trainer.save_model("Medical/final")

if __name__ == "__main__":
    main()
