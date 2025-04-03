from trl import SFTTrainer

def get_trainer(model, tokenizer, dataset, train_config, lora_config):
    return SFTTrainer(
        model=model,
        args=train_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
