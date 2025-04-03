from trl import SFTConfig

train_config = SFTConfig(
    output_dir="Medical",
    num_train_epochs=1,
    dataset_text_field="text",
    max_steps=60,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    learning_rate=2e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    logging_steps=10,
    save_steps=10,
    bf16=True,
    push_to_hub=False,
    report_to="wandb",
)
