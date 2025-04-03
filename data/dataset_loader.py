from datasets import load_dataset

def load_medical_dataset():
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train[0:240]", trust_remote_code=True)
    return dataset
