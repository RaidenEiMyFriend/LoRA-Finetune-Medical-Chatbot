# Fine-Tuning Qwen2.5-7B-Instruct on a Medical Dataset

This project demonstrates how to fine-tune the **Qwen2.5-7B-Instruct** model on a medical dataset using **LoRA fine-tuning**.

## 🚀 How to Run the Project

### 🔹 1. Install Dependencies
Make sure you have all required Python libraries installed:

```bash
pip install -r requirements.txt
```

### 🔹 2. Fine-Tune the Model from Scratch
Run the following command to train the model using your dataset:

```bash
python main.py
```

📌 This script will:
- ✅ Load the **Qwen2.5-7B-Instruct** model
- ✅ Load and preprocess the medical dataset
- ✅ Apply **LoRA fine-tuning**
- ✅ Save the trained model in `Medical/final/`

### 🔹 3. Monitor Training Progress
Training metrics such as **mean token accuracy** and **loss** are monitored using **Weights & Biases (wandb)**. Below are the visualizations of the training process:

#### Train Accuracy & Loss Plots
![Training Metrics](results/image1.png)

### 🔹 4. Resume Training from a Checkpoint
If you previously trained the model and want to continue training, run:

```bash
python train_from_checkpoint.py
```

📌 This script will:
- ✅ Load the last saved model from `Medical/final/`
- ✅ Resume training with the same dataset and parameters
- ✅ Save the updated model to the same directory

### 🔹 5. Run Inference (Chat with the Model)
Once the model is trained, you can test it by running:

```bash
python inference.py
```

📌 This script will:
- ✅ Load the fine-tuned model from `Medical/final/`
- ✅ Accept a medical question as input
- ✅ Generate a response using the chatbot
- ✅ Print the result to the console

#### Example Inference
![Inference Example](results/image2.png)

---


