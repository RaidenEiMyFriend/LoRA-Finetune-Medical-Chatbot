🚀 How to Run the Project
🔹 1. Install Dependencies
Make sure you have all required Python libraries installed:

bash
Copy
Edit
pip install -r requirements.txt
🔹 2. Fine-Tune the Model from Scratch
Run the following command to train the model using your dataset:

bash
Copy
Edit
python main.py
📌 This script will:
✅ Load the Qwen2.5-7B-Instruct model
✅ Load and preprocess the medical dataset
✅ Apply LoRA fine-tuning
✅ Save the trained model in Medical/final/

🔹 3. Resume Training from a Checkpoint
If you previously trained the model and want to continue training, run:

bash
Copy
Edit
python train_from_checkpoint.py
📌 This script will:
✅ Load the last saved model from Medical/final/
✅ Resume training with the same dataset and parameters
✅ Save the updated model to the same directory

🔹 4. Run Inference (Chat with the Model)
Once the model is trained, you can test it by running:

bash
Copy
Edit
python inference.py
📌 This script will:
✅ Load the fine-tuned model from Medical/final/
✅ Accept a medical question as input
✅ Generate a response using the chatbot
✅ Print the result to the console