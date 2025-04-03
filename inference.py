import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class MedicalChatbot:
    def __init__(self, model_path="Medical/final"):
        """
        Khởi tạo chatbot với mô hình đã fine-tune.
        
        Args:
            model_path (str): Đường dẫn đến model đã train xong.
        """
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate_response(self, question, max_new_tokens=2080, temperature=0.6, top_p=0.9):
        messages = [
            {"role": "system", "content": "You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. Please answer the following medical question."},
            {"role": "user", "content": f"Question: {question}"}
        ]
        model_input = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        model_input = model_input.to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                model_input, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                top_p=top_p, 
                do_sample=True
            )


        response_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

     
        response_parts = response_text.split("assistant\n")
        if len(response_parts) > 1:
            return response_parts[-1].strip()
        
        return response_text.strip()


if __name__ == "__main__":
    chatbot = MedicalChatbot(model_path="Medical/final")

    question = "A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or sneezing but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings, what would cystometry most likely reveal about her residual volume and detrusor contractions?"
    
    response = chatbot.generate_response(question)
    print("\n🩺 **Medical Chatbot Response:**")
    print(response)
