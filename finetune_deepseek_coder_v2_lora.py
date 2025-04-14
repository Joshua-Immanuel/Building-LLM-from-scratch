import os
import torch
import logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/scratch/user/joshua9/hf_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
log_file = "/scratch/user/joshua9/outputs/deepseek_coder_6.7b_lora/train.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Clear GPU memory
torch.cuda.empty_cache()
logger.info("Cleared GPU memory")

# Model and dataset
model_name = "deepseek-ai/DeepSeek-Coder-6.7B-Instruct"
dataset_name = "sahil2801/CodeAlpaca-20k"
output_dir = "/scratch/user/joshua9/outputs/deepseek_coder_6.7b_lora"

# Cache directory
cache_dir = "/scratch/user/joshua9/hf_cache"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Load dataset
logger.info("Loading dataset...")
try:
    dataset = load_dataset(dataset_name, cache_dir=cache_dir, split="train")
    # Split dataset into train (90%) and validation (10%)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
    logger.info(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation examples")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    print(f"Error loading dataset: {e}")
    exit(1)

# Load model with 4-bit quantization
logger.info("Loading model and tokenizer...")
try:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        cache_dir=cache_dir,
        device_map={"": 0},
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
    model.config.pad_token_id = tokenizer.pad_token_id
except Exception as e:
    logger.error(f"Error loading model: {e}")
    print(f"Error loading model: {e}")
    exit(1)

# Preprocess dataset
def preprocess_function(examples):
    prompts = [
        f"### Instruction:\n{instr}\n### Response:\n{out}\n<|END_OF_TEXT|>"
        for instr, out in zip(examples["instruction"], examples["output"])
    ]
    return tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )

logger.info("Preprocessing dataset...")
try:
    tokenized_train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    logger.info("Dataset preprocessing complete")
except Exception as e:
    logger.error(f"Error preprocessing dataset: {e}")
    print(f"Error preprocessing dataset: {e}")
    exit(1)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
logger.info("Applied LoRA configuration")

# Training configuration
sft_config = SFTConfig(
    max_seq_length=512,
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch size ~16
    num_train_epochs=1,
    learning_rate=5e-5,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    remove_unused_columns=False,
)

# Initialize trainer
logger.info("Initializing trainer...")
try:
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=sft_config,
    )
except Exception as e:
    logger.error(f"Error initializing trainer: {e}")
    print(f"Error initializing trainer: {e}")
    exit(1)

# Train model
logger.info("Starting fine-tuning...")
try:
    history = trainer.train()
    logger.info(f"Training history: {history}")
    print("Training history:", history)
except Exception as e:
    logger.error(f"Error during training: {e}")
    print(f"Error during training: {e}")
    exit(1)

# Save model
logger.info("Saving fine-tuned model...")
try:
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    logger.info(f"Model saved to {output_dir}/final_model")
    print(f"Model saved to {output_dir}/final_model")
except Exception as e:
    logger.error(f"Error saving model: {e}")
    print(f"Error saving model: {e}")