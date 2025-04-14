import os
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/scratch/user/joshua9/model_inference_streamlit.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/scratch/user/joshua9/hf_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Model configurations
model_configs = {
    "Qwen2.5-Coder-7B-Instruct-Original": {
        "name": "Qwen2.5-Coder-7B-Instruct (Original)",
        "path": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "use_quantization": True,
        "pipeline_params": {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "truncation": True,
        },
    },
    "Qwen2.5-Coder-7B-Instruct-FineTuned": {
        "name": "Qwen2.5-Coder-7B-Instruct (Fine-Tuned)",
        "path": "/scratch/user/joshua9/outputs/qwen25_coder_7b_lora/final_model",
        "use_quantization": True,
        "pipeline_params": {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "truncation": True,
        },
    },
    "DeepSeek-Coder-6.7B-Instruct-Original": {
        "name": "DeepSeek-Coder-6.7B-Instruct (Original)",
        "path": "deepseek-ai/DeepSeek-Coder-6.7B-Instruct",
        "use_quantization": True,
        "pipeline_params": {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "truncation": True,
        },
    },
    "DeepSeek-Coder-6.7B-Instruct-FineTuned": {
        "name": "DeepSeek-Coder-6.7B-Instruct (Fine-Tuned)",
        "path": "/scratch/user/joshua9/outputs/deepseek_coder_6.7b_lora/final_model",
        "use_quantization": True,
        "pipeline_params": {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "truncation": True,
        },
    },
    "Qwen1.5-0.5B-Original": {
        "name": "Qwen1.5-0.5B (Original)",
        "path": "Qwen/Qwen1.5-0.5B",
        "use_quantization": False,
        "pipeline_params": {
            "max_new_tokens": 700,
            "temperature": 0.7,
            "do_sample": True,
            "truncation": True,
        },
    },
    "Qwen1.5-0.5B-FineTuned": {
        "name": "Qwen1.5-0.5B (Fine-Tuned)",
        "path": "/scratch/user/joshua9/20240527_Fine_Tuning_Qwen_1_5_for_Coding/outputs/qwen_05b_code/best_model",
        "use_quantization": False,
        "pipeline_params": {
            "max_new_tokens": 700,
            "temperature": 0.7,
            "do_sample": True,
            "truncation": True,
        },
    },
    "gpt2-medium-Original": {
        "name": "gpt2-medium (Original)",
        "path": "gpt2-medium",
        "use_quantization": False,
        "pipeline_params": {
            "max_length": 700,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "truncation": True,
        },
    },
    "gpt2-medium-FineTuned": {
        "name": "gpt2-medium (Fine-Tuned)",
        "path": "/scratch/user/joshua9/outputs/gpt2_medium_code_full_all/best_model_gpt",
        "use_quantization": False,
        "pipeline_params": {
            "max_length": 700,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "truncation": True,
        },
    },
}

# Function to load a model
def load_model(model_path, use_quantization, cache_dir="/scratch/user/joshua9/hf_cache"):
    logger.info(f"Loading model from {model_path}...")
    try:
        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                cache_dir=cache_dir,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer.padding_side = "right"
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        
        # Fix tokenizer
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.config.pad_token_id = tokenizer.pad_token_id
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        
        model.eval()
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        st.error(f"Error loading model from {model_path}: {e}")
        return None, None

# Function to create pipeline
def create_pipeline(model, tokenizer, pipeline_params):
    try:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **pipeline_params,
        )
        return pipe
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        st.error(f"Error creating pipeline: {e}")
        return None

# Function to generate response
def generate_response(pipe, prompt, pipeline_params):
    try:
        max_tokens = pipeline_params.get("max_new_tokens", pipeline_params.get("max_length", 200))
        output = pipe(prompt, num_return_sequences=1)[0]["generated_text"]
        if len(output) > max_tokens + len(prompt):
            output = output[: max_tokens + len(prompt)]
        return output
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error generating response: {e}"

# Streamlit app
def main():
    st.title("Model Code Generation App")
    st.write("Select a model and enter a prompt to generate code (e.g., coin change problem).")

    # Initialize session state
    if "current_model_key" not in st.session_state:
        st.session_state.current_model_key = None
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "pipeline_params" not in st.session_state:
        st.session_state.pipeline_params = None

    # Model selection
    model_options = list(model_configs.keys())
    selected_model = st.selectbox(
        "Select Model",
        ["Select a model"] + [model_configs[key]["name"] for key in model_options],
        index=0,
    )

    # Load model if selection changes or no model is loaded
    if selected_model != "Select a model":
        selected_model_key = [key for key in model_configs if model_configs[key]["name"] == selected_model][0]
        if selected_model_key != st.session_state.current_model_key:
            # Clear previous model
            if st.session_state.pipeline is not None:
                del st.session_state.pipeline
                st.session_state.pipeline = None
                torch.cuda.empty_cache()
                logger.info("Cleared previous model and GPU memory")
            
            # Load new model
            config = model_configs[selected_model_key]
            st.write(f"Loading {config['name']}...")
            model, tokenizer = load_model(config["path"], config["use_quantization"])
            if model and tokenizer:
                pipeline_obj = create_pipeline(model, tokenizer, config["pipeline_params"])
                if pipeline_obj:
                    st.session_state.pipeline = pipeline_obj
                    st.session_state.pipeline_params = config["pipeline_params"]
                    st.session_state.current_model_key = selected_model_key
                    st.success(f"Loaded {config['name']} successfully!")
                else:
                    st.error(f"Failed to create pipeline for {config['name']}.")
            else:
                st.error(f"Failed to load {config['name']}.")
    else:
        if st.session_state.pipeline is not None:
            # Clear model if user selects "Select a model"
            del st.session_state.pipeline
            st.session_state.pipeline = None
            st.session_state.current_model_key = None
            st.session_state.pipeline_params = None
            torch.cuda.empty_cache()
            logger.info("Cleared model and GPU memory due to no selection")

    # Prompt input
    default_prompt = (
        "Write a Python function to solve the coin change problem: given a list of coin denominations "
        "and a target amount, return the minimum number of coins needed to make the amount. If the "
        "amount cannot be made, return -1. Include comments to explain the steps. Use dynamic "
        "programming for efficiency."
    )
    prompt = st.text_area("Enter your prompt:", value=default_prompt, height=200)

    # Generate button
    if st.button("Generate Response"):
        if st.session_state.pipeline is None:
            st.error("Please select a model before generating a response.")
        elif not prompt.strip():
            st.error("Prompt cannot be empty.")
        else:
            with st.spinner("Generating response..."):
                logger.info(f"Generating response for prompt: {prompt}")
                response = generate_response(st.session_state.pipeline, prompt, st.session_state.pipeline_params)
                st.subheader("Generated Response")
                st.code(response, language="python")

if __name__ == "__main__":
    main()
