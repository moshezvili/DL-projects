from unsloth import FastLanguageModel
import torch
import json

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

max_seq_length = 8192  # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/tinyllama-bnb-4bit", # "unsloth/tinyllama" for 16bit loading
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    rope_scaling    = 4.0,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Currently only supports dropout = 0
    bias = "none",    # Currently only supports bias = "none"
    use_gradient_checkpointing = False, # @@@ IF YOU GET OUT OF MEMORY - set to True @@@
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("json", data_files="log_alpaca.jsonl", split = "train")

# Validate dataset
print(f"Dataset size: {len(dataset)}")
print(f"First example keys: {dataset[0].keys()}")
if len(dataset) > 0:
    print(f"Sample instruction: {dataset[0]['instruction'][:100]}...")
    print(f"Sample input length: {len(dataset[0]['input'])}")

dataset = dataset.map(formatting_prompts_func, batched = True,)

from trl import SFTConfig, SFTTrainer

print("Starting training...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = True, # Packs short sequences together to save time!
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 1,
        learning_rate = 2e-5,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.1,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
        save_steps = 5, # Save model every 5 steps
    ),
)

trainer_stats = trainer.train()
print("Training completed!")
print(f"Training stats: {trainer_stats}")

# Save the trained model
print("Saving model...")
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
print("Model saved to ./trained_model")

# Prepare for inference
print("Preparing for inference...")
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# Load test data
print("Loading test data...")
datas = []
try:
    with open("log_alpaca.jsonl", 'r') as f:
        for line in f:
            datas.append(json.loads(line))
    print(f"Loaded {len(datas)} test examples")
except Exception as e:
    print(f"Error loading test data: {e}")
    exit(1)

# Test inference with the first example
if len(datas) > 0:
    data = datas[0]
    print(f"Testing with input length: {len(data['input'])}")
    
    # Use full input instead of half
    test_input = data['input']
    
    try:
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    data['instruction'], # instruction
                    test_input, # input - using full input
                    "", # output - leave this blank for generation!
                )
            ], return_tensors = "pt").to(device)

        print("Generating response...")
        outputs = model.generate(**inputs,
                                 max_new_tokens = 200,
                                 temperature    = 0.2,
                                 use_cache = True,
                                 do_sample = True,
                                 pad_token_id = tokenizer.eos_token_id)
        
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        print("="*50)
        print("INFERENCE RESULTS:")
        print("="*50)
        print(f"Input: {test_input[:200]}...")
        print(f"Generated Output: {decoded_outputs[0]}")
        print("="*50)
        
        # Save results to file
        with open("inference_results.txt", "w", encoding="utf-8") as f:
            f.write(f"Input: {test_input}\n\n")
            f.write(f"Generated Output: {decoded_outputs[0]}\n")
        print("Results saved to inference_results.txt")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Make sure you have sufficient GPU memory and CUDA is available")
else:
    print("No test data available for inference")

