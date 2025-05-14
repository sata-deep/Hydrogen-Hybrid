import torch
from transformers import GPT2Config
from transformers import BertTokenizerFast
from transformers import GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import argparse
import os # Import os module

parser = argparse.ArgumentParser()

# --- MODIFIED DEFAULTS ---
parser.add_argument('--tokenizer', default='tokenizer', # Assumes vocab.txt is in a dir named 'tokenizer'
	                    help='path to vocab directory (default: ./tokenizer/ which should contain vocab.txt)')

parser.add_argument('--train_data', default='train.dat', # Changed default
	                    help='path to training data (default: ./train.dat)')

parser.add_argument('--valid_data', default='valid.dat', # Changed default
	                    help='path to validation data (default: ./valid.dat)')

parser.add_argument('--output_dir', default='TrainedModel', # Changed default
	                    help='output directory (default: ./TrainedModel)')
# --- END OF MODIFIED DEFAULTS ---

parser.add_argument('--n_position', type=int, default=256,
	                    help='the number of position')

parser.add_argument('--n_embd', type=int, default=180,
	                    help='the number of embedding')

parser.add_argument('--n_layer', type=int, default=6,
	                    help='the number of layer')

parser.add_argument('--n_head', type=int, default=2,
	                    help='the number of head. It is divisible by n_embd')

parser.add_argument('--epochs', type=int, default=500,
	                    help='epochs')

parser.add_argument('--train_batch', type=int, default=256,
	                    help='train batch size')

parser.add_argument('--valid_batch', type=int, default=256,
	                    help='valid batch size')
args = parser.parse_args()

# --- ENSURE OUTPUT DIRECTORY EXISTS ---
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    print(f"Created output directory: {args.output_dir}")
# --- END OF ENSURE OUTPUT DIRECTORY ---

print(f"Using Tokenizer from: {args.tokenizer}")
print(f"Using Training data: {args.train_data}")
print(f"Using Validation data: {args.valid_data}")
print(f"Outputting to: {args.output_dir}")


# BertTokenizerFast.from_pretrained expects a directory containing vocab.txt
# So, if args.tokenizer is 'tokenizer', it will look for 'tokenizer/vocab.txt'
tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer, max_len=512, do_lower_case=False)

config = GPT2Config(
    vocab_size = tokenizer.vocab_size, # Dynamically set vocab_size from tokenizer
    n_positions = args.n_position,
    n_embd = args.n_embd,
    n_layer = args.n_layer,
    n_head = args.n_head,
    bos_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id, # Use appropriate special tokens
    eos_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id, # Use appropriate special tokens
)
# NOTE: In this code, the vocab_size, bos_token_id, eos_token_id are derived from the the tokenizer itself.

model = GPT2LMHeadModel(config=config)

print(f"Loading datasets: train='{args.train_data}', validation='{args.valid_data}'")
datasets = load_dataset("text", data_files={"train": args.train_data, "validation": args.valid_data})

def tokenize_function(examples):
    # Ensure tokenizer is called with truncation and padding if needed,
    # though group_texts handles block_size later.
    # Here, we just tokenize.
    return tokenizer(examples["text"], truncation=False) # truncation=False because group_texts will handle it

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

block_size = args.n_position # Use n_position for block_size to match model's context window
print(f"Using block_size (context window): {block_size}")

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the small remainder, and if the total_length is less than block_size, we skip this batch.
    if total_length < block_size:
        return {k: [] for k in examples.keys()}
        
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000, # This is batch_size for mapping, not training
    num_proc=8, # Be careful with num_proc, ensure you have enough cores and memory
)

# Filter out empty examples that might result from group_texts if total_length < block_size
lm_datasets["train"] = lm_datasets["train"].filter(lambda example: len(example['input_ids']) > 0)
lm_datasets["validation"] = lm_datasets["validation"].filter(lambda example: len(example['input_ids']) > 0)


if len(lm_datasets["train"]) == 0 or len(lm_datasets["validation"]) == 0:
    print("Error: Training or validation dataset is empty after processing. Check your data and block_size.")
    exit()

print("Example from processed training data: ", lm_datasets["train"][0])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False # For GPT-2, mlm should be False. MLM is for BERT-like models.
)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.train_batch,
    per_device_eval_batch_size=args.valid_batch,
    save_steps=5000,
    save_total_limit=80,
    do_train=True,
    do_eval=True,
    evaluation_strategy='steps',
    # prediction_loss_only=True, # Setting this to False allows metrics like perplexity if eval_dataset has labels
    logging_steps=500, # Added for better visibility during training
    eval_steps=1000,   # Evaluate every 1000 steps (adjust as needed)
    load_best_model_at_end=True, # Optional: loads the best model at the end of training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator
)

print("Starting training...")
trainer.train()

print(f"Training finished. Saving model to {args.output_dir}")
trainer.save_model(args.output_dir) # This will save to the 'TrainedModel' directory or whatever args.output_dir is
tokenizer.save_pretrained(args.output_dir) # Also save the tokenizer
print("Model and tokenizer saved.")
