import os
import torch
import argparse
import pandas as pd
import random
from pymatgen.core.composition import Composition # Ensure pymatgen is installed
from transformers import BertTokenizerFast, GPT2LMHeadModel

# --- Argument Parser Setup ---
parser = argparse.ArgumentParser(description='Generate chemical formulas using a trained GPT-2 model.')

# Script behavior arguments
parser.add_argument("--loop_num", type=int, default=100, help="Number of sequences to attempt to generate.")
parser.add_argument("--num_beam", type=int, default=1, help="Beam size for generation (1 for greedy).")
parser.add_argument("--max_length", type=int, default=256, help="Max length of the generated sequence (including prompt).")

# Paths and model configuration arguments
parser.add_argument("--tokenizer", type=str, default='TrainedModel',
                    help="Path to the directory containing the tokenizer's vocab.txt and config files (default: ./TrainedModel).")
parser.add_argument("--model_name", type=str, default='GPT2LMHeadModel',
                    help="Name of the model class to use (default: GPT2LMHeadModel).")
parser.add_argument("--model_path", type=str, default='TrainedModel',
                    help="Path to the directory containing the trained model files (pytorch_model.bin, config.json) (default: ./TrainedModel).")
parser.add_argument("--save_path", type=str, default='GeneratedFormulas',
                    help="Directory to save the generated sequences CSV file (default: ./GeneratedFormulas).")

args = parser.parse_args()

# --- Ensure Save Directory Exists ---
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print(f"Created save directory: {args.save_path}")

# --- Load Tokenizer ---
print(f"Loading tokenizer from: {args.tokenizer}")
try:
    # Consider adding padding_side='left' if the warning persists and is problematic,
    # ensure pad_token is also set (e.g., tokenizer.pad_token = tokenizer.eos_token or ensure [PAD] is used)
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer, max_len=args.max_length, do_lower_case=False)
    if tokenizer.pad_token is None:
        # GPT-2 often uses EOS as PAD if no specific PAD token is set during training/tokenization
        # Or, if [PAD] is in your vocab (it is), BertTokenizerFast should pick it up.
        # tokenizer.pad_token = tokenizer.eos_token # Common for GPT models
        print(f"Tokenizer pad_token not set. Using pad_token_id={tokenizer.pad_token_id} (should be [PAD]'s id if available). In generate, eos_token_id is used for padding.")

except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print(f"Please ensure that '{args.tokenizer}' contains the necessary tokenizer files (e.g., vocab.txt, tokenizer_config.json).")
    exit()

# --- Load Model ---
print(f"Loading model '{args.model_name}' from: {args.model_path}")
model_loaded = False
if args.model_name == 'GPT2LMHeadModel':
    try:
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
        model_loaded = True
    except Exception as e:
        print(f"Error loading GPT2LMHeadModel: {e}")
        print(f"Please ensure that '{args.model_path}' contains the necessary model files (e.g., pytorch_model.bin, config.json).")
        exit()
else:
    print(f"Error: Model name '{args.model_name}' is not supported by this script.")
    print("Currently, only 'GPT2LMHeadModel' is supported.")
    exit()

if not model_loaded:
    print("Failed to load the model. Exiting.")
    exit()

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# --- Element List for Prompt Generation ---
mapping_list = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",	"Al",	"Si",	"P",	"S",	"Cl",	"Ar",	"K", "Ca",	"Sc",	"Ti",	"V",	"Cr",	"Mn",	"Fe",	"Co",	"Ni",	"Cu",	"Zn",	"Ga",	"Ge",	"As",	"Se",	"Br",	"Kr",	"Rb",	"Sr",	"Y",	"Zr","Nb","Mo",	"Tc",	"Ru",	"Rh",	"Pd",	"Ag",	"Cd",	"In",	"Sn",	"Sb",	"Te",	"I",	"Xe",	"Cs",	"Ba",	"La",	"Ce",	"Pr",	"Nd",	"Pm",	"Sm",	"Eu",	"Gd",	"Tb",	"Dy",	"Ho",	"Er",	"Tm",  "Yb",	"Lu",	"Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",	"Hg",	"Tl",	"Pb",	"Bi",	"Po",	"At",	"Rn",	"Fr",	"Ra",	"Ac",	"Th",	"Pa",	"U",	"Np",	"Pu",	"Am",	"Cm",	"Bk",	"Cf",	"Es", "Fm",	"Md",	"No",	"Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg","Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]

# --- Generation Loop ---
print(f"\nStarting generation of {args.loop_num} sequences...")
all_generated_texts = []
for i in range(args.loop_num):
    prompt_elements = [mapping_list[random.randint(0, len(mapping_list)-1)] for _ in range(4)]
    input_str = " ".join(prompt_elements) + " "
    
    # Encode the prompt and create an attention mask
    encoded_prompt = tokenizer.encode(input_str, add_special_tokens=True, return_tensors="pt")
    input_ids = encoded_prompt.to(device)
    attention_mask = torch.ones_like(input_ids, device=device) # Create attention mask

    # prompt_length_tokens = input_ids.shape[1] # Length of the tokenized prompt

    output_sequences = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask, # *** ADDED ATTENTION MASK ***
        max_length=args.max_length,
        num_beams=args.num_beam,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id # Use EOS token for padding during generation
    )

    special_tokens_to_remove_manually = ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]']

    for generated_sequence_ids in output_sequences:
        text = tokenizer.decode(generated_sequence_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        
        decoded_text_full = text
        for spec in special_tokens_to_remove_manually:
            decoded_text_full = decoded_text_full.replace(spec, "")
        
        generated_part_text = decoded_text_full.strip()
        # Remove prompt. Using strip on input_str ensures leading/trailing spaces on it match behavior
        prompt_to_remove = input_str.strip()
        if generated_part_text.startswith(prompt_to_remove):
             generated_part_text = generated_part_text[len(prompt_to_remove):].strip()

        all_generated_texts.append(generated_part_text)

    if (i + 1) % 10 == 0 or (i + 1) == args.loop_num:
        print(f"  Generated {i+1}/{args.loop_num} raw sequences. Sample: '{all_generated_texts[-1][:100]}...'")


# --- Post-processing and Filtering ---
print("\nPost-processing generated texts...")
tmp_list = []
for idx in range(len(all_generated_texts)):
    tmp = all_generated_texts[idx]
    x = tmp.split(".") 

    for part_index, tmp_text_part in enumerate(x):
        cleaned_part = tmp_text_part.strip()
        if cleaned_part and (len(cleaned_part) > 1): # Changed from >2 to >1 to allow short elements/formulas
            tmp_list.append(cleaned_part)

tmp_list = list(set(tmp_list)) # Unique segments
print(f"Found {len(tmp_list)} unique segments after splitting and initial filtering.")

formulas=[]
invalid_format_count = 0
skipped_segments_for_debug = []

for s in tmp_list:
    if "<" in s: 
        skipped_segments_for_debug.append(f"REASON: Contains '<' | SEGMENT: {s}")
        invalid_format_count += 1 # Count this as invalid
        continue
    
    original_segment_tokens = s.split()
    if not original_segment_tokens:
        skipped_segments_for_debug.append(f"REASON: Empty after split | SEGMENT: {s}")
        invalid_format_count += 1 # Count this as invalid
        continue

    # *** MODIFICATION START: Extract leading element tokens ***
    formula_tokens = []
    for token in original_segment_tokens:
        if token in mapping_list:
            formula_tokens.append(token)
        else:
            # Stop at the first non-element token (e.g., a digit, or other symbol)
            break 
    
    if not formula_tokens:
        skipped_segments_for_debug.append(f"REASON: No valid element tokens found at start | SEGMENT: {s}")
        invalid_format_count +=1
        continue
    # *** MODIFICATION END ***

    # Apply filters to the extracted formula_tokens
    if len(set(formula_tokens)) == 1 and len(formula_tokens) > 1: 
        # This filter skips formulas like "H H" (H2) or "O O O" (O3).
        # Consider if this is desired behavior.
        skipped_segments_for_debug.append(f"REASON: Single unique element repeated | SEGMENT: {s} | FORMULA_TOKENS: {' '.join(formula_tokens)}")
        # invalid_format_count +=1 # Original code didn't increment for this, decide if it's an "error"
        continue # Skip this segment from becoming a formula

    if len(set(formula_tokens)) > 8: # Max 8 unique elements
        skipped_segments_for_debug.append(f"REASON: Too many unique elements | SEGMENT: {s} | FORMULA_TOKENS: {' '.join(formula_tokens)}")
        invalid_format_count +=1
        continue
    
    composition_dict = {}
    for el in formula_tokens: 
        composition_dict[el] = composition_dict.get(el, 0) + 1

    if not composition_dict: # Should not be reached if formula_tokens is not empty
        skipped_segments_for_debug.append(f"REASON: Empty composition_dict (unexpected) | SEGMENT: {s} | FORMULA_TOKENS: {' '.join(formula_tokens)}")
        invalid_format_count +=1
        continue

    if sum(composition_dict.values()) > 30: # Max 30 atoms in total
        skipped_segments_for_debug.append(f"REASON: Too many total atoms | SEGMENT: {s} | FORMULA_TOKENS: {' '.join(formula_tokens)}")
        invalid_format_count +=1
        continue
        
    try:
        comp = Composition(composition_dict)
        formulas.append(comp.to_pretty_string()) 
    except Exception as e: 
        skipped_segments_for_debug.append(f"REASON: Pymatgen error ({e}) | SEGMENT: {s} | DICT: {composition_dict} | FORMULA_TOKENS: {' '.join(formula_tokens)}")
        invalid_format_count +=1
        continue
            
total_initial_segments = len(tmp_list)
final_valid_formulas = len(formulas) # This was unused, len(formulas) is used later

formulas = sorted(list(set(formulas))) 

print(f"Total unique segments processed: {total_initial_segments}")
if invalid_format_count > 0:
    print(f"Segments skipped or yielding no valid formula part: {invalid_format_count}") # Reworded slightly
    print("\n--- DEBUG: SKIPPED SEGMENTS / FAILED EXTRACTIONS ---")
    for item in skipped_segments_for_debug[:20]: 
        print(item)
    if len(skipped_segments_for_debug) > 20:
        print(f"... and {len(skipped_segments_for_debug) - 20} more.")
    print("--- END DEBUG ---")

print(f"Valid formulas generated: {len(formulas)}")

if formulas:
    df1 = pd.DataFrame(formulas, columns=['pretty_formula'])
    output_csv_path = os.path.join(args.save_path, 'generated_formulas.csv')
    df1.to_csv(output_csv_path, index=False)
    print(f"Generated formulas saved to: {output_csv_path}")
else:
    print("No valid formulas were generated after filtering.")

