import os
import json
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ========= Path ========= videollama3_original_model
# 1) The original VideoLLaMA3 model in HF style (videollama3_original_model)
HF_BASE_MODEL = "CrashChat/videollama3_original_model"

# 2) The directory containing the trained LoRA checkpoint (including adapter_model and non_lora_trainables.bin)
LORA_PATH = "/ckpt/crashchat_7B_causal_reasoning/independent_monotask_models_causal_reasoning/checkpoint-xxxx"

# 3) Output directory: Generates the "merged and format-converted" CKPT file.
OUT_DIR = "/ckpt/independent_monotask_models_causal_reasoning"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ===== [1] Loading the HF base model =====
    print(f"[1] Load base HF VideoLLaMA3 model from: {HF_BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        HF_BASE_MODEL,
        torch_dtype=torch.bfloat16,     
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code=True,
    )

    # ===== [2] Apply non_lora_trainables first. =====
    nlt_path = os.path.join(LORA_PATH, "non_lora_trainables.bin")
    if os.path.exists(nlt_path):
        print(f"[2] Load non_lora_trainables from: {nlt_path}")
        non_lora = torch.load(nlt_path, map_location="cpu")

        non_lora = {
            (k[11:] if k.startswith("base_model.") else k): v
            for k, v in non_lora.items()
        }
        if any(k.startswith("model.model.") for k in non_lora):
            non_lora = {
                (k[6:] if k.startswith("model.") else k): v
                for k, v in non_lora.items()
            }

        missing, unexpected = base_model.load_state_dict(non_lora, strict=False)
        print(
            f"    loaded non_lora_trainables, missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
    else:
        print(f"[2] no non_lora_trainables.bin found in {LORA_PATH}, skip this step.")

    # ===== [3] Load and merge the LoRA model. =====
    print(f"[3] Load LoRA adapter from: {LORA_PATH}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
        torch_dtype=torch.bfloat16,    
    )

    print("[4] Merge LoRA weights into base model (merge_and_unload)...")
    merged_model = peft_model.merge_and_unload()  
    merged_model.to(torch.bfloat16)

    # ===== [5] Convert the state_dict according to the rules of `convert_hf_checkpoint`. =====
    print("[5] Convert merged state_dict keys for VideoLLaMA3 local format...")
    new_state_dict = {}
    for key, tensor in merged_model.state_dict().items():
        new_key = key.replace("vision_encoder", "vision_encoder.vision_encoder")
        if new_key != key:
            print(f"  Convert {key} -> {new_key}")
        new_state_dict[new_key] = tensor

    pt_path = os.path.join(OUT_DIR, "pytorch_model.bin")
    torch.save(new_state_dict, pt_path)
    print(f"[6] Saved merged & converted weights to: {pt_path}")

    # ===== [6] Write config.json =====
    config = merged_model.config.to_dict()
    config["vision_encoder"] = "DAMO-NLP-SG/SigLIP-NaViT"
    config["torch_dtype"] = "bfloat16"

    config_path = os.path.join(OUT_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"[7] Saved config.json to: {config_path}")

    # ===== [7] Save the tokenizer =====
    print("[8] Save tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            LORA_PATH,
            trust_remote_code=True,
            use_fast=False,
        )
        print("    tokenizer loaded from LoRA ckpt.")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            HF_BASE_MODEL,
            trust_remote_code=True,
            use_fast=False,
        )
        print("    tokenizer loaded from HF base.")

    tokenizer.save_pretrained(OUT_DIR)
    print(f"[9] Tokenizer saved to: {OUT_DIR}")

    print("\n[Done] Merged + converted (bf16) VideoLLaMA3 checkpoint is ready:")
    print(f"      {OUT_DIR}")


if __name__ == "__main__":
    main()
