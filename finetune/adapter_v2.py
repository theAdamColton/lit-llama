"""
Expects training data as a json, with 
"""
import os
import random
import json
import sys
import time
from pathlib import Path
import shutil

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.adapter_v2 import LLaMA, LLaMAConfig, mark_only_adapter_as_trainable, adapter_state_from_state_dict
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt
from lightning.fabric.strategies import DeepSpeedStrategy


eval_interval = 100
save_interval = 50
eval_iters = 20
log_interval = 1
devices = 1

# Hyperparameters
learning_rate = 9e-3
batch_size = 64 // devices
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size
epoch_size = 50000  # train dataset size
num_epochs = 5
max_iters = num_epochs * epoch_size // devices
weight_decay = 0.02
max_seq_length = 256  # see scripts/prepare_alpaca.py
warmup_steps = epoch_size * 2 // micro_batch_size // devices

ds_config = {
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "zero_optimization": {"stage": 2},
}


class MixedAlpacaConceptDataset(IterableDataset):
    """
    Gives batches of mixed instruction-following, image_retrieval, and
    image_description tasks
    """
    def __init__(self, concept_jsonpath: str, alpaca_pt_path: str, tokenizer: Tokenizer, alpaca_chance=0.3):
        # json field names prefixes
        self.concept_prompt_types = ["retrieval_", "describe_"]

        self.tokenizer = tokenizer
        
        self.alpaca_chance = alpaca_chance

        with open(concept_jsonpath, "r") as f:
            self.concept_d = json.load(f)

        self.alpaca_d = torch.load(alpaca_pt_path)
        
        self.batch_size = batch_size


    def __iter__(self):
        return self

    def __next__(self):
        """
        returns a random prompt, response
        """
        if torch.rand((1,)).item() >= self.alpaca_chance:
            i = torch.randint(len(self.concept_d), (1,)).item()
            prompt_type = random.choice(self.concept_prompt_types)
            x = torch.LongTensor(self.concept_d[i][f"{prompt_type}input_ids"])
            y = torch.LongTensor(self.concept_d[i][f"{prompt_type}labels"])
        else:
            i = torch.randint(len(self.alpaca_d), (1,)).item()
            x = self.alpaca_d[i]["input_ids"].to(torch.int64)
            y = self.alpaca_d[i]["labels"].to(torch.int64)

        block_size = 256
        x = x[:block_size]
        y = y[:block_size]
        
        return x, y

def collate_token_batch(batch):
    input_ids, labels = zip(*batch)

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    return x, y

def get_dataloader(concept_dataset: MixedAlpacaConceptDataset, fabric: L.Fabric, **kwargs):
    dl= DataLoader(concept_dataset, collate_fn=collate_token_batch, **kwargs)
    return fabric.setup_dataloaders(dl)


def load_datasets(path, tokenizer):
    train_ds = MixedAlpacaConceptDataset(path + "/llama_adapter_v2_custom/concepts_tokenized.json", path + "alpaca/train.pt", tokenizer)
    test_ds = MixedAlpacaConceptDataset(path + "/llama_adapter_v2_custom/concepts_tokenized_test.json", path + "alpaca/test.pt", tokenizer)
    return train_ds, test_ds


def main(
    # Should contain 2 json files, train and test
    data_dir: str = "data/", 
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth",
    tokenizer_path: str = "checkpoints/lit-llama/tokenizer-modified.model",
    out_dir: str = "out/adapter/alpaca",
):
    tokenizer = Tokenizer(tokenizer_path)

    fabric = L.Fabric(
        accelerator="cuda", 
        devices=1, 
        precision="bf16-true",
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_ds, val_ds = load_datasets(data_dir, tokenizer)
    train_dl, val_dl = get_dataloader(train_ds, fabric, batch_size=micro_batch_size), get_dataloader(val_ds, fabric, batch_size=micro_batch_size)

    config = LLaMAConfig(block_size=max_seq_length)

    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(
            f"Can't find the pretrained weights at {pretrained_path}."
            " Please follow the instructions in the README to download them."
        )
    checkpoint = torch.load(pretrained_path)

    with fabric.init_module():
        model = LLaMA(config)
        # strict=False because missing keys due to adapter weights and bias/scale not containted in state dict
        model.load_state_dict(checkpoint, strict=False)

    mark_only_adapter_as_trainable(model)


    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    num_wte_params = sum([p.numel() for n, p in model.named_parameters() if p.requires_grad and "wte" in n])
    num_b_s_params = sum([p.numel() for n, p in model.named_parameters() if p.requires_grad and ("_bias" in n or "_scale" in n)])
    total_params = sum([p.numel() for p in model.parameters()])
    print(f"{num_params / 1000000:.4f}M trainable {total_params / 1000000:.4f}M total")
    print(f"{num_wte_params / 1000000:.4f}M wte {num_b_s_params / 1000000:.4f}M scale/bias")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_dl, val_dl, out_dir, tokenizer)

    # Save the final checkpoint at the end of training
    save_model_checkpoint(fabric, model, os.path.join(out_dir, "lit-llama-adapterv2-finetuned.pth"), config)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dl: DataLoader,
    val_dl: DataLoader,
    out_dir: str,
    tokenizer: Tokenizer,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0

    for iter_num in range(max_iters):

        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets = next(iter(train_dl))

        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_steps != 0)):
            fabric.backward(loss / gradient_accumulation_steps)

        if (iter_num + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
                
            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_dl, tokenizer)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            if step_count % save_interval == 0:
                print(f"Saving adapter weights to {out_dir}")
                # TODO: Provide a function/script to merge the adapter weights with pretrained weights
                save_model_checkpoint(fabric, model, os.path.join(out_dir, f"iter-{iter_num:06d}.pth"), model.config)

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")


def generate_response(model, instruction, tokenizer, input=""):
    sample = {"instruction": instruction, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
        temperature=0.8,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dl, tokenizer:Tokenizer) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = next(iter(val_dl))
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    val_loss = losses.mean()

    # produce an example:
    # V0.0 prompt:
    instruction = "What is the meaning of life?"
    output = generate_response(model, instruction, tokenizer)
    fabric.print(instruction)
    fabric.print(output)

    model.train()
    return val_loss.item()

def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss
    
def save_model_checkpoint(fabric, model, file_path, config):
    file_path = Path(file_path)

    if isinstance(fabric.strategy, DeepSpeedStrategy):
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

        tmp_path = file_path.with_suffix(".tmp")
        fabric.save(tmp_path, {"model": model})
        fabric.barrier()
        if fabric.global_rank == 0:
            # Create a consolidated checkpoint with the same name next to the deepspeed checkpoint
            # and only keep the adapter weights
            state_dict = get_fp32_state_dict_from_zero_checkpoint(tmp_path)
            state_dict = adapter_state_from_state_dict(state_dict, config)
            torch.save(state_dict, file_path)
            shutil.rmtree(tmp_path)
    else:
        state_dict = adapter_state_from_state_dict(model.state_dict(), config)
        if fabric.global_rank == 0:
            torch.save(state_dict, file_path)
        fabric.barrier()


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    CLI(main)
