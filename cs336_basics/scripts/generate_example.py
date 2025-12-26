import logging
import torch
import hydra

from dataclasses import dataclass, asdict

from omegaconf import DictConfig, OmegaConf

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.models import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.serialization import load_checkpoint
from cs336_basics.generation import generate


@dataclass
class TransformerLMConfig:
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    rope_theta: float

TOKENIZER_PATH = "/public/home/wangfei/user_home/gzy/proj/CS336/results/tokenizer/tinystories-train-10k"
SERIALIZATION_PATH = "/public/home/wangfei/user_home/gzy/proj/CS336/results/ckpts"
PROMPT = "Hello, who are you?"

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    log = logging.getLogger(__name__)
    log.info(f"Full Config:\n{OmegaConf.to_yaml(cfg)}")
    
    log.info(f"Instantiating model <{cfg.model._target_}>")

    model = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating optimizer <{cfg.optimizer._target_}>")

    optimizer = hydra.utils.instantiate(
        cfg.optimizer,
        params=model.parameters(),
    )

    loaded_iterations = load_checkpoint(SERIALIZATION_PATH, model, optimizer)

    vocab_filepath = f"{TOKENIZER_PATH}/vocab.json"
    merges_filepath = f"{TOKENIZER_PATH}/merges.json"
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath)

    inputs = tokenizer.encode(PROMPT)

    inputs = torch.tensor(inputs, dtype=torch.int64).reshape(1, -1)

    output = generate(model, inputs, max_gen_len=256, temperature=0.7, top_p=0.7, eos_id=256)

    output_ls = output[0, :]

    output_text = tokenizer.decode(output_ls.tolist())

    print(output_text)

if __name__ == "__main__":
    main()