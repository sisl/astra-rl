"""
llamaGuard.py (TODO)
Moderator to call into the Llama Guard 3 engine.
Warning: this code directly downloads llamaguard 3 and requires a lot of memeory (roughly 112 MiB)
"""

from typing import Sequence
from astra_rl.core.moderator import Moderator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
### Do we need the following two imports?
# from accelerate.inference import prepare_pippy
# from accelerate import infer_auto_device_map


class LlamaGuardModerator(Moderator[str, str]):
    """Moderator that wraps LlamaGuard 3 (8b) for toxicity detection.

    https://huggingface.co/meta-llama/Llama-Guard-3-8B

    Attributes:
        harm_category (str): The category of harm to detect (default is "toxicity"); see below.
        variant (str): The variant of the Detoxify model to use (default is "original").

    Notes:
        Possible harm categories include:
        S1 (Violent Crimes), S2 (Non-Violent Crimes), S3 (Sex-Related Crimes),
        S4 (Child Sexual Exploitation), S5 (Defamation), S6 (Specialized Advice),
        S7 (Privacy), S8 (Intellectual Property), S9 (Indiscriminate Weapons),
        S10 (Hate), S11 (Suicide & Self-Harm), S12 (Sexual Content),
        S13 (Elections), S14 (Code Interpreter Abuse)
    """

    def __init__(self, DEVICE: str = "cuda"):
        # set model and tokenizer for llamaguard moderator
        # note: this is a large model! Check your space capabilities.
        model_id = "meta-llama/Llama-Guard-3-8B"
        self.device = torch.device(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to(self.device)

        self.safe_id = self.tokenizer.convert_tokens_to_ids("safe")
        self.unsafe_id = self.tokenizer.convert_tokens_to_ids("unsafe")
        self.safe_seq = self.tokenizer.encode("\n\nsafe", add_special_tokens=False)
        self.unsafe_seq = self.tokenizer.encode("\n\nunsafe", add_special_tokens=False)

        self.model.eval()

    def moderate(self, x: Sequence[str]) -> Sequence[float]:
        results = []
        with torch.no_grad():
            for s in x:
                chat = [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": s},
                ]

                # tokenize input
                input_ids = self.tokenizer.apply_chat_template(
                    chat, return_tensors="pt"
                ).to(self.device)

                # create final safe and unsafe output sequences
                # safe_seq = (torch.cat([input_ids, torch.tensor(self.safe_seq).repeat(input_ids.size(0), 1)], dim=-1))
                unsafe_seq = torch.cat(
                    [
                        input_ids,
                        torch.tensor(self.unsafe_seq, device=self.device).repeat(
                            input_ids.size(0), 1
                        ),
                    ],
                    dim=-1,
                )

                # make the model score the sequences
                # safe_seq_scored = self.model(safe_seq).logits
                unsafe_seq_scored = self.model(unsafe_seq).logits

                # and then compute the relavent probabilities
                # probs_safe = safe_seq_scored[0][input_ids.size(-1)]
                probs_unsafe = unsafe_seq_scored[0][input_ids.size(-1)]

                # probs_safe = torch.nn.functional.softmax(probs_safe, dim=-1)
                probs_unsafe = torch.nn.functional.softmax(probs_unsafe, dim=-1)

                # prob_safe = probs_safe[self.safe_id]
                prob_unsafe = probs_unsafe[self.unsafe_id]
                results.append(prob_unsafe.item())

                # optional: if you also want to return Llama Guard's original classification
                # (safe v unsafe label, categories of harm violated) uncomment the following 4 lines
                # output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
                # prompt_len = input_ids.shape[-1]
                # labelList = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
                # labelList = labelList.strip().split("\n")

        return results
