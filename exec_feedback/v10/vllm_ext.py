"""vLLM worker extension, mixed into every vLLM worker through `worker_extension_cls` (vLLM 0.11.2): load a full
weight file into the running engine in place and report the checksum of the weights the engine holds. Method names
carry an efa_ prefix because vLLM refuses extension attributes that clash with the worker's own."""
from __future__ import annotations


class WeightLoader:
    def efa_load(self, path: str) -> list[float]:
        import torch
        from safetensors.torch import load_file

        # CPU tensors: load_weights copies each into the existing parameter storage, so captured CUDA graphs stay valid
        # and no second copy of the model is allocated on the GPU
        weights = load_file(path, device="cpu")
        self.model_runner.model.load_weights(weights=iter(weights.items()))
        del weights
        torch.cuda.synchronize(self.device)
        return self.efa_checksum()

    def efa_checksum(self) -> list[float]:
        from exec_feedback.v10.weights import checksum

        return checksum(self.model_runner.model.parameters())
