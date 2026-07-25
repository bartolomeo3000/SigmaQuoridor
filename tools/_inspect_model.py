import torch
ckpt = torch.load("runs/models_7x7/best.pt", map_location="cpu", weights_only=False)
sd = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt
keys = list(sd.keys())
print("Keys sample:", keys[:8])
# filters: output channels of the first conv weight
first_weight = next(k for k in keys if "weight" in k and sd[k].ndim == 4)
print(f"First conv weight shape: {sd[first_weight].shape}  (key: {first_weight})")
# residual blocks: count unique block indices
res_keys = [k for k in keys if k.startswith("res")]
block_indices = set(k.split(".")[1] for k in res_keys)
print(f"Residual blocks: {len(block_indices)}")
