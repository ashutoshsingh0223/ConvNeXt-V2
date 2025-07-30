import argparse
from collections import OrderedDict

import torch

def reverse_remap_checkpoint_keys(ckpt):
    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        parts = k.split('.')
        last = parts[-1]
        key_base = '.'.join(parts[:-1])

        # --- Identify key categories ---
        is_downsample = k.startswith("downsample_layers.")
        is_downsample_0_0 = k.startswith("downsample_layers.0.0")
        is_downsample_1_2_3 = any(k.startswith(f"downsample_layers.{i}.0") for i in [1, 2, 3])
        is_pwconv = 'pwconv' in k
        is_norm = 'norm' in k

        # --- Insert 'ln' ---
        if is_downsample_1_2_3 or is_norm:
            parts.insert(-1, 'ln')
            k = '.'.join(parts)

        # --- Insert 'linear' ---
        if is_pwconv:
            parts = k.split('.')
            parts.insert(-1, 'linear')
            k = '.'.join(parts)

        # --- Reverse GRN reshape ---
        if 'grn' in k and v.dim() == 3:
            v = v.squeeze(0).squeeze(0)

        # --- Leave bias as-is ---
        if k.endswith('bias') and v.dim() == 1:
            if "dwconv" in k or k in ["downsample_layers.1.1.bias", "downsample_layers.2.1.bias", "downsample_layers.3.1.bias"]:
                new_ckpt[k] = v.unsqueeze(0)
            else:
                new_ckpt[k] = v
            continue

        # --- Reshape weight to kernel unless in downsample.0.0 ---
        if k.endswith('weight') and not is_downsample_0_0:
            base_k = k.rsplit('.', 1)[0]  # remove '.weight'
            weight = v

            if weight.dim() == 4:
                out_dim, in_dim, kh, kw = weight.shape
                kv = kh * kw

                if in_dim == 1:
                    # Depthwise conv
                    new_weight = weight.transpose(3, 2).reshape(out_dim, kv).permute(1, 0)
                else:
                    # Standard conv
                    new_weight = weight.transpose(3, 2).reshape(out_dim, in_dim, kv).permute(2, 1, 0)

                new_ckpt[base_k + '.kernel'] = new_weight
                continue

        # --- Default case ---
        new_ckpt[k] = v

    return new_ckpt


def main():
    parser = argparse.ArgumentParser(description="Reverse remap ConvNeXtV2(FCMAE) checkpoint SparseConvNeXtV2.")
    parser.add_argument('--input', type=str, required=True, help='Path to input .pth or .pt checkpoint file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the remapped checkpoint')
    args = parser.parse_args()

    print(f"Loading checkpoint from: {args.input}")
    ckpt = torch.load(args.input, map_location='cpu')

    # Support for checkpoint dicts with nested 'state_dict'

    ckpt_dict = ckpt['model']

    print("Remapping keys...")
    remapped_ckpt = reverse_remap_checkpoint_keys(ckpt_dict)

    ckpt['model'] = remapped_ckpt
    torch.save(ckpt, args.output)

    print(f"Saved remapped checkpoint to: {args.output}")

if __name__ == "__main__":
    main()


