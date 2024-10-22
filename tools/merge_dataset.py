import torch
import argparse

def merge_checkpoints(input_paths, output_path):
    merged_ckpt = {}

    fields_to_merge = ["agent_hist", "neigh_hist", "map_polyline", "target_pos"]

    for input_path in input_paths:
        ckpt = torch.load(input_path)

        for field in fields_to_merge:
            if field in ckpt:
                if field in merged_ckpt:
                    merged_ckpt[field] = torch.cat([merged_ckpt[field], ckpt[field]], dim=0)
                else:
                    merged_ckpt[field] = ckpt[field]

    torch.save(merged_ckpt, output_path)
    print(f"Checkpoint saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge multiple checkpoint .pth files.')
    parser.add_argument('--input_paths', nargs='+', required=True, help='List of input .pth file paths to merge.')
    parser.add_argument('--output_path', required=True, help='Output path for the merged .pth file.')

    args = parser.parse_args()

    merge_checkpoints(args.input_paths, args.output_path)