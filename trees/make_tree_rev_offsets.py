from tree_reader import map_tree_rev_steps_bytes

import argparse

parser = argparse.ArgumentParser(
    description="Generate reverse offsets for tree mapping"
)
parser.add_argument("fname", type=str, help="Input file name")
parser.add_argument("out_path", type=str, help="Output file path")
parser.add_argument("--star", action="store_true", help="Enable star mode")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")

args = parser.parse_args()


map_tree_rev_steps_bytes(args.fname, args.out_path, star=args.star, debug=args.debug)
