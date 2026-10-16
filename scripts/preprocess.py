"""CLI entry point: inspect first, review manifest, then split."""
import argparse
import json
from scripts.data import inspect, split_manifest


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('inspect'); a.add_argument('root'); a.add_argument('output')
    a=sub.add_parser('split'); a.add_argument('manifest'); a.add_argument('output'); a.add_argument('--seed',type=int,default=42); a.add_argument('--per-class',type=int,default=30)
    args=p.parse_args()
    if args.command=='inspect': print(json.dumps(inspect(args.root,args.output),indent=2))
    else: print('Selected clips:',len(split_manifest(args.manifest,args.output,args.seed,args.per_class)))


if __name__=='__main__': main()
