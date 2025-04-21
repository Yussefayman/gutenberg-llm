#!/usr/bin/env python3
# pretokenize_simple.py
"""
Simple script to pre-tokenize text files and save them as PyTorch tensor files.
"""

import argparse
import os
import tiktoken
import torch
from pathlib import Path
from tqdm import tqdm

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pre-tokenize text files for LLM training')
    parser.add_argument('--data_dir', type=str, default='gutenberg/data',
                        help='Directory containing the text files to tokenize')
    parser.add_argument('--output_dir', type=str, default='tokenized_data',
                        help='Directory where tokenized files will be saved')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        help='Tokenizer to use (default: gpt2)')
    args = parser.parse_args()
    
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding(args.tokenizer)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all text files
    all_files = [os.path.join(path, name) for path, subdirs, files
                in os.walk(args.data_dir) for name in files if name.endswith((".txt"))]
    total_files = len(all_files)
    
    print(f"Found {total_files} text files to tokenize")
    
    total_tokens = 0
    
    # Process each file
    for i, file_path in enumerate(tqdm(all_files), 1):
        # Read the text file
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read() + " <|endoftext|> "  # Add end token as in the training script
        
        # Tokenize
        tokens = tokenizer.encode(text,allowed_special={"<|endoftext|>"})
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        
        # Create output filename
        base_name = os.path.basename(file_path)
        output_name = os.path.splitext(base_name)[0] + ".pt"
        output_path = output_dir / output_name
        
        # Save tokenized file
        torch.save(tokens_tensor, output_path)
        
        total_tokens += len(tokens)
        
        # Print progress
        if i % 10 == 0 or i == total_files:
            print(f"Processed {i}/{total_files} files. Total tokens: {total_tokens:,}")
    
    print(f"Tokenization complete! Processed {total_files} files with {total_tokens:,} tokens total.")
    print(f"Tokenized data saved to {output_dir}")

if __name__ == "__main__":
    main()