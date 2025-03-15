#!/usr/bin/env python3
"""
File Splitter

A utility for splitting large files into smaller chunks.
Designed to help prepare content for AI systems with context size limitations.

Usage:
    python file_splitter.py -i input_file -o output_directory [options]
    python file_splitter.py --input input_file --output output_directory --size 500

Examples:
    python file_splitter.py -i large_document.txt -o ./chunks
    python file_splitter.py -i dataset.json -o ./split_files --size 10000 --unit KB
    python file_splitter.py -i code_base.py -o ./code_chunks --size 0.5 --unit MB --numbered
"""

import os
import sys
import argparse
import math
from pathlib import Path
import shutil


def get_size_in_bytes(size: float, unit: str) -> int:
    """
    Convert size with unit to bytes.
    
    Args:
        size: The numeric size value
        unit: The unit (B, KB, MB, GB)
        
    Returns:
        Size in bytes
    """
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024
    }
    
    if unit not in units:
        raise ValueError(f"Invalid unit: {unit}. Valid units are: {', '.join(units.keys())}")
    
    return int(size * units[unit])


def get_human_readable_size(size_bytes: int) -> str:
    """
    Convert bytes to human-readable size string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def estimate_tokens(bytes_size: int, bytes_per_token: int = 4) -> int:
    """
    Estimate the number of tokens based on bytes.
    
    Args:
        bytes_size: Size in bytes
        bytes_per_token: Estimated bytes per token (default: 4)
        
    Returns:
        Estimated number of tokens
    """
    return math.ceil(bytes_size / bytes_per_token)


def split_file_by_size(
    input_file: str, 
    output_dir: str, 
    chunk_size_bytes: int, 
    use_numbered_filenames: bool = False,
    preserve_extension: bool = True
) -> list:
    """
    Split a file into chunks of a specified size.
    
    Args:
        input_file: Path to the input file
        output_dir: Directory to save chunks
        chunk_size_bytes: Size of each chunk in bytes
        use_numbered_filenames: Use numbered filenames (part_001, part_002, etc.)
        preserve_extension: Keep the original file extension
        
    Returns:
        List of created output files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    input_path = Path(input_file)
    base_name = input_path.stem
    extension = input_path.suffix if preserve_extension else ""
    
    # Determine total file size
    total_size = os.path.getsize(input_file)
    total_chunks = math.ceil(total_size / chunk_size_bytes)
    digit_count = len(str(total_chunks))
    
    output_files = []
    
    with open(input_file, 'rb') as f:
        chunk_num = 1
        
        while True:
            # Read a chunk of the specified size
            chunk_data = f.read(chunk_size_bytes)
            
            # If no data, we're done
            if not chunk_data:
                break
            
            # Create output filename
            if use_numbered_filenames:
                # Format: base_001.ext, base_002.ext, etc.
                out_filename = f"{base_name}_{chunk_num:0{digit_count}d}{extension}"
            else:
                # Format: base_part1.ext, base_part2.ext, etc.
                out_filename = f"{base_name}_part{chunk_num}{extension}"
            
            out_path = os.path.join(output_dir, out_filename)
            
            # Write the chunk to the output file
            with open(out_path, 'wb') as out_f:
                out_f.write(chunk_data)
            
            output_files.append(out_path)
            print(f"Created chunk {chunk_num}/{total_chunks}: {out_filename} - {get_human_readable_size(len(chunk_data))}")
            
            chunk_num += 1
    
    return output_files


def split_file_by_lines(
    input_file: str, 
    output_dir: str, 
    lines_per_chunk: int,
    use_numbered_filenames: bool = False,
    preserve_extension: bool = True
) -> list:
    """
    Split a text file into chunks with a specified number of lines.
    
    Args:
        input_file: Path to the input file
        output_dir: Directory to save chunks
        lines_per_chunk: Number of lines in each chunk
        use_numbered_filenames: Use numbered filenames (part_001, part_002, etc.)
        preserve_extension: Keep the original file extension
        
    Returns:
        List of created output files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    input_path = Path(input_file)
    base_name = input_path.stem
    extension = input_path.suffix if preserve_extension else ""
    
    # Count total lines
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        total_lines = sum(1 for _ in f)
    
    total_chunks = math.ceil(total_lines / lines_per_chunk)
    digit_count = len(str(total_chunks))
    
    output_files = []
    
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        chunk_num = 1
        line_count = 0
        chunk_lines = []
        
        for line in f:
            chunk_lines.append(line)
            line_count += 1
            
            # If we've reached the desired number of lines, write the chunk
            if line_count >= lines_per_chunk:
                # Create output filename
                if use_numbered_filenames:
                    out_filename = f"{base_name}_{chunk_num:0{digit_count}d}{extension}"
                else:
                    out_filename = f"{base_name}_part{chunk_num}{extension}"
                
                out_path = os.path.join(output_dir, out_filename)
                
                # Write the chunk to the output file
                with open(out_path, 'w', encoding='utf-8') as out_f:
                    out_f.writelines(chunk_lines)
                
                chunk_size = os.path.getsize(out_path)
                output_files.append(out_path)
                print(f"Created chunk {chunk_num}/{total_chunks}: {out_filename} - {get_human_readable_size(chunk_size)}")
                
                # Reset for next chunk
                chunk_lines = []
                line_count = 0
                chunk_num += 1
        
        # Write any remaining lines
        if chunk_lines:
            if use_numbered_filenames:
                out_filename = f"{base_name}_{chunk_num:0{digit_count}d}{extension}"
            else:
                out_filename = f"{base_name}_part{chunk_num}{extension}"
            
            out_path = os.path.join(output_dir, out_filename)
            
            with open(out_path, 'w', encoding='utf-8') as out_f:
                out_f.writelines(chunk_lines)
            
            chunk_size = os.path.getsize(out_path)
            output_files.append(out_path)
            print(f"Created chunk {chunk_num}/{total_chunks}: {out_filename} - {get_human_readable_size(chunk_size)}")
    
    return output_files


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Split large files into smaller chunks for AI processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input file to split')
    parser.add_argument('-o', '--output', required=True, help='Output directory for chunks')
    parser.add_argument('--size', type=float, default=500, 
                        help='Size of each chunk (default: 500 KB)')
    parser.add_argument('--unit', choices=['B', 'KB', 'MB', 'GB'], default='KB',
                        help='Unit for chunk size (default: KB)')
    parser.add_argument('--by-lines', action='store_true',
                        help='Split by number of lines instead of size')
    parser.add_argument('--lines', type=int, default=1000,
                        help='Number of lines per chunk when using --by-lines (default: 1000)')
    parser.add_argument('--numbered', action='store_true',
                        help='Use numbered filenames (e.g., file_001.txt instead of file_part1.txt)')
    parser.add_argument('--preserve-extension', action='store_true', default=True,
                        help='Preserve the original file extension (default: True)')
    parser.add_argument('--token-estimate', action='store_true',
                        help='Show token estimates for each chunk (rough approximation)')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1
    
    # Get file info
    file_size = os.path.getsize(args.input)
    print(f"Input file: {args.input}")
    print(f"File size: {get_human_readable_size(file_size)}")
    
    if args.by_lines:
        print(f"Splitting method: By lines ({args.lines} lines per chunk)")
        output_files = split_file_by_lines(
            args.input, 
            args.output, 
            args.lines,
            args.numbered,
            args.preserve_extension
        )
    else:
        # Calculate chunk size in bytes
        chunk_size_bytes = get_size_in_bytes(args.size, args.unit)
        print(f"Splitting method: By size ({get_human_readable_size(chunk_size_bytes)} per chunk)")
        output_files = split_file_by_size(
            args.input, 
            args.output, 
            chunk_size_bytes,
            args.numbered,
            args.preserve_extension
        )
    
    # Print summary
    print("\nSummary:")
    print(f"Total chunks created: {len(output_files)}")
    print(f"Output directory: {os.path.abspath(args.output)}")
    
    # Token estimates
    if args.token_estimate:
        print("\nToken Estimates (approximate):")
        for file_path in output_files:
            size = os.path.getsize(file_path)
            tokens = estimate_tokens(size)
            print(f"{os.path.basename(file_path)}: ~{tokens:,} tokens")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 