#!/usr/bin/env python3
"""
File Concatenator

A flexible utility for concatenating multiple files into a single output file.
Supports adding headers to identify file sources, sorting input files,
and handles different file encodings.

Usage:
    python file_concatenator.py output_file file1 file2 file3 ...
    python file_concatenator.py -o output_file -i file1 file2 file3 ...
    python file_concatenator.py -o output_file -d directory -p "*.txt"

Examples:
    python file_concatenator.py combined.txt file1.txt file2.txt
    python file_concatenator.py -o combined.txt -i file1.txt file2.txt --add-headers
    python file_concatenator.py -o requirements.txt -d ./requirements -p "*.txt" --sort
"""

import os
import sys
import glob
import argparse
import fnmatch
from datetime import datetime


def get_file_creation_time(file_path):
    """Get the creation time of a file."""
    try:
        return os.path.getctime(file_path)
    except OSError:
        # If we can't get creation time, use modification time
        return os.path.getmtime(file_path)


def sort_files(files, sort_by='name'):
    """Sort files by name or creation date."""
    if sort_by == 'name':
        return sorted(files)
    elif sort_by == 'date':
        return sorted(files, key=get_file_creation_time)
    return files


def read_file_content(file_path, encoding='utf-8'):
    """Read content from a file with error handling."""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, try with latin-1 which can read any byte sequence
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()
    except IOError as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return None


def create_file_header(file_path):
    """Create a header for a file to help identify its source in the concatenated output."""
    filename = os.path.basename(file_path)
    separator = "=" * 80
    return f"\n{separator}\n# FILE: {filename}\n{separator}\n\n"


def concatenate_files(output_file, input_files, add_headers=False, encoding='utf-8'):
    """Concatenate multiple files into a single file."""
    success_count = 0
    failed_files = []
    
    with open(output_file, 'w', encoding=encoding) as outfile:
        for i, file_path in enumerate(input_files):
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}", file=sys.stderr)
                failed_files.append(file_path)
                continue
                
            content = read_file_content(file_path, encoding)
            if content is None:
                failed_files.append(file_path)
                continue
                
            # Add a header if requested (and not the first file or if we want headers for all files)
            if add_headers:
                outfile.write(create_file_header(file_path))
            
            outfile.write(content)
            
            # Add a newline between files if headers are not added
            if not add_headers and i < len(input_files) - 1:
                outfile.write("\n")
                
            success_count += 1
    
    return success_count, failed_files


def find_files_in_directory(directory, pattern):
    """Find files in a directory matching a pattern."""
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory", file=sys.stderr)
        return []
        
    matched_files = []
    
    # Walk through the directory
    for root, _, files in os.walk(directory):
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                matched_files.append(os.path.join(root, filename))
    
    return matched_files


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Concatenate multiple files into a single file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Add arguments
    parser.add_argument('-o', '--output', dest='output_file', required=False,
                        help='Output file path')
    parser.add_argument('-i', '--input', dest='input_files', nargs='+',
                        help='Input files to concatenate')
    parser.add_argument('-d', '--directory', dest='directory',
                        help='Directory to search for files')
    parser.add_argument('-p', '--pattern', dest='pattern', default="*",
                        help='File pattern to match (default: *)')
    parser.add_argument('--add-headers', dest='add_headers', action='store_true',
                        help='Add headers to identify file sources')
    parser.add_argument('--sort', dest='sort', choices=['name', 'date', 'none'],
                        default='none', help='Sort files by name or creation date')
    parser.add_argument('--encoding', dest='encoding', default='utf-8',
                        help='File encoding (default: utf-8)')
    
    # For backward compatibility with the simplest usage
    parser.add_argument('files', nargs='*', help='Files to concatenate (if no named arguments)')
    
    args = parser.parse_args()
    
    # Handle the case where files are specified directly
    if args.files and len(args.files) >= 2 and not args.output_file and not args.input_files:
        args.output_file = args.files[0]
        args.input_files = args.files[1:]
    
    # Ensure we have an output file
    if not args.output_file:
        parser.error("No output file specified. Use -o/--output to specify an output file.")
    
    # Ensure we have input files from one of the methods
    if not args.input_files and not args.directory:
        parser.error("No input files specified. Use -i/--input, -d/--directory or list files directly.")
    
    return args


def main():
    """Main function."""
    args = parse_arguments()
    
    input_files = []
    
    # Get input files from directory if specified
    if args.directory:
        input_files = find_files_in_directory(args.directory, args.pattern)
        if not input_files:
            print(f"No files found in {args.directory} matching pattern {args.pattern}", file=sys.stderr)
            return 1
    else:
        input_files = args.input_files
    
    # Sort files if requested
    if args.sort != 'none':
        input_files = sort_files(input_files, args.sort)
    
    # Perform concatenation
    success_count, failed_files = concatenate_files(
        args.output_file, input_files, args.add_headers, args.encoding
    )
    
    # Report results
    print(f"Successfully concatenated {success_count} files into {args.output_file}")
    
    if failed_files:
        print(f"Failed to process {len(failed_files)} files:", file=sys.stderr)
        for file_path in failed_files:
            print(f"  - {file_path}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 