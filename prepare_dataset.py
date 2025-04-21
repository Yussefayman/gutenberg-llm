# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
"""
Script that processes the Project Gutenberg files into fewer larger files.
"""
import argparse
import os
import re
from tqdm import tqdm
from langdetect import detect
from langdetect import LangDetectException

def remove_gutenberg_boilerplate(text):
    """
    Remove Project Gutenberg boilerplate by simply removing the first 15 lines
    """
    lines = text.split('\n')
    if len(lines) <= 15:
        return text
    return '\n'.join(lines[40:])

def check_english_files(folder_path):
    """
    Reads all text files in a folder and checks if they're in English.
    
    Args:
        folder_path (str): Path to the folder containing text files
        
    Returns:
        dict: Dictionary with filenames as keys and language detection results as values
    """
    results = {}
    
    # Check if the directory exists
    if not os.path.isdir(folder_path):
        print(f"Error: Directory '{folder_path}' does not exist.")
        return results
    
    # Get all files in the directory and subdirectories
    all_files = []
    for path, subdirs, files in os.walk(folder_path):
        for name in files:
            if name.endswith((".txt", ".txt.utf8")):
                all_files.append(os.path.join(path, name))
    
    # Process all the files
    for file_path in tqdm(all_files, desc="Checking language"):
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                # Read the content (first 1000 characters should be enough for language detection)
                content = file.read(1000).strip()
                
                # Skip empty files
                if not content:
                    results[filename] = {"is_english": False, "language": "unknown", "error": "Empty file"}
                    continue
                
                # Detect language
                detected_lang = detect(content)
                is_english = detected_lang == 'en'
                
                results[filename] = {
                    "is_english": is_english,
                    "language": detected_lang,
                    "error": None
                }
        
        except UnicodeDecodeError:
            # Try with fallback encoding
            try:
                with open(file_path, 'r', encoding='latin1') as file:
                    content = file.read(1000).strip()
                    if not content:
                        results[filename] = {"is_english": False, "language": "unknown", "error": "Empty file"}
                        continue
                    
                    detected_lang = detect(content)
                    is_english = detected_lang == 'en'
                    
                    results[filename] = {
                        "is_english": is_english,
                        "language": detected_lang,
                        "error": None
                    }
            except Exception as e:
                results[filename] = {"is_english": False, "language": "unknown", "error": f"Error with fallback encoding: {str(e)}"}
        except LangDetectException as e:
            results[filename] = {"is_english": False, "language": "unknown", "error": f"Language detection error: {str(e)}"}
        except Exception as e:
            results[filename] = {"is_english": False, "language": "unknown", "error": f"Error processing file: {str(e)}"}
    
    return results

def print_summary(results):
    """Print a summary of the language detection results"""
    total = len(results)
    english_count = sum(1 for file_result in results.values() if file_result["is_english"])
    non_english_count = total - english_count
    
    print(f"\nLanguage Detection Summary:")
    print(f"Total files processed: {total}")
    print(f"English files: {english_count}")
    print(f"Non-English files: {non_english_count}")
    
    # Count by language
    language_counts = {}
    for result in results.values():
        lang = result["language"]
        if lang not in language_counts:
            language_counts[lang] = 0
        language_counts[lang] += 1
    
    print("\nLanguage distribution:")
    for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"- {lang}: {count} files")
    
    if non_english_count > 0:
        print("\nSample of non-English files:")
        count = 0
        for filename, result in results.items():
            if not result["is_english"]:
                lang = result["language"]
                print(f"- {filename}: {lang}")
                count += 1
                if count >= 10:  # Show just 10 examples
                    break

def is_english_using_langdetect(text):
    """Use langdetect to check if text is in English"""
    try:
        # Use just the first 1000 characters for faster detection
        sample = text[:1000]
        return detect(sample) == 'en'
    except LangDetectException:
        return False

def combine_files(file_paths, target_dir, max_size_mb=500, separator="<|endoftext|>", fallback_encoding="latin1"):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    current_content = []
    current_size = 0
    file_counter = 1
    
    language_stats = {"en": 0, "other": 0}
    
    for file_path in tqdm(file_paths, desc="Processing files"):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:
            # Attempt to read the file with a fallback encoding
            tqdm.write(f"Warning: UnicodeDecodeError encountered. Trying fallback encoding for {file_path}")
            with open(file_path, "r", encoding=fallback_encoding) as file:
                content = file.read()
        
        if not is_english_using_langdetect(content):
            tqdm.write(f"Skipping {file_path} as it does not contain primarily English text.")
            language_stats["other"] += 1
            continue
            
        language_stats["en"] += 1
        
        # Remove first 15 lines
        content = remove_gutenberg_boilerplate(content)
        
        # Regular expression to replace multiple blank lines with a single blank line
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # Skip if content is too short after cleaning
        if len(content.strip()) < 100:
            tqdm.write(f"Skipping {file_path} as content is too short after cleaning.")
            continue
        
        estimated_size = len(content.encode("utf-8"))
        if current_size + estimated_size > max_size_mb * 1024 * 1024:
            target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
            with open(target_file_path, "w", encoding="utf-8") as target_file:
                target_file.write(separator.join(current_content))
            file_counter += 1
            current_content = [content]
            current_size = estimated_size
        else:
            current_content.append(content)
            current_size += estimated_size
            
    if current_content:
        target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
        with open(target_file_path, "w", encoding="utf-8") as target_file:
            target_file.write(separator.join(current_content))
    
    print("\nLanguage statistics during processing:")
    print(f"English files processed: {language_stats['en']}")
    print(f"Non-English files skipped: {language_stats['other']}")
            
    return file_counter

def clean_and_verify_gutenberg(file_path, fallback_encoding="latin1"):
    """
    Clean a single Gutenberg file and check the results
    """
    # Read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding=fallback_encoding) as f:
            original_content = f.read()
    
    # Clean the content
    cleaned_content = remove_gutenberg_boilerplate(original_content)
    
    # Display stats
    original_lines = original_content.split('\n')
    cleaned_lines = cleaned_content.split('\n')
    
    print(f"\nFile: {os.path.basename(file_path)}")
    print(f"Original length: {len(original_content)} characters, {len(original_lines)} lines")
    print(f"Cleaned length: {len(cleaned_content)} characters, {len(cleaned_lines)} lines")
    print(f"Removed {len(original_content) - len(cleaned_content)} characters ({(len(original_content) - len(cleaned_content)) / len(original_content) * 100:.2f}%)")
    
    # Display removed lines
    print("\nRemoved lines:")
    for i, line in enumerate(original_lines[:15]):
        print(f"{i+1:02d}. {line[:80]}{'...' if len(line) > 80 else ''}")
    
    # Display first few lines of cleaned content
    print("\nFirst 5 lines of cleaned content:")
    for i, line in enumerate(cleaned_lines[:5]):
        if line.strip():
            print(f"> {line[:80]}{'...' if len(line) > 80 else ''}")
    
    return original_content, cleaned_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and combine text files for pretraining")
    parser.add_argument("--data_dir", type=str, default="gutenberg/data/raw",
                        help="Directory containing the downloaded raw training data")
    parser.add_argument("--max_size_mb", type=int, default=500,
                        help="The maximum file size for each concatenated file in megabytes")
    parser.add_argument("--output_dir", type=str, default="gutenberg_preprocessed",
                        help="Directory where the preprocessed data will be saved")
    parser.add_argument("--check_languages", action="store_true",
                        help="Run language detection on files before processing")
    parser.add_argument("--test_cleanup", type=str, default=None,
                        help="Test cleanup on a specific file and show detailed comparison")
    args = parser.parse_args()

    # Test cleanup on a specific file if requested
    if args.test_cleanup:
        original, cleaned = clean_and_verify_gutenberg(args.test_cleanup)
        exit()

    all_files = [os.path.join(path, name) for path, subdirs, files in os.walk(args.data_dir)
                 for name in files if name.endswith((".txt", ".txt.utf8"))]
    
    print(f"{len(all_files)} file(s) found.")
    
    if args.check_languages:
        print(f"Checking language of all text files in '{args.data_dir}'...")
        results = check_english_files(args.data_dir)
        print_summary(results)
        
        user_input = input("\nContinue with processing and combining files? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Processing cancelled.")
            exit()
    
    file_counter = combine_files(all_files, args.output_dir, max_size_mb=args.max_size_mb)
    print(f"{file_counter} file(s) saved in {os.path.abspath(args.output_dir)}")
    
    # Print final language statistics summary
    print("\nProcessing complete!")