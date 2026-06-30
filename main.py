"""
PDF & Image Table Extraction Pipeline
Uses LangChain + OpenAI SDK to extract tables from PDFs and images.

Usage:
    python main.py                                  # process all files in ./input
    python main.py --input ./my_pdfs                # custom input folder
    python main.py --file doc_01.pdf                # single PDF
    python main.py --file screenshot.png            # single image (PNG/JPG/JPEG/BMP/TIFF)
    python main.py --no-validate                    # skip LLM validation
    python main.py --model gpt-4o-mini              # use a cheaper model
"""
import argparse
import os
import sys
import glob
import time

from dotenv import load_dotenv

load_dotenv(override=True)


def main():
    parser = argparse.ArgumentParser(description="Extract tables from PDFs and images")
    parser.add_argument("--input", default="./input", help="Input directory containing files")
    parser.add_argument("--output", default="./output", help="Output directory for results")
    parser.add_argument("--file", help="Process a single file (PDF or image)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument("--no-validate", action="store_true", help="Skip LLM validation step")
    parser.add_argument("--max-retries", type=int, default=2, help="Max validation retries")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Create a .env file or set the environment variable.")
        sys.exit(1)

    from src.pipeline import process_pdf, process_image

    SUPPORTED_IMAGES = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')

    if args.file:
        if os.path.isabs(args.file) or os.path.exists(args.file):
            input_files = [args.file]
        else:
            input_files = [os.path.join(args.input, args.file)]
    else:
        input_files = sorted(glob.glob(os.path.join(args.input, "*.pdf")))
        for ext in SUPPORTED_IMAGES:
            input_files.extend(sorted(glob.glob(os.path.join(args.input, f"*{ext}"))))

    if not input_files:
        print(f"No supported files found in '{args.input}'.")
        print("Place PDF or image files in the input directory and run again.")
        sys.exit(0)

    print(f"Found {len(input_files)} file(s) to process.")
    print(f"Model: {args.model}")
    print(f"Validation: {'OFF' if args.no_validate else 'ON'}")

    start = time.time()
    results = []

    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"WARNING: File not found: {file_path}")
            continue

        ext = os.path.splitext(file_path)[1].lower()
        if ext in SUPPORTED_IMAGES:
            doc = process_image(
                image_path=file_path,
                output_dir=args.output,
                model=args.model,
                validate=not args.no_validate,
                max_retries=args.max_retries,
            )
        else:
            doc = process_pdf(
                pdf_path=file_path,
                output_dir=args.output,
                model=args.model,
                validate=not args.no_validate,
                max_retries=args.max_retries,
            )
        results.append(doc)

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {len(results)}")
    total_tables = sum(len(d.tables) for d in results)
    print(f"Tables extracted: {total_tables}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Output directory: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
