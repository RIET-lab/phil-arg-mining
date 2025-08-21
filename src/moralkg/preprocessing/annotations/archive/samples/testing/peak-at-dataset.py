import argparse
from datasets import load_dataset, Dataset, IterableDataset

def main():
    parser = argparse.ArgumentParser(
        description="""
        Peek at the first few rows of a Hugging Face dataset.
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the dataset on Hugging Face Hub (e.g., 'imdb' or 'org/dataset').",
    )
    parser.add_argument(
        "-n", "--num_rows", type=int, default=5, help="Number of rows to display."
    )
    parser.add_argument(
        "-s", "--split",
        type=str,
        default="train",
        help="Dataset split to use (e.g., 'train', 'test', 'validation').",
    )
    parser.add_argument(
        "-k", "--token",
        type=str,
        default=None,
        help="Hugging Face Hub token for private datasets.",
    )

    args = parser.parse_args()

    try:
        print(f"Loading dataset '{args.dataset_name}' with split '{args.split}'...")
        dataset = load_dataset(args.dataset_name, split=args.split, token=args.token)

        print(f"Showing the first {args.num_rows} rows:\n")

        if isinstance(dataset, Dataset):
            num_to_show = min(args.num_rows, len(dataset)) 
            for i in range(num_to_show):
                print(dataset[i]) 
                print("-" * 20)
        elif isinstance(dataset, IterableDataset):
            for idx, row in enumerate(dataset):
                if idx >= args.num_rows:
                    break
                print(row)
                print("-" * 20)
        else:
            print("Unsupported dataset type.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
