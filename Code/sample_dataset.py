import pandas as pd

def create_books_subdata(input_csv_path='dataset/books_data.csv',
                         output_csv_path='dataset/books_subdata.csv',
                         n_samples_per_category=333,
                         top_k=3,
                         random_state=42,
                         verbose=True):
    """
    Creates a subset of the book dataset by sampling N rows from the top K categories.

    Parameters:
        input_csv_path (str): Path to the original dataset CSV.
        output_csv_path (str): Path where the subset CSV will be saved.
        n_samples_per_category (int): Number of samples to draw from each top category.
        top_k (int): Number of top categories to include.
        random_state (int): Seed for reproducible sampling.
        verbose (bool): If True, prints summary info.

    Returns:
        pd.DataFrame: The sampled subset DataFrame.
    """
    df = pd.read_csv(input_csv_path)
    top_categories = df['categories'].value_counts().nlargest(top_k).index.tolist()

    books_subdata = pd.concat([
        df[df['categories'] == category].sample(n=n_samples_per_category, random_state=random_state)
        for category in top_categories
    ], ignore_index=True)

    books_subdata.to_csv(output_csv_path, index=False)

    if verbose:
        print("Top categories sampled:")
        print(books_subdata['categories'].value_counts())
        print(f"\nTotal rows in books_subdata: {len(books_subdata)}")
        print(f"Saved to: {output_csv_path}")

    return books_subdata
