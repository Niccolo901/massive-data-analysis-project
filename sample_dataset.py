import pandas as pd

# Load dataset
df = pd.read_csv('dataset/books_data.csv')

# Identify the top 3 categories
top_categories = df['categories'].value_counts().nlargest(3).index.tolist()

# Sample 333 rows from each of the top categories
books_subdata = pd.concat([
    df[df['categories'] == category].sample(n=333, random_state=42)
    for category in top_categories
], ignore_index=True)

# Display basic info about the new dataset
print(books_subdata['categories'].value_counts())
print(f"\nTotal rows in books_subdata: {len(books_subdata)}")

#Optionally, save to CSV
books_subdata.to_csv('dataset/books_subdata.csv', index=False)
