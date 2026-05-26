import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    csv_path = 'newd/db/train.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Load data
    df = pd.read_csv(csv_path)
    
    # 85% train, 15% validation
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    
    # Save splits
    train_df.to_csv('newd/db/train_split.csv', index=False)
    val_df.to_csv('newd/db/val_split.csv', index=False)
    
    print(f"Split complete. Train: {len(train_df)}, Val: {len(val_df)}")

if __name__ == "__main__":
    main()
