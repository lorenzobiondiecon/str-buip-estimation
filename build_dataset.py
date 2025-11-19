from src.data_builder import DataBuilder
import logging

# Setup simple logging to console
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    builder = DataBuilder()
    try:
        df = builder.run()
        print("\nSUCCESS: Data built successfully.")
        print(f"Shape: {df.shape}")
        print(f"Countries included: {df['country'].unique()}")
        print(df.head())
    except Exception as e:
        print(f"\nFAILURE: Data build process failed.\nError: {e}")

if __name__ == "__main__":
    main()
