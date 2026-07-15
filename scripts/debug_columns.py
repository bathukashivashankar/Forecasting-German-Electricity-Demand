from src.electricity_demand.data import load_opsd_raw

def main():
    df = load_opsd_raw()
    de_cols = [c for c in df.columns if "DE_load" in c]
    print("German load-related columns:")
    print(de_cols)

if __name__ == "__main__":
    main()