import pandas as pd
from pathlib import Path


def parse_arff_to_df(arff_path: Path) -> pd.DataFrame:
    # Parse the ARFF file into a csv DataFrame

    attribute_names = []
    data_rows = []
    data_started = False

    with open(arff_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("%"):
                continue

            upper = line.upper()

            if upper.startswith("@ATTRIBUTE"):
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[1]
                    attribute_names.append(name)
                continue

            if upper.startswith("@DATA"):
                data_started = True
                continue

            # skip other @ lines
            if line.startswith("@"):
                continue

            if data_started:
                parts = [p for p in line.split(",") if p != ""]
                if len(parts) >= len(attribute_names):
                    data_rows.append(parts[: len(attribute_names)])

    df = pd.DataFrame(data_rows, columns=attribute_names)
    return df


def main():
    base_dir = Path("Scenario A2-ARFF")

    vpn_arff = base_dir / "TimeBasedFeatures-Dataset-30s-VPN.arff"
    novpn_arff = base_dir / "TimeBasedFeatures-Dataset-30s-NO-VPN.arff"

    print("Parsing VPN ARFF:", vpn_arff)
    df_vpn = parse_arff_to_df(vpn_arff)
    print("VPN shape:", df_vpn.shape)

    print("Parsing NO-VPN ARFF:", novpn_arff)
    df_novpn = parse_arff_to_df(novpn_arff)
    print("NO-VPN shape:", df_novpn.shape)

    # add binary label: 1 = VPN, 0 = NO-VPN
    df_vpn["vpn_label"] = 1
    df_novpn["vpn_label"] = 0

    # concatenate
    df_all = pd.concat([df_vpn, df_novpn], ignore_index=True)

    # ensure numeric where possible
    for col in df_all.columns:
        if col not in ["class1"]:  # leave class1 as string label
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    print("Combined shape:", df_all.shape)
    print("Label distribution (class1):")
    print(df_all["class1"].value_counts())
    print("\nVPN binary label distribution:")
    print(df_all["vpn_label"].value_counts())

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "a2_30s_vpn_binary.csv"
    df_all.to_csv(out_csv, index=False)
    print("Saved combined CSV to:", out_csv)


if __name__ == "__main__":
    main()
