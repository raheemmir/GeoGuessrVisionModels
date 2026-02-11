import pycountry

INPUT_FILE = "v2/assets/geoguessr_countries_raw.txt"
OUTPUT_FILE = "v2/assets/geoguessr_countries_iso2.txt"

def main():
    with open(INPUT_FILE, "r") as f:
        names = [line.strip() for line in f if line.strip()]

    iso_2_codes = []

    for name in names:
        try:
            iso_2 = pycountry.countries.lookup(name).alpha_2
            iso_2_codes.append(iso_2)
            print(f"{name} -> {iso_2}")
        except LookupError:
            print(f"Fail: {name}")

    print(f"# of Countries: {len(names)}")

    with open(OUTPUT_FILE, "w") as f:
        for code in iso_2_codes:
            f.write(code + "\n")
    
    print("Successfully mapped country names to ISO 3166 (alpha-2) codes")
        

if __name__ == "__main__":
    main()