from pathlib import Path

def main():
    sections = [f"4.{i}.md" for i in range(1, 13)]
    dest = Path("4.md")
    with dest.open("w", encoding="utf-8") as out:
        for section in sections:
            path = Path(section)
            if not path.exists():
                raise FileNotFoundError(f"{section} missing")
            out.write(f"<!-- start {section} -->\n")
            out.write(path.read_text(encoding="utf-8"))
            out.write(f"\n<!-- end {section} -->\n\n")
    print(f"Wrote combined file to {dest}")


if __name__ == "__main__":
    main()
