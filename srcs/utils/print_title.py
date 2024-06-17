def print_title(title: str):
    """
    Print a title
    """
    BLUE = "\033[94m"
    END = "\033[0m"
    print("\n" + "=" * 80)
    print(BLUE + title + END)
    print("=" * 80 + "\n")
