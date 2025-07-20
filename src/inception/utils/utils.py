import os

def generate_pretty_tree(output_filename="project_tree.txt"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

    def _tree(dir_path, prefix=""):
        entries = sorted(os.listdir(dir_path))
        entries = [e for e in entries if not e.startswith('.')]  # optional: skip hidden files

        tree_lines = []
        total = len(entries)
        for i, entry in enumerate(entries):
            path = os.path.join(dir_path, entry)
            connector = "└─" if i == total - 1 else "├─"

            if os.path.isdir(path):
                tree_lines.append(f"{prefix}{connector} {entry}")
                extension = "   " if i == total - 1 else "│  "
                tree_lines.extend(_tree(path, prefix + extension))
            else:
                tree_lines.append(f"{prefix}{connector} {entry}")

        return tree_lines

    # Root line (no prefix)
    tree_output = [os.path.basename(project_root)]
    tree_output.extend(_tree(project_root))

    output_path = os.path.join(project_root, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tree_output))

    print(f"Pretty project tree saved to {output_path}")


if __name__ == "__main__":
    generate_pretty_tree()
