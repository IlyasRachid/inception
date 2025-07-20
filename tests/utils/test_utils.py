import os
import tempfile
from inception.utils.utils import generate_pretty_tree

def test_generate_pretty_tree_creates_file():
    # Use a temporary directory to avoid cluttering your project root
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Run the function with a custom output filename inside temp dir
        output_file = os.path.join(tmpdirname, "tree.txt")
        generate_pretty_tree(output_filename=output_file)
        
        # Check the file was created
        assert os.path.isfile(output_file)
        
        # Optional: check the file is not empty
        assert os.path.getsize(output_file) > 0