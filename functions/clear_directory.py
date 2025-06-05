def clear_directory(path):
    # Use scandir for faster performance with many small files
    with os.scandir(path) as entries:
        for entry in entries:
            try:
                if entry.is_file() or entry.is_symlink():
                    os.unlink(entry.path)
                elif entry.is_dir():
                    shutil.rmtree(entry.path)
            except Exception as e:
                print(f"Failed to delete {entry.path}: {e}")
