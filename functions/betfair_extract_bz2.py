def betfair_extract_bz2(bz2_file_path, output_folder):
    
    """
    This function takes a path to a bz2 file and extracts it into a new folder
    
    """

    filename = os.path.basename(bz2_file_path).replace(".bz2", "")
    output_file_path = os.path.join(output_folder, filename).replace("\\", "/")
    
    # === Decompress to output directory ===
    with bz2.open(bz2_file_path, 'rb') as f_in:
        with open(output_file_path, 'wb') as f_out:
            f_out.write(f_in.read())
            
    return None
            