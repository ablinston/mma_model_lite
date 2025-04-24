def betfair_filter_bz2_for_fights(input_folder, output_folder):
    
    """
    This function takes a folder containing raw downloaded betfair bz2 files within subfolders and looks in each one to see if it's a fight.
    It will extract the files from the bz2 compression for those files that are fights and output them in the output folder
    
    Returns: DataFrame of the filenames and whether they are fights or not
    
    """
    
    # pdb.set_trace()
    
    # file_list = []
    # for path, subdirs, files in os.walk(input_folder):
    #     for name in files:
    #         file_list.append(path.replace('\\', '/') + '/' + name)
    file_list = [str(p.resolve()) for p in Path(input_folder).rglob("*.bz2")]

    new_files_df = pd.DataFrame({'filepath': file_list, 'fight_flag': np.nan})

    # # Loop through the files to check which are fights and flag them on the data
    # for i in range(0, len(new_files_df)):
        
    #     if i % 100 == 0:
    #         print('Read in ' + str(i) + ' of ' + str(len(new_files_df)))
            
    #     with bz2.open(new_files_df['filepath'][i], "rt", encoding="utf-8") as f:
    #         for line in f:
    #             if "Fight Result" in line:
    #                 new_files_df.loc[i, 'fight_flag'] = 1
    #                 break
            
    #             new_files_df.loc[i, 'fight_flag'] = 0
    
    # Get the flags as to whether the file is a fight or not
    fight_flags = parallel_run_function(
        func = betfair_flag_bz2_as_fight,
        input_list = new_files_df['filepath'].tolist(),
        max_workers = config['num_threads'],
        collapse_output_to_df = False)
            

    # fp.write('./data/betfair_fileflags.parquet', new_files_df)

    fight_files = new_files_df[pd.Series(fight_flags) == 1]

    # for i in range(0, len(fight_files)):
        
    #     if i % 100 == 0:
    #         print('Written ' + str(i) + ' of ' + str(len(fight_files)))
        
    #     bz2_file_path = fight_files['filepath'].iloc[i]
        
    #     # === Create output filename by stripping .bz2 extension ===
    #     filename = os.path.basename(bz2_file_path).replace(".bz2", "")
    #     output_file_path = os.path.join(output_folder, filename).replace("\\", "/")
        
    #     # === Decompress to output directory ===
    #     with bz2.open(bz2_file_path, 'rb') as f_in:
    #         with open(output_file_path, 'wb') as f_out:
    #             f_out.write(f_in.read())
    
    # For each file, extract the bz2 into an output folder
    parallel_run_function(
        func = betfair_extract_bz2,
        input_list = fight_files['filepath'].str.replace("\\", "/").tolist(),
        max_workers = config['num_threads'],
        collapse_output_to_df = False,
        output_folder = output_folder)
                
    return new_files_df
