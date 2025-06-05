def parallel_run_function(
        func,
        input_list: list,
        max_workers: int,
        collapse_output_to_df: bool = False,
        **kwargs):

    """
    This function takes a function and applies it to a list with parallel runs
    
    """

    # Set up a list to contain all the results
    output_list = []
    completed_count = 0
    
    # Run the function in parallel
    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        future_to_url = {executor.submit(func, item, **kwargs): item for item in input_list}

        # Collect results as they complete
        for future in as_completed(future_to_url):
            try:
                result = future.result()
                # Add the result to the list
                output_list.append(result)
                
            except Exception as e:
                print(f"Error for item {future}: {e}")

            # Increment the completed count
            completed_count += 1
            if completed_count % 100 == 0:
                print('Completed ' + str(completed_count) + ' of ' + str(len(input_list)))
                
    # This ensures that all threads are done
    executor.shutdown(wait=True)
    
    if collapse_output_to_df:
        return pd.concat(output_list, axis = 0)
    else:
        return output_list
