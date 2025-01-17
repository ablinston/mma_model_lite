# Load all the packages and functions needed
exec(open('global.py').read())

# Load what answers are expected
parse_fight_details_url_data_correct = fp.ParquetFile('./tests/data/parse_fight_details_url_data_correct.parquet').to_pandas()

# This test looks at a specific URL for an historic event and checks that the results match what is expected
def test_parse_fight_details_url():
    
    result = parse_fight_details_url('http://ufcstats.com/event-details/0c01568b1ac77bff')
    
    pd.testing.assert_frame_equal(result, parse_fight_details_url_data_correct)
    
    assert result.equals(parse_fight_details_url_data_correct) == True
    


# fp.write('./tests/data/parse_fight_details_url_data_correct.parquet', result, compression = 'ZSTD')
