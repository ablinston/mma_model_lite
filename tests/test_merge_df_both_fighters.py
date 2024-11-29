import unittest
import pandas as pd


class TestMergeBothFighters(unittest.TestCase):
    def test_merge_df_both_fighters:
        
        # Define test data
        df_base = pd.DataFrame({
            'event': [1, 1, 2, 2], 
            'bout': ['a', 'b', 'c', 'd'], 
            'fighter1': ['w', 'x', 'y', 'z'], 
            'fighter2': ['r', 's', 't', 'u']})
    
        df_fighter = pd.DataFrame({
            'event': pd.concat([df_base['event']] * 2), 
            'bout': pd.concat([df_base['bout']] * 2),
            'fighter' : pd.concat([df_base['fighter1'], df_base['fighter2']]),
            'stat': [i * 0.5 for i in range(8)]
        })
    
        result = merge_df_both_fighters(df_base, [df_fighter])
        # Correct answer
        correct = df_base
        correct['fighter1_stat'] = [i * 0.5 for i in range(4)]
        correct['fighter2_stat'] = [2+ i * 0.5 for i in range(4)]
    
        self.assertEqual(result, correct)

if __name__ == '__main__':
    unittest.main()