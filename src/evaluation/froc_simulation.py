import pandas as pd

def simulate_false_positive_reduction(candidates_df,true_nodule_threshold_intensity=-560):
    total_candidates = len(candidates_df)
    filtered_items=candidates_df[candidates_df['mean_intensity']> true_nodule_threshold_intensity]
    survived_count =len(filtered_items)
    removed=total_candidates-survived_count
    print(f"total candidates:{total_candidates}")
    print(f"survived filters:{survived_count}")
    print(f"removed:{removed}")

    pass

if __name__=="__main__":
    sample_candidates=pd.DataFrame({
        'x':[339,300,291,222,72],
        'y':[170,193,244,218,219],
        'area':[74,62,214,74,55],
        'mean_intensity':[-569,-578,-563,-564,-571]
    })
    simulate_false_positive_reduction(sample_candidates)

