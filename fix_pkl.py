import pickle
import copy
import os

# Paths to your original files
train_path = "/storage/project/r-gchou3-0/spanse30/OpenPCDet/output/cfgs/custom_models/centerpoint_multiframe_waymo/default/eval/epoch_36/train/default/result_fixed.pkl"
val_path = "/storage/project/r-gchou3-0/spanse30/OpenPCDet/output/cfgs/custom_models/centerpoint_multiframe_waymo/default/eval/epoch_36/val/default/result_fixed.pkl"

def fix_file(path, suffix="_fixed"):
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return

    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"  - Loaded {len(data)} samples.")
    
    fixed_boxes_count = 0
    fixed_scores_count = 0
    
    for sample in data:
        # 1. Fix Boxes: boxes_lidar -> pred_boxes
        if 'boxes_lidar' in sample and 'pred_boxes' not in sample:
            sample['pred_boxes'] = sample['boxes_lidar']
            fixed_boxes_count += 1
            
        # 2. Fix Scores: score -> pred_scores
        if 'score' in sample and 'pred_scores' not in sample:
            sample['pred_scores'] = sample['score']
            fixed_scores_count += 1
            
    output_path = path.replace(".pkl", f"{suffix}.pkl")
    print(f"  - Added 'pred_boxes' to {fixed_boxes_count} samples.")
    print(f"  - Added 'pred_scores' to {fixed_scores_count} samples.")
    print(f"  - Saving to {output_path}...")
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print("Done!")

if __name__ == "__main__":
    fix_file(train_path)
    fix_file(val_path)