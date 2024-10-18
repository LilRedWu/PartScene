

#!/bin/bash

#!/bin/bash


#######Zero-shot Part segmentation
# Propose 3D masks for per-instance segmentation
python part_seg_ppl.py

# Perform 2D part segmentation
python part_seg_2d_ppl.py

# Classify the generated masks
python mask_classification.py

# Evaluate the results of the previous steps
python evaluate.py