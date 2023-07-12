#! /bin/bash

#==========
set -exu
set -o pipefail
#==========

#==========preprocess  & its subsets
python preprocess_ind_data.py --task fb237_v1
python preprocess_ind_data.py --task fb237_v1_ind
python negative_sampling.py --task fb237_v1
python negative_sampling.py --task fb237_v1_ind

#python preprocess_ind_data.py --task fb237_v2
#python preprocess_ind_data.py --task fb237_v2_ind
#python negative_sampling.py --task fb237_v2
#python negative_sampling.py --task fb237_v2_ind

#python preprocess_ind_data.py --task fb237_v3
#python preprocess_ind_data.py --task fb237_v3_ind
#python negative_sampling.py --task fb237_v3
#python negative_sampling.py --task fb237_v3_ind
#
#python preprocess_ind_data.py --task fb237_v4
#python preprocess_ind_data.py --task fb237_v4_ind
#python negative_sampling.py --task fb237_v4
#python negative_sampling.py --task fb237_v4_ind

#==========preprocess  & its subsets
#python preprocess_ind_data.py --task fb237_v1_ind
#python preprocess_ind_data.py --task fb237_v2_ind
#python preprocess_ind_data.py --task fb237_v3_ind
#python preprocess_ind_data.py --task fb237_v4_ind
#
##==========preprocess  & its subsets
#python preprocess_ind_data.py --task nell_v1
#python preprocess_ind_data.py --task nell_v2
#python preprocess_ind_data.py --task nell_v3
#python preprocess_ind_data.py --task nell_v4
#
##==========preprocess  & its subsets
#python preprocess_ind_data.py --task nell_v1_ind
#python preprocess_ind_data.py --task nell_v2_ind
#python preprocess_ind_data.py --task nell_v3_ind
#python preprocess_ind_data.py --task nell_v4_ind
#
#python preprocess_ind_data.py --task WN18RR_v1
#python preprocess_ind_data.py --task WN18RR_v1_ind
#python negative_sampling.py --task WN18RR_v1
#python negative_sampling.py --task WN18RR_v1_ind

#python preprocess_ind_data.py --task WN18RR_v2
#python preprocess_ind_data.py --task WN18RR_v2_ind
#python negative_sampling.py --task WN18RR_v2
#python negative_sampling.py --task WN18RR_v2_ind

#python preprocess_ind_data.py --task WN18RR_v3
#python preprocess_ind_data.py --task WN18RR_v3_ind
#python negative_sampling.py --task WN18RR_v3
#python negative_sampling.py --task WN18RR_v3_ind
#
#python preprocess_ind_data.py --task WN18RR_v4
#python preprocess_ind_data.py --task WN18RR_v4_ind
#python negative_sampling.py --task WN18RR_v4
#python negative_sampling.py --task WN18RR_v4_ind

#python preprocess_ind_data.py --task nell_v1
#python preprocess_ind_data.py --task nell_v1_ind
#python negative_sampling.py --task nell_v1
#python negative_sampling.py --task nell_v1_ind

#python preprocess_ind_data.py --task nell_v2
#python preprocess_ind_data.py --task nell_v2_ind
#python negative_sampling.py --task nell_v2
#python negative_sampling.py --task nell_v2_ind
#
#python preprocess_ind_data.py --task nell_v3
#python preprocess_ind_data.py --task nell_v3_ind
#python negative_sampling.py --task nell_v3
#python negative_sampling.py --task nell_v3_ind
#
#python preprocess_ind_data.py --task nell_v4
#python preprocess_ind_data.py --task nell_v4_ind
#python negative_sampling.py --task nell_v4
#python negative_sampling.py --task nell_v4_ind