#!/bin/bash

# Example run script for Xenium Proseg Pipeline
# Usage: ./run_pipeline.sh <data_path> <sections>

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <data_path> <sections>"
    echo "Example: $0 /path/to/data section_01,section_02,section_03"
    exit 1
fi

DATA_PATH=$1
SECTIONS=$2

# Run the Nextflow pipeline
nextflow run xenium_proseg_pipeline.nf \
    --data_path ${DATA_PATH} \
    --sections ${SECTIONS} 

nextflow clean -f -a
    
echo "Pipeline completed."