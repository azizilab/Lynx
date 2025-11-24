#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

// Parameters
params.data_path = null
params.sections = null
params.protocol_version = null  // 'V1' for nuclei-chan; 'V2' for multi-chan
params.factor = 0.2
params.diam = 50
params.n_cores = 12
params.n_gb = 32
params.cellpose_scale = 0.2125
params.prior_seg_reassignment_prob = 0.5

// Validate required parameters
if (!params.data_path) {
    error "Please provide --data_path parameter"
}
if (!params.sections) {
    error "Please provide --sections parameter (comma-separated list)"
}

// Convert sections string to list
section_list = params.sections.split(',').collect { it.trim() }

workflow {
    // Create channel with sample and section information
    sections_ch = Channel.from(section_list)
        .map { section -> 
            def sample_id = file(params.data_path).name
            def full_data_path = file(params.data_path).parent
            tuple(params.protocol_version, sample_id, section, full_data_path)
        }
    
    // Run resegmentation
    reseg_results = RESEGMENT(sections_ch)
    
    // Run proseg refinement
    proseg_results = PROSEG_REFINE(reseg_results)
    
    // Run xeniumranger import
    XENIUMRANGER_IMPORT(proseg_results)
}

process RESEGMENT {
    tag "${sample_id}_${section_id}"
    maxForks 1
    
    input:
    tuple val(protocol_version), val(sample_id), val(section_id), path(data_path)

    output:
    tuple val(sample_id), val(section_id), path(data_path), path("reseg_masks.npy")
    
    script:
    """
    python3 ${projectDir}/reseg.py \
        --data-path ${data_path} \
        --sample-id ${sample_id} \
        --section-id ${section_id} \
        --factor ${params.factor} \
        --protocol-version ${protocol_version} \
        --diam ${params.diam}
    
    # Copy the generated masks to working directory
    cp ${data_path}/${sample_id}/${section_id}/reseg_masks.npy .
    """
}

process PROSEG_REFINE {
    tag "${sample_id}_${section_id}"
    maxForks 1
    
    input:
    tuple val(sample_id), val(section_id), path(data_path), path(reseg_masks)
    
    output:
    tuple val(sample_id), val(section_id), path(data_path), path("${data_path}/${sample_id}/${section_id}/baysor")
    
    script:
    """
    # Set up paths
    xenium_path="${data_path}/${sample_id}/${section_id}"
    proseg_path="\${xenium_path}/baysor"
    
    # Copy reseg masks to xenium path
    cp ${reseg_masks} \${xenium_path}
    
    # Create proseg directory
    mkdir -p \${proseg_path}
    
    # Set output file paths
    cell_polygon="\${proseg_path}/proseg-to-baysor-cell-polygons.geojson"
    transcript_metadata="\${proseg_path}/proseg-to-baysor-transcript-metadata.csv"
    
    # Run proseg segmentation
    proseg --xenium \${xenium_path}/transcripts.parquet \
        --cellpose-masks \${xenium_path}/reseg_masks.npy \
        --cellpose-scale ${params.cellpose_scale} \
        --nthreads ${params.n_cores} \
        --prior-seg-reassignment-prob ${params.prior_seg_reassignment_prob} \
        --overwrite \
        --output-spatialdata \${xenium_path}/output.zarr
    
    # Convert to baysor format
    proseg-to-baysor \${xenium_path}/output.zarr \
        --output-transcript-metadata \${transcript_metadata} \
        --output-cell-polygons \${cell_polygon}
    """
}

process XENIUMRANGER_IMPORT {
    tag "${sample_id}_${section_id}"
    publishDir "${params.data_path}", mode: 'copy'
    maxForks 1
    
    input:
    tuple val(sample_id), val(section_id), path(data_path), path(baysor_dir)
    
    output:
    path "${section_id}_proseg"
    
    script:
    """
    # Set up paths
    xenium_path="${data_path}/${sample_id}/${section_id}/"
    tmp_path="${section_id}_tmp"
    
    cell_polygon="${baysor_dir}/proseg-to-baysor-cell-polygons.geojson"
    transcript_metadata="${baysor_dir}/proseg-to-baysor-transcript-metadata.csv"
    
    # Run xeniumranger import-segmentation
    xeniumranger import-segmentation \
        --id \${tmp_path} \
        --xenium-bundle \${xenium_path} \
        --viz-polygons \${cell_polygon} \
        --transcript-assignment \${transcript_metadata} \
        --units microns \
        --localcores=${params.n_cores} \
        --localmem=${params.n_gb}

     mv \${tmp_path}/outs ${section_id}_proseg
     rm -rf \${tmp_path}
    """
}

workflow.onComplete {
    println "Pipeline succeeded."
}