#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

// Parameters
params.data_path = null
params.sections = null
params.n_cores = 12
params.n_gb = 32
params.prior_seg_reassignment_prob = 0.5
params.cell_compactness = 0.1

// Validate required parameters
if (!params.data_path) {
    error "Please provide --data_path parameter"
}

workflow {
    if (params.sections) {
        // Multi-section mode: sections provided
        section_list = params.sections.split(',').collect { it.trim() }
        sections_ch = Channel.from(section_list)
            .map { section -> 
                def sample_id = file(params.data_path).name
                def full_data_path = file(params.data_path).parent
                tuple(sample_id, section, full_data_path, true)
            }
    } else {
        // Single-section mode: no sections provided, data is directly under data_path
        def data_path_file = file(params.data_path)
        def sample_id = data_path_file.name
        def parent_path = data_path_file.parent
        sections_ch = Channel.of(tuple(sample_id, null, parent_path, false))
    }
    
    // Run proseg refinement
    proseg_results = PROSEG_REFINE(sections_ch)
    
    // Run xeniumranger import
    XENIUMRANGER_IMPORT(proseg_results)
}

process PROSEG_REFINE {
    tag "${sample_id}${section_id ? '_' + section_id : ''}"
    maxForks 1
    
    input:
    tuple val(sample_id), val(section_id), path(data_path), val(multi_section)
    
    output:
    tuple val(sample_id), val(section_id), path(data_path), path(xenium_path), val(multi_section)
    
    script:
    if (multi_section) {
        xenium_path = "${data_path}/${sample_id}/${section_id}"
    } else {
        xenium_path = "${data_path}/${sample_id}"
    }
    
    """
    # Set up paths
    xenium_path="${xenium_path}"
    proseg_path="\${xenium_path}/baysor"
    
    # Create proseg directory
    mkdir -p \${proseg_path}
    
    # Set output file paths
    cell_polygon="\${proseg_path}/proseg-to-baysor-cell-polygons.geojson"
    transcript_metadata="\${proseg_path}/proseg-to-baysor-transcript-metadata.csv"
    
    # Run proseg segmentation using default Xenium workflow
    proseg --xenium \${xenium_path}/transcripts.csv.gz \
        --nthreads ${params.n_cores} \
        --prior-seg-reassignment-prob ${params.prior_seg_reassignment_prob} \
        --cell-compactness ${params.cell_compactness} \
        --output-spatialdata \${xenium_path}/output.zarr \
        --overwrite 
    
    # Convert to baysor format
    proseg-to-baysor \${xenium_path}/output.zarr \
        --output-transcript-metadata \${transcript_metadata} \
        --output-cell-polygons \${cell_polygon}
    """
}

process XENIUMRANGER_IMPORT {
    tag "${sample_id}${section_id ? '_' + section_id : ''}"
    publishDir "${params.data_path}", mode: 'copy'
    maxForks 1
    
    input:
    tuple val(sample_id), val(section_id), path(data_path), path(xenium_path), val(multi_section)
    
    output:
    path output_dir
    
    script:
    if (multi_section) {
        output_dir = "${section_id}_proseg"
        tmp_path = "${section_id}_tmp"
        baysor_dir = "${xenium_path}/baysor"
    } else {
        output_dir = "${sample_id}_proseg"
        tmp_path = "${sample_id}_tmp"
        baysor_dir = "${xenium_path}/baysor"
    }
    
    """
    # Set up paths
    xenium_data_path="${xenium_path}/"
    tmp_path="${tmp_path}"
    
    cell_polygon="${baysor_dir}/proseg-to-baysor-cell-polygons.geojson"
    transcript_metadata="${baysor_dir}/proseg-to-baysor-transcript-metadata.csv"
    
    # Run xeniumranger import-segmentation
    xeniumranger import-segmentation \
        --id \${tmp_path} \
        --xenium-bundle \${xenium_data_path} \
        --viz-polygons \${cell_polygon} \
        --transcript-assignment \${transcript_metadata} \
        --units microns \
        --localcores=${params.n_cores} \
        --localmem=${params.n_gb}

     mv \${tmp_path}/outs ${output_dir}
     rm -rf \${tmp_path}
    """
}

workflow.onComplete {
    println "Pipeline succeeded."
}