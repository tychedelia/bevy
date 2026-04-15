// Expands GPU-authored instance-batch range entries into PreprocessWorkItems.
//
// Sibling to `unpack_bins.wgsl`. Each invocation handles one of
// `metadata.total_instance_count` instances across all ranges, finds its
// owning range via cumulative-offset scan, and writes one PreprocessWorkItem
// into the work-item buffer at `metadata.work_item_base + global_id`.

#import bevy_pbr::mesh_preprocess_types::PreprocessWorkItem

struct RangeUnpackingMetadata {
    range_count: u32,
    total_instance_count: u32,
    work_item_base: u32,
    pad: u32,
}

struct RangeWorkItem {
    base_input_index: u32,
    base_output_or_indirect_parameters_index: u32,
    count: u32,
    cumulative_offset: u32,
}

@group(0) @binding(0) var<uniform> metadata: RangeUnpackingMetadata;
@group(0) @binding(1) var<storage> ranges: array<RangeWorkItem>;
@group(0) @binding(2) var<storage, read_write> preprocess_work_items: array<PreprocessWorkItem>;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let global_id = global_invocation_id.x;
    if (global_id >= metadata.total_instance_count) {
        return;
    }

    // Linear scan to find the range containing this instance. Range counts
    // are typically small (one per GPU instance batch, usually < 100) so a
    // scan is cheaper than a binary search and keeps the shader tiny.
    var range_idx: u32 = 0u;
    for (var i: u32 = 0u; i < metadata.range_count; i = i + 1u) {
        let start = ranges[i].cumulative_offset;
        let end = start + ranges[i].count;
        if (global_id >= start && global_id < end) {
            range_idx = i;
            break;
        }
    }

    let range = ranges[range_idx];
    let i_in_range = global_id - range.cumulative_offset;
    let work_item_idx = metadata.work_item_base + global_id;

    preprocess_work_items[work_item_idx].input_index =
        range.base_input_index + i_in_range;
    preprocess_work_items[work_item_idx].output_or_indirect_parameters_index =
        range.base_output_or_indirect_parameters_index;
}
