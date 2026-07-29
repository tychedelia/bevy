// Expands GPU-authored instance-batch ranges into `PreprocessWorkItem`s.
//
// The CPU pushes one `RangeWorkItem` per batch; this shader writes one work
// item per instance slot, which the mesh preprocessing shader then consumes
// exactly as it does the CPU-built work items.
//
// Both bindings are sized to this frame's contents, so `arrayLength` returns
// the range count and the total instance count respectively.

#import bevy_pbr::mesh_preprocess_types::PreprocessWorkItem

struct RangeWorkItem {
    base_input_index: u32,
    indirect_parameters_index: u32,
    count: u32,
    cumulative_offset: u32,
}

@group(0) @binding(0) var<storage> ranges: array<RangeWorkItem>;
@group(0) @binding(1) var<storage, read_write> work_items: array<PreprocessWorkItem>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let instance_index = global_invocation_id.x;
    if (instance_index >= arrayLength(&work_items)) {
        return;
    }

    // Binary search for the last range whose cumulative offset is at most
    // `instance_index`. The ranges are sorted by cumulative offset by
    // construction.
    var lo = 0u;
    var hi = arrayLength(&ranges);
    while (lo + 1u < hi) {
        let mid = (lo + hi) / 2u;
        if (ranges[mid].cumulative_offset <= instance_index) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let range = ranges[lo];
    let offset_in_range = instance_index - range.cumulative_offset;
    work_items[instance_index].input_index = range.base_input_index + offset_in_range;
    work_items[instance_index].output_or_indirect_parameters_index =
        range.indirect_parameters_index;
}
