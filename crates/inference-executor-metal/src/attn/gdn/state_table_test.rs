use std::mem::size_of;
use std::path::PathBuf;

use half::bf16;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::attn::GDNReplayShape;
use inference_executor_core::attn::gdn::GDNOutputGate;
use inference_runtime_core::compute::ExecutorHibernationPlan;

use super::GDNRequestStateTable;
use super::GDNStateCapacity;
use crate::attn::gdn::batch_metadata::GDNMetadataBuffers;
use crate::attn::gdn::request_slots::GDNRequestSlots;
use crate::model::state_snapshot::FullStateIO;
use crate::model::state_snapshot::GDNStateSnapshotFiles;
use crate::model::state_snapshot::SelectedStateIO;
use crate::model::state_snapshot::StateSnapshotFile;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;

const TEST_PAGE_BYTES: usize = 32 * 1024;
const LIFECYCLE_PAGE_BYTES: usize = 16;
const TEST_NUM_CACHE_PAGES: usize = 1024;
const SNAPSHOT_FILES: GDNStateSnapshotFiles = GDNStateSnapshotFiles::new(
    StateSnapshotFile::MainGDNRequestStateTable,
    StateSnapshotFile::MainGDNRecurrentState,
    StateSnapshotFile::MainGDNConvState,
);

#[derive(Debug, PartialEq)]
struct GDNStateReference {
    request_state: GDNRequestSlots,
    recurrent_state: Vec<u16>,
    conv_state: Vec<u16>,
}

#[test]
fn test_full_state_unload_load_fixed() {
    let device = Device::system_default();
    let mut state = new_lifecycle_state(&device);
    let pages_per_state = state.num_pages_per_state_slot();
    let page_ids = (0..2 * pages_per_state)
        .map(|index| u32::try_from(10 + index).unwrap())
        .collect::<Vec<_>>();
    populate_durable_request_state(&state, &device, &page_ids);
    advance_to_distinct_state_slots(&state, 1);

    let (num_recurrent_values, num_conv_values) = state_value_counts(&state);
    let recurrent_state = fixed_values(num_recurrent_values, 0.25);
    let conv_state = fixed_values(num_conv_values, -0.5);
    write_state_values(&state, &recurrent_state, &conv_state);
    let reference = capture_state(&state);

    assert_eq!(reference.recurrent_state, recurrent_state);
    assert_eq!(reference.conv_state, conv_state);
    assert_unload_load("fixed", &device, &mut state, reference);
}

#[test]
fn test_full_state_unload_load_random() {
    let device = Device::system_default();
    let mut state = new_lifecycle_state(&device);
    let mut random = TestRandom::new(0x4744_4e5f_5354_4154);
    let page_ids = (0..2 * state.num_pages_per_state_slot())
        .map(|_| random.next_u32() % 1024)
        .collect::<Vec<_>>();
    populate_durable_request_state(&state, &device, &page_ids);
    advance_to_distinct_state_slots(&state, 1);

    let (num_recurrent_values, num_conv_values) = state_value_counts(&state);
    let recurrent_state = random.values(num_recurrent_values);
    let conv_state = random.values(num_conv_values);
    write_state_values(&state, &recurrent_state, &conv_state);
    let reference = capture_state(&state);

    assert_unload_load("random", &device, &mut state, reference);
}

#[test]
fn test_selected_state_unload_load() {
    let device = Device::system_default();
    let mut state = GDNRequestStateTable::new(
        &device,
        &[core(0), core(1)],
        3,
        GDNStateCapacity::new(3, 2, 1, 8),
        2,
        TEST_NUM_CACHE_PAGES,
        LIFECYCLE_PAGE_BYTES,
        32,
    );
    advance_to_distinct_state_slots_for_req(&state, 1, 0);
    let selected_recurrent_state_slot =
        usize::try_from(state.request_table().borrow().current_recurrent_state_slot(1)).unwrap();
    let selected_conv_state_slot = usize::try_from(state.request_table().borrow().current_conv_state_slot(1)).unwrap();
    assert_ne!(selected_recurrent_state_slot, selected_conv_state_slot);
    let request_state = state.request_table().borrow().clone();
    let (num_recurrent_values, num_conv_values) = state_value_counts(&state);
    let recurrent_state = fixed_values(num_recurrent_values, 0.25);
    let conv_state = fixed_values(num_conv_values, -0.5);
    write_state_values(&state, &recurrent_state, &conv_state);

    let selected_request_slot_ranges = std::iter::once(1..2).collect::<Vec<_>>();
    let plan = ExecutorHibernationPlan::selected(selected_request_slot_ranges.to_vec(), Vec::new());
    let snapshot_path = snapshot_path("selected");
    let buffer_io = inference_backend_metal::metal::BufferIO::new(&device);
    let snapshot_files = [
        SNAPSHOT_FILES.request_state_table(),
        SNAPSHOT_FILES.recurrent_state(),
        SNAPSHOT_FILES.conv_state(),
    ];
    let mut writer = StateSnapshotWriter::new(&snapshot_path, &snapshot_files, &plan, &buffer_io).unwrap();
    state
        .write_selected_state(&mut writer, SNAPSHOT_FILES, &selected_request_slot_ranges)
        .unwrap();
    writer.commit().unwrap();

    state.release_resources();
    state.allocate_resources(&device);
    let mut reader = StateSnapshotReader::open(&snapshot_path, &snapshot_files, &plan, &buffer_io).unwrap();
    state
        .read_selected_state(&mut reader, SNAPSHOT_FILES, &selected_request_slot_ranges)
        .unwrap();
    reader.finish().unwrap();

    let restored = capture_state(&state);
    assert_eq!(restored.request_state, request_state);
    assert_selected_state_values(
        &restored.recurrent_state,
        &recurrent_state,
        state.layout.num_gdn_layers,
        state.layout.num_state_slots,
        selected_recurrent_state_slot,
        state.recurrent_state_bytes() / size_of::<u16>(),
    );
    assert_selected_state_values(
        &restored.conv_state,
        &conv_state,
        state.layout.num_gdn_layers,
        state.layout.num_state_slots,
        selected_conv_state_slot,
        state.conv_state_bytes() / size_of::<u16>(),
    );
    std::fs::remove_dir_all(snapshot_path).unwrap();
}

#[test]
#[should_panic(expected = "GDN state slots must include current state and all materialized states")]
fn test_capacity_requires_current_state_slot() {
    let _ = GDNStateCapacity::new(3, 3, 1, 8);
}

#[test]
fn test_layout() {
    let device = Device::system_default();
    let state = GDNRequestStateTable::new(
        &device,
        &[core(0), core(1)],
        2,
        GDNStateCapacity::new(4, 3, 1, 8),
        16,
        TEST_NUM_CACHE_PAGES,
        TEST_PAGE_BYTES,
        32,
    );

    assert_eq!(state.layer_bindings(0).recurrent_layer_offset_bytes, 0);
    assert_eq!(state.layer_bindings(0).conv_layer_offset_bytes, 0);
    let log = state.layer_bindings(1).replay;
    assert_eq!(log.num_total_tokens, 16);
    assert_eq!(log.token_offset, 16);
    assert_eq!(log.alpha.len_bytes(), 2 * 16 * size_of::<f32>());
    assert_eq!(log.k.len_bytes(), 2 * 16 * 4 * size_of::<f32>());
    assert_eq!(log.u.len_bytes(), 2 * 16 * 4 * size_of::<f32>());
    assert_eq!(log.qkv.len_bytes(), 2 * 16 * 12 * size_of::<u16>());
    assert_eq!(
        state.layer_bindings(1).recurrent_layer_offset_bytes,
        8 * 16 * size_of::<u16>() as u64
    );
    assert_eq!(
        state.layer_bindings(1).conv_layer_offset_bytes,
        8 * 24 * size_of::<u16>() as u64
    );
    assert_eq!(
        state.layer_bindings(0).recurrent_states.as_raw_ptr(),
        state.layer_bindings(1).recurrent_states.as_raw_ptr()
    );
    assert_eq!(
        state.layer_bindings(0).conv_states.as_raw_ptr(),
        state.layer_bindings(1).conv_states.as_raw_ptr()
    );
    assert_eq!(state.num_pages_per_state_slot(), 4);
}

#[test]
#[should_panic(expected = "runtime supplied a page ID outside the cache-page buffer")]
fn test_page_id_domain_panics() {
    let device = Device::system_default();
    let state = GDNRequestStateTable::new(&device, &[core(0)], 1, GDNStateCapacity::new(3, 2, 1, 8), 2, 2, 16, 32);
    state.prepare(
        &[0],
        &[0],
        &[0],
        &[0, 1],
        &[0],
        1,
        &[vec![vec![2; state.num_pages_per_state_slot()]]],
    );
}

#[test]
fn test_mixed_state_commit_and_deferred_publish() {
    let device = Device::system_default();
    let state = GDNRequestStateTable::new(
        &device,
        &[core(0), core(1)],
        3,
        GDNStateCapacity::new(4, 3, 2, 8),
        2,
        TEST_NUM_CACHE_PAGES,
        LIFECYCLE_PAGE_BYTES,
        32,
    );
    let metadata = GDNMetadataBuffers::new(&device, 3, 12);
    let pages_per_state = state.num_pages_per_state_slot() as u32;
    let pages_2 = (10..10 + pages_per_state).collect::<Vec<_>>();
    let pages_4 = (20..20 + pages_per_state).collect::<Vec<_>>();
    let pages_6 = (30..30 + pages_per_state).collect::<Vec<_>>();
    prepare_state(
        &state,
        &metadata,
        &[0, 1, 2],
        &[0, 0, 0],
        &[0, 0, 0],
        &[0, 4, 8, 12],
        &[0, 2, 4],
        1,
        &[
            vec![pages_2.clone(), pages_4.clone(), pages_6.clone()],
            Vec::new(),
            Vec::new(),
        ],
    );
    let table = state.request_table().borrow();
    let rec_2 = table.materialized_recurrent_state_slot(0, 2);
    let rec_4 = table.materialized_recurrent_state_slot(0, 4);
    let conv_2 = table.materialized_conv_state_slot(0, 2);
    let conv_4 = table.materialized_conv_state_slot(0, 4);
    let source_rec = (0..3)
        .map(|slot| table.current_recurrent_state_slot(slot))
        .collect::<Vec<_>>();
    let source_conv = (0..3)
        .map(|slot| table.current_conv_state_slot(slot))
        .collect::<Vec<_>>();
    drop(table);
    assert_eq!(
        metadata.flat_recurrent_state_write_slots().read_typed::<u32>(0, 12),
        vec![
            u32::MAX,
            rec_2,
            u32::MAX,
            rec_4,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX
        ]
    );
    assert_eq!(
        metadata.flat_conv_state_write_slots().read_typed::<u32>(0, 12),
        vec![
            u32::MAX,
            conv_2,
            u32::MAX,
            conv_4,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX
        ]
    );
    let jobs = state.commit(&[4, 3, 0]);
    assert_eq!(jobs.len(), 1);
    assert_eq!((jobs[0].replay_token_begin, jobs[0].num_tokens), (0, 3));
    assert_eq!(jobs[0].src_recurrent_state_slot, source_rec[1]);
    assert_eq!(jobs[0].src_conv_state_slot, source_conv[1]);
    assert!(!source_rec.contains(&jobs[0].dst_recurrent_state_slot));
    assert!(!source_conv.contains(&jobs[0].dst_conv_state_slot));
    let table = state.request_table().borrow();
    assert_eq!(table.current_recurrent_state_slot(0), rec_4);
    assert_eq!(table.current_conv_state_slot(0), conv_4);
    assert_eq!(table.current_recurrent_state_slot(1), jobs[0].dst_recurrent_state_slot);
    assert_eq!(table.current_conv_state_slot(1), jobs[0].dst_conv_state_slot);
    assert_eq!(table.current_recurrent_state_slot(2), source_rec[2]);
    assert_eq!(table.current_conv_state_slot(2), source_conv[2]);
    assert_eq!(
        (0..3).map(|slot| table.current_state_version(slot)).collect::<Vec<_>>(),
        [4, 3, 0]
    );
    drop(table);
    assert_eq!(
        state
            .publishes()
            .iter()
            .map(|p| (p.state_version, p.page_ids.clone()))
            .collect::<Vec<_>>(),
        [(2, pages_2), (4, pages_4)]
    );
    state.finish_commit();
    // A previously queued boundary is materialized only after its prefix is accepted.
    prepare_state(&state, &metadata, &[0], &[2], &[4], &[0, 2], &[1], 0, &[Vec::new()]);
    assert_eq!(
        metadata.flat_recurrent_state_write_slots().read_typed::<u32>(0, 2),
        [u32::MAX; 2]
    );
    let jobs = state.commit(&[6]);
    assert_eq!(jobs.len(), 1);
    assert_eq!((jobs[0].replay_token_begin, jobs[0].num_tokens), (0, 2));
    assert_eq!(state.publishes()[0].state_version, 6);
    assert_eq!(state.publishes()[0].page_ids, pages_6);
}

#[test]
fn test_restore_and_reset_preserve_neighbor_request_state() {
    let device = Device::system_default();
    let state = GDNRequestStateTable::new(
        &device,
        &[core(0), core(1)],
        2,
        GDNStateCapacity::new(4, 3, 1, 8),
        1024,
        TEST_NUM_CACHE_PAGES,
        LIFECYCLE_PAGE_BYTES,
        32,
    );
    let batch_metadata = GDNMetadataBuffers::new(&device, 1, 1);
    let snapshot_page_ids = (10..10 + u32::try_from(state.num_pages_per_state_slot()).unwrap()).collect::<Vec<_>>();
    advance_to_distinct_state_slots_for_req(&state, 0, 0);
    advance_to_distinct_state_slots_for_req(&state, 1, 0);
    let (
        restore_recurrent_state_slot,
        restore_conv_state_slot,
        neighbor_recurrent_state_slot,
        neighbor_conv_state_slot,
        neighbor_state_version,
    ) = {
        let table = state.request_table().borrow();
        (
            table.current_recurrent_state_slot(0),
            table.current_conv_state_slot(0),
            table.current_recurrent_state_slot(1),
            table.current_conv_state_slot(1),
            table.current_state_version(1),
        )
    };
    prepare_state(
        &state,
        &batch_metadata,
        &[0],
        &[0],
        &[1024],
        &[0, 1],
        &[0],
        1,
        &[vec![snapshot_page_ids.clone()]],
    );

    assert_eq!(state.restores().len(), 1);
    assert_eq!(state.restores()[0].state_version, 1024);
    assert_eq!(state.restores()[0].page_ids, snapshot_page_ids);
    assert_eq!(
        state.restores()[0].dst_recurrent_state_slot,
        restore_recurrent_state_slot
    );
    assert_eq!(state.restores()[0].dst_conv_state_slot, restore_conv_state_slot);
    assert_eq!(
        batch_metadata.src_recurrent_state_slots().read_typed::<u32>(0, 1),
        [restore_recurrent_state_slot]
    );
    assert_eq!(
        batch_metadata.src_conv_state_slots().read_typed::<u32>(0, 1),
        [restore_conv_state_slot]
    );
    state.finish_restore();
    state.commit(&[1025]);

    let (num_recurrent_values, num_conv_values) = state_value_counts(&state);
    let recurrent_state = fixed_values(num_recurrent_values, 1.0);
    let conv_state = fixed_values(num_conv_values, -1.0);
    write_state_values(&state, &recurrent_state, &conv_state);
    state.reset_req_slot(0);

    let table = state.request_table().borrow();
    let reset_recurrent_state_slot = table.current_recurrent_state_slot(0) as usize;
    let reset_conv_state_slot = table.current_conv_state_slot(0) as usize;
    assert_eq!(table.current_state_version(0), 0);
    assert_eq!(table.current_recurrent_state_slot(1), neighbor_recurrent_state_slot);
    assert_eq!(table.current_conv_state_slot(1), neighbor_conv_state_slot);
    assert_eq!(table.current_state_version(1), neighbor_state_version);
    drop(table);

    assert_reset_state_values(
        &state,
        &recurrent_state,
        &conv_state,
        reset_recurrent_state_slot,
        reset_conv_state_slot,
    );
}

#[test]
#[should_panic(expected = "GDN current state version must match the runtime input token index")]
fn test_prepare_requires_source_state_for_first_input_token() {
    let device = Device::system_default();
    let state = GDNRequestStateTable::new(
        &device,
        &[core(0), core(1)],
        1,
        GDNStateCapacity::new(4, 3, 1, 8),
        2,
        TEST_NUM_CACHE_PAGES,
        16,
        32,
    );
    let batch_metadata = GDNMetadataBuffers::new(&device, 1, 1);

    prepare_state(
        &state,
        &batch_metadata,
        &[0],
        &[0],
        &[2],
        &[0, 1],
        &[0],
        1,
        &[Vec::new()],
    );
}

#[test]
fn test_decode_commit_selects_next_main_source_and_saves_crossed_boundary() {
    let device = Device::system_default();
    for num_accepted_tokens in 0..=3 {
        let state = GDNRequestStateTable::new(
            &device,
            &[core(0)],
            1,
            GDNStateCapacity::new(7, 6, 2, 8),
            4,
            TEST_NUM_CACHE_PAGES,
            LIFECYCLE_PAGE_BYTES,
            32,
        );
        let metadata = GDNMetadataBuffers::new(&device, 1, 4);
        prepare_state(&state, &metadata, &[0], &[0], &[0], &[0, 3], &[0], 1, &[Vec::new()]);
        state.commit(&[3]);
        state.finish_commit();

        let page_ids = (0..state.num_pages_per_state_slot() as u32).collect::<Vec<_>>();
        // S3 --w--> S4 --x1--> S5 --x2--> S6 --x3--> S7.
        // The Main block ending at S4 is stable for every rejection outcome.
        prepare_state(
            &state,
            &metadata,
            &[0],
            &[0],
            &[3],
            &[0, 4],
            &[3],
            0,
            &[vec![page_ids.clone()]],
        );
        let selected_version = 4 + num_accepted_tokens;
        assert_eq!(
            metadata.flat_recurrent_state_write_slots().read_typed::<u32>(0, 4),
            [u32::MAX; 4]
        );
        let jobs = state.commit(&[selected_version]);
        let selected_job = jobs.last().unwrap();
        let (recurrent_slot, conv_slot) = (selected_job.dst_recurrent_state_slot, selected_job.dst_conv_state_slot);
        assert_eq!(selected_job.num_tokens, selected_version - 3);
        assert_eq!(jobs[0].num_tokens, 1);
        assert_eq!(state.publishes().len(), 1);
        assert_eq!(state.publishes()[0].state_version, 4);
        assert_eq!(state.publishes()[0].page_ids, page_ids);
        state.finish_commit();

        // y is pending. Its forward must read the selected state, without replaying w/x1.
        prepare_state(
            &state,
            &metadata,
            &[0],
            &[0],
            &[selected_version],
            &[0, 1],
            &[0],
            0,
            &[Vec::new()],
        );
        assert_eq!(
            metadata.src_recurrent_state_slots().read_typed::<u32>(0, 1),
            [recurrent_slot]
        );
        assert_eq!(metadata.src_conv_state_slots().read_typed::<u32>(0, 1), [conv_slot]);
        let jobs = state.commit(&[selected_version + 1]);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].num_tokens, 1);
    }
}

#[test]
fn test_commit_publish_restore_reuse_recording() {
    use inference_backend_metal::metal::Buffer;
    use inference_backend_metal::metal::ReplayArguments;
    use inference_backend_metal::metal::ReplayParameterKey;
    use inference_backend_metal::metal::Stream;

    use crate::def::replay_op::MetalReplayRuntime;

    let device = Device::system_default();
    let stream = Stream::new(&device);
    let runtime = MetalReplayRuntime::new(&stream);
    let state = GDNRequestStateTable::new(
        &device,
        &[core(0), core(1)],
        2,
        GDNStateCapacity::new(4, 3, 2, 8),
        2,
        TEST_NUM_CACHE_PAGES,
        LIFECYCLE_PAGE_BYTES,
        8,
    );
    let metadata = GDNMetadataBuffers::new(&device, 2, 5);
    let num_pages = state.num_pages_per_state_slot() as u32;
    let pages = Buffer::from_slice(
        &device,
        &vec![0x1234_u16; TEST_NUM_CACHE_PAGES * LIFECYCLE_PAGE_BYTES / 2],
    );
    let mut replay = None;
    let mut restore_replay = None;
    let active_jobs = ReplayParameterKey::new("test.gdn.commit.num_active_jobs");
    let active_publishes = ReplayParameterKey::new("test.gdn.publish.num_active_requests");
    let active_restores = ReplayParameterKey::new("test.gdn.restore.num_active_requests");
    // The second use changes page IDs and log values without recording new commands.
    for iteration in 0..2 {
        state.reset_req_slots(&[0, 1]);
        let boundary_pages = (iteration * 2 * num_pages..(iteration * 2 + 1) * num_pages).collect::<Vec<_>>();
        let rejected_pages = ((iteration * 2 + 1) * num_pages..(iteration * 2 + 2) * num_pages).collect::<Vec<_>>();
        prepare_state(
            &state,
            &metadata,
            &[0, 1],
            &[0, 0],
            &[0, 0],
            &[0, 1, 5],
            &[0, 3],
            0,
            &[Vec::new(), vec![boundary_pages.clone(), rejected_pages]],
        );
        // Synthetic replay logs isolate commit/publish from the separately tested forward math.
        for layer in 0..2 {
            let log = state.layer_bindings(layer).replay;
            let start = log.token_offset as usize;
            log.alpha.write_typed(start, &[0.5_f32; 5]);
            log.k.write_typed(start * 4, &[0.25_f32; 20]);
            log.u.write_typed(start * 4, &[0.0625_f32; 20]);
            log.qkv.write_typed(
                start * 12,
                &[bf16::from_f32(layer as f32 + iteration as f32 + 1.0).to_bits(); 60],
            );
        }
        let jobs = state.commit(&[1, 3]);
        assert_eq!(jobs.len(), 3);
        let num_publishes = state.prepare_publish(&pages);
        assert!(num_publishes > 0);
        let replay = replay.get_or_insert_with(|| {
            let mut recorder = runtime.create_recorder();
            state.record_commit(&mut recorder, jobs.len() as u32, ReplayU32::Parameter(active_jobs));
            state.record_publish(
                &mut recorder,
                &pages,
                num_publishes,
                ReplayU32::Parameter(active_publishes),
            );
            recorder.build()
        });
        let arguments = ReplayArguments::new()
            .with_u32(active_jobs, jobs.len() as u32)
            .with_u32(active_publishes, num_publishes);
        runtime.submit_replay_with_arguments(replay, &arguments).wait();
        let resources = state.resources();
        for layer in 0..2 {
            for job in &jobs {
                let mut expected = 0.0_f32;
                for _ in 0..job.num_tokens {
                    expected = 0.5 * expected + 0.25 * 0.0625;
                }
                let offset = (layer * state.layout.num_state_slots + job.dst_recurrent_state_slot as usize) * 16;
                assert_eq!(
                    resources.recurrent_states.read_typed::<u16>(offset, 16),
                    vec![bf16::from_f32(expected).to_bits(); 16]
                );
            }
            // Each layer has two recurrent pages followed by three convolution pages.
            let page_start = (iteration as usize * 2 * num_pages as usize + layer * 5) * LIFECYCLE_PAGE_BYTES / 2;
            assert_eq!(
                pages.read_typed::<u16>(page_start, 16),
                vec![bf16::from_f32(0.0234375).to_bits(); 16]
            );
            assert_eq!(
                pages.read_typed::<u16>(page_start + 16, 24),
                vec![bf16::from_f32(layer as f32 + iteration as f32 + 1.0).to_bits(); 24]
            );
        }
        assert_eq!(
            pages.read_typed::<u16>(
                (iteration as usize * 2 + 1) * num_pages as usize * LIFECYCLE_PAGE_BYTES / 2,
                num_pages as usize * LIFECYCLE_PAGE_BYTES / 2
            ),
            vec![0x1234; num_pages as usize * LIFECYCLE_PAGE_BYTES / 2]
        );
        state.finish_commit();
        state.assert_snapshot_ready();

        state.reset_req_slot(1);
        prepare_state(
            &state,
            &metadata,
            &[1],
            &[0],
            &[2],
            &[0, 1],
            &[0],
            1,
            &[vec![boundary_pages]],
        );
        let num_restores = state.prepare_restore(&pages);
        assert_eq!(num_restores, 1);
        let restore_replay = restore_replay.get_or_insert_with(|| {
            let mut recorder = runtime.create_recorder();
            state.record_restore(
                &mut recorder,
                &pages,
                num_restores,
                ReplayU32::Parameter(active_restores),
            );
            recorder.build()
        });
        runtime
            .submit_replay_with_arguments(
                restore_replay,
                &ReplayArguments::new().with_u32(active_restores, num_restores),
            )
            .wait();
        let restore = &state.restores()[0];
        for layer in 0..2 {
            let recurrent_start =
                (layer * state.layout.num_state_slots + restore.dst_recurrent_state_slot as usize) * 16;
            let conv_start = (layer * state.layout.num_state_slots + restore.dst_conv_state_slot as usize) * 24;
            assert_eq!(
                resources.recurrent_states.read_typed::<u16>(recurrent_start, 16),
                vec![bf16::from_f32(0.0234375).to_bits(); 16]
            );
            assert_eq!(
                resources.conv_states.read_typed::<u16>(conv_start, 24),
                vec![bf16::from_f32(layer as f32 + iteration as f32 + 1.0).to_bits(); 24]
            );
        }
        state.finish_restore();
        // This test checks state I/O; the recurrent forward math has separate coverage.
        state.commit(&[3]);
        state.finish_commit();
        state.assert_snapshot_ready();
    }
}

fn assert_unload_load(name: &str, device: &Device, state: &mut GDNRequestStateTable, reference: GDNStateReference) {
    let snapshot_path = snapshot_path(name);
    let buffer_io = inference_backend_metal::metal::BufferIO::new(device);
    let snapshot_files = [
        SNAPSHOT_FILES.request_state_table(),
        SNAPSHOT_FILES.recurrent_state(),
        SNAPSHOT_FILES.conv_state(),
    ];
    let mut writer = StateSnapshotWriter::new(
        &snapshot_path,
        &snapshot_files,
        &ExecutorHibernationPlan::All,
        &buffer_io,
    )
    .unwrap();
    state.write_full_state(&mut writer, SNAPSHOT_FILES).unwrap();
    writer.commit().unwrap();

    state.release_resources();
    state.allocate_resources(device);

    let mut reader = StateSnapshotReader::open(
        &snapshot_path,
        &snapshot_files,
        &ExecutorHibernationPlan::All,
        &buffer_io,
    )
    .unwrap();
    state.read_full_state(&mut reader, SNAPSHOT_FILES).unwrap();
    reader.finish().unwrap();

    assert_eq!(capture_state(state), reference);
    std::fs::remove_dir_all(snapshot_path).unwrap();
}

fn new_lifecycle_state(device: &Device) -> GDNRequestStateTable {
    GDNRequestStateTable::new(
        device,
        &[core(0), core(1)],
        1,
        GDNStateCapacity::new(5, 4, 3, 8),
        2,
        TEST_NUM_CACHE_PAGES,
        LIFECYCLE_PAGE_BYTES,
        32,
    )
}

fn populate_durable_request_state(state: &GDNRequestStateTable, device: &Device, page_ids: &[u32]) {
    let pages_per_state = state.num_pages_per_state_slot();
    assert_eq!(page_ids.len(), 2 * pages_per_state);
    let metadata = GDNMetadataBuffers::new(device, 1, 1);
    prepare_state(
        state,
        &metadata,
        &[0],
        &[0],
        &[0],
        &[0, 1],
        &[0],
        1,
        &[page_ids.chunks_exact(pages_per_state).map(<[u32]>::to_vec).collect()],
    );
    state.commit(&[1]);
}

fn advance_to_distinct_state_slots(state: &GDNRequestStateTable, current_state_version: u32) {
    advance_to_distinct_state_slots_for_req(state, 0, current_state_version);
}

fn advance_to_distinct_state_slots_for_req(state: &GDNRequestStateTable, req_slot: u32, current_state_version: u32) {
    let first_state_version = current_state_version + 1;
    let second_state_version = first_state_version + 1;
    let mut request_table = state.request_table().borrow_mut();
    request_table.prepare_states(
        req_slot,
        &[first_state_version],
        &[first_state_version, second_state_version],
    );
    let _ = request_table.commit(req_slot, first_state_version);
    request_table.prepare_states(req_slot, &[second_state_version], &[second_state_version]);
    let _ = request_table.commit(req_slot, second_state_version);
    assert_ne!(
        request_table.current_recurrent_state_slot(req_slot),
        request_table.current_conv_state_slot(req_slot)
    );
}

fn write_state_values(state: &GDNRequestStateTable, recurrent_state: &[u16], conv_state: &[u16]) {
    let resources = state.resources();
    assert_eq!(
        std::mem::size_of_val(recurrent_state),
        resources.recurrent_states.len_bytes()
    );
    assert_eq!(std::mem::size_of_val(conv_state), resources.conv_states.len_bytes());
    resources.recurrent_states.write_typed(0, recurrent_state);
    resources.conv_states.write_typed(0, conv_state);
}

fn capture_state(state: &GDNRequestStateTable) -> GDNStateReference {
    let resources = state.resources();
    let (num_recurrent_values, num_conv_values) = state_value_counts(state);
    GDNStateReference {
        request_state: state.request_table().borrow().clone(),
        recurrent_state: resources.recurrent_states.read_typed(0, num_recurrent_values),
        conv_state: resources.conv_states.read_typed(0, num_conv_values),
    }
}

fn state_value_counts(state: &GDNRequestStateTable) -> (usize, usize) {
    let resources = state.resources();
    (
        resources.recurrent_states.len_bytes() / size_of::<u16>(),
        resources.conv_states.len_bytes() / size_of::<u16>(),
    )
}

fn fixed_values(len: usize, offset: f32) -> Vec<u16> {
    (0..len)
        .map(|index| bf16::from_f32(((index % 17) as f32 - 8.0) * 0.25 + offset).to_bits())
        .collect()
}

fn assert_selected_state_values(
    restored: &[u16],
    source: &[u16],
    num_layers: usize,
    num_state_slots: usize,
    selected_state_slot: usize,
    values_per_state: usize,
) {
    let mut expected = vec![0_u16; source.len()];
    for layer_index in 0..num_layers {
        let start = (layer_index * num_state_slots + selected_state_slot) * values_per_state;
        let end = start + values_per_state;
        expected[start..end].copy_from_slice(&source[start..end]);
    }
    assert_eq!(restored, expected);
}

fn assert_reset_state_values(
    state: &GDNRequestStateTable,
    recurrent_before_reset: &[u16],
    conv_before_reset: &[u16],
    reset_recurrent_state_slot: usize,
    reset_conv_state_slot: usize,
) {
    let mut expected_recurrent = recurrent_before_reset.to_vec();
    let mut expected_conv = conv_before_reset.to_vec();
    let recurrent_values_per_state = state.recurrent_state_bytes() / size_of::<u16>();
    let conv_values_per_state = state.conv_state_bytes() / size_of::<u16>();
    for layer_index in 0..state.layout.num_gdn_layers {
        let recurrent_start =
            (layer_index * state.layout.num_state_slots + reset_recurrent_state_slot) * recurrent_values_per_state;
        expected_recurrent[recurrent_start..recurrent_start + recurrent_values_per_state].fill(0);
        let conv_start = (layer_index * state.layout.num_state_slots + reset_conv_state_slot) * conv_values_per_state;
        expected_conv[conv_start..conv_start + conv_values_per_state].fill(0);
    }
    let restored = capture_state(state);
    assert_eq!(restored.recurrent_state, expected_recurrent);
    assert_eq!(restored.conv_state, expected_conv);
}

fn snapshot_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "psi-dec-gdn-state-unload-load-{}-{name}.state",
        std::process::id()
    ))
}

struct TestRandom(u64);

impl TestRandom {
    fn new(seed: u64) -> Self {
        assert_ne!(seed, 0);
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 as u32
    }

    fn values(&mut self, len: usize) -> Vec<u16> {
        (0..len)
            .map(|_| {
                let value = (self.next_u32() % 20_001) as i32 - 10_000;
                bf16::from_f32(value as f32 / 128.0).to_bits()
            })
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_state(
    state: &GDNRequestStateTable,
    metadata: &GDNMetadataBuffers,
    req_slots: &[u32],
    block_indices: &[usize],
    token_indices: &[u32],
    cu_tokens: &[u32],
    num_spec_tokens: &[u32],
    num_recurrent_requests: usize,
    state_page_ids_by_req: &[Vec<Vec<u32>>],
) -> GDNReplayShape {
    let prepared = state.prepare(
        req_slots,
        block_indices,
        token_indices,
        cu_tokens,
        num_spec_tokens,
        num_recurrent_requests,
        state_page_ids_by_req,
    );
    metadata.update(
        cu_tokens,
        num_recurrent_requests as u32,
        &prepared.src_recurrent_state_slots,
        &prepared.src_conv_state_slots,
        &prepared.flat_recurrent_state_write_slots,
        &prepared.flat_conv_state_write_slots,
        prepared.src_recurrent_state_slots.len() as u32,
        cu_tokens.last().copied().unwrap(),
    )
}

fn core(model_layer_index: usize) -> GDNCore {
    GDNCore {
        model_layer_index,
        hidden_dim: 4,
        num_qk_heads: 1,
        qk_head_dim: 4,
        num_v_heads: 1,
        v_head_dim: 4,
        conv_kernel_size: 3,
        q_scale: 1.0,
        output_gate: GDNOutputGate::Silu,
    }
}
