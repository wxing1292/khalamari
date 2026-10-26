use super::*;

#[test]
fn test_mtp_head_cpu_reference_fixed_random_and_batch() {
    use inference_executor_core::model::qwen::v4_x::weight_layout::resolve_qwen4x_mtp_weight_bindings;

    use crate::model::qwen::v4_x::mtp::Qwen4xMTP;
    use crate::model::qwen::v4_x::mtp::Qwen4xMTPArgs;
    for (seed, dense) in [(0, true), (0x4d54505f52454631, true), (0x4d54505f52454632, false)] {
        let mut fixture = Fixture::new(seed, dense);
        fixture.add_mtp();
        let model = fixture.executor();
        let config = fixture.config.mtp_config().unwrap();
        let device = model.runtime.device();
        let mut state = Qwen4xState::new_with_carry_history(
            device,
            model.runtime.stream().new_on_same_queue(),
            &config,
            model.config,
            4,
        )
        .unwrap();
        let mut store = SafeTensorStore::from_model_dir(&fixture.path).unwrap();
        let bindings = resolve_qwen4x_mtp_weight_bindings(&fixture.config, store.index().tensor_names()).unwrap();
        let mut head = Qwen4xMTP::new(device, &config, &bindings, model.config.max_tokens as u32, &state).unwrap();
        head.load_weights(device, &mut store, &config, &bindings).unwrap();
        let mut head = Replay::new("Qwen4x MTP test", head);
        let h = config.text_config.hidden_size;
        let wide = h * config.text_config.hc_count;
        let previous: Vec<f32> = (0..10 * wide)
            .map(|i| half::bf16::from_f32(((i * 17 % 113) as f32 - 56.0) / 64.0).to_f32())
            .collect();
        let embedding: Vec<f32> = (0..10 * h)
            .map(|i| half::bf16::from_f32(((i * 7 % 127) as f32 - 63.0) / 64.0).to_f32())
            .collect();
        let previous_buffer = Buffer::new_zeroed_elements(device, model.config.max_tokens * wide, Dtype::Bfloat16);
        previous_buffer.write_typed(
            0,
            &previous
                .iter()
                .map(|&x| half::bf16::from_f32(x).to_bits())
                .collect::<Vec<_>>(),
        );
        let embedding_buffer = Buffer::new_zeroed_elements(device, model.config.max_tokens * h, Dtype::Bfloat16);
        embedding_buffer.write_typed(
            0,
            &embedding
                .iter()
                .map(|&x| half::bf16::from_f32(x).to_bits())
                .collect::<Vec<_>>(),
        );
        let output = Buffer::new_zeroed_elements(device, model.config.max_tokens * h, Dtype::Bfloat16);
        let mut requests = Vec::new();
        let mut next_page = 0;
        for (slot, count) in [(0, 3), (1, 7)] {
            let kv: Vec<u32> = (next_page..next_page + state.kv_pages_per_block() as u32).collect();
            next_page += kv.len() as u32;
            let aux: Vec<u32> = (next_page..next_page + state.aux_pages_per_block() as u32).collect();
            next_page += aux.len() as u32;
            requests.push(DeviceRequest::new(
                slot as usize + 1,
                slot,
                QueryTokens::Prefill {
                    epoch: 1,
                    token_index: 0,
                    tokens: vec![Token::new(3); count],
                    window: count,
                },
                DecoderSyncBlocks::new(0, vec![vec![], vec![kv]], vec![vec![], vec![aux]]),
                None,
                vec![],
                SamplingConfig::default(),
            ));
        }
        let request = BatchDeviceRequest {
            seq: 1,
            dev_reqs: requests,
        };
        let batch = Qwen35ModelBatchRequest::from_core_batch(&request, 0, vec![SamplerConfig::default(); 2]);
        let total = head.component().capacity(10);
        state.prepare_lane(
            device,
            model.runtime.stream(),
            model.pages.buffer(),
            request.dev_reqs.iter().map(|request| &request.decoder_sync_blocks),
            batch.microbatch(),
            total,
            0,
            1,
        );
        let hidden_routes = Buffer::from_slice(
            device,
            &(0..model.config.max_tokens as u32)
                .flat_map(|i| [0, i])
                .collect::<Vec<_>>(),
        );
        let write_routes = Buffer::new_zeroed_elements(device, 2, Dtype::Uint32);
        let input = Qwen4xMTPArgs {
            module_index: 0,
            previous_hidden: &previous_buffer,
            hidden_cache: &previous_buffer,
            hidden_routes: &hidden_routes,
            write_routes: &write_routes,
            num_write_rows: 0,
            token_embedding: &embedding_buffer,
            hidden_output: &output,
            pages: model.pages.buffer(),
            state: &state,
        };
        let args = head.component().replay_arguments(&input);
        let (key, _) = head.record(model.runtime.stream(), &input);
        model
            .runtime
            .stream()
            .submit_replay_with_arguments(head.replay(&key), &args)
            .wait();
        for (start, count) in [(0, 3), (3, 7)] {
            let (expected, expected_channels) = reference::mtp(
                &config,
                &bindings,
                &fixture.decoded,
                &previous[start * wide..(start + count) * wide],
                &embedding[start * h..(start + count) * h],
            );
            let read = |buffer: &Buffer, offset, len| {
                buffer
                    .read_typed::<u16>(offset, len)
                    .iter()
                    .map(|&x| half::bf16::from_bits(x).to_f32())
                    .collect::<Vec<_>>()
            };
            assert_close(&read(&output, start * h, count * h), &expected, 0.025);
            assert_close(
                &read(head.component().channels(), start * wide, count * wide),
                &expected_channels,
                0.025,
            );
        }
        state.commit(
            device,
            model.pages.buffer(),
            |_| unreachable!(),
            batch.microbatch(),
            &[3, 7],
        );
        let replacements = [(0, 1usize, 2usize), (1, 6, 1)];
        let mut retained_previous = [previous[..3 * wide].to_vec(), previous[3 * wide..].to_vec()];
        let mut retained_embedding = [embedding[..3 * h].to_vec(), embedding[3 * h..].to_vec()];
        let mut packed_previous = Vec::new();
        let mut packed_embedding = Vec::new();
        let mut rollback_request = request;
        rollback_request.seq = 2;
        let rollback_requests = &mut rollback_request.dev_reqs;
        for (slot, position, count) in replacements {
            for (j, value) in retained_previous[slot][position * wide..(position + count) * wide]
                .iter_mut()
                .enumerate()
            {
                *value = half::bf16::from_f32(*value + ((j % 7) as f32 - 3.0) / 32.0).to_f32();
            }
            for (j, value) in retained_embedding[slot][position * h..(position + count) * h]
                .iter_mut()
                .enumerate()
            {
                *value = half::bf16::from_f32(*value - ((j % 5) as f32 - 2.0) / 16.0).to_f32();
            }
            packed_previous.extend_from_slice(&retained_previous[slot][position * wide..(position + count) * wide]);
            packed_embedding.extend_from_slice(&retained_embedding[slot][position * h..(position + count) * h]);
            rollback_requests[slot].decoder_query_tokens = QueryTokens::Prefill {
                epoch: 1,
                token_index: position,
                tokens: vec![Token::new(3); count],
                window: count,
            };
        }
        let rollback_batch =
            Qwen35ModelBatchRequest::from_core_batch(&rollback_request, 0, vec![SamplerConfig::default(); 2]);
        previous_buffer.write_typed(
            0,
            &packed_previous
                .iter()
                .map(|&x| half::bf16::from_f32(x).to_bits())
                .collect::<Vec<_>>(),
        );
        embedding_buffer.write_typed(
            0,
            &packed_embedding
                .iter()
                .map(|&x| half::bf16::from_f32(x).to_bits())
                .collect::<Vec<_>>(),
        );
        state.prepare_lane(
            device,
            model.runtime.stream(),
            model.pages.buffer(),
            rollback_request
                .dev_reqs
                .iter()
                .map(|request| &request.decoder_sync_blocks),
            rollback_batch.microbatch(),
            head.component().capacity(3),
            0,
            1,
        );
        let input = Qwen4xMTPArgs {
            module_index: 0,
            previous_hidden: &previous_buffer,
            hidden_cache: &previous_buffer,
            hidden_routes: &hidden_routes,
            write_routes: &write_routes,
            num_write_rows: 0,
            token_embedding: &embedding_buffer,
            hidden_output: &output,
            pages: model.pages.buffer(),
            state: &state,
        };
        let args = head.component().replay_arguments(&input);
        let (key, _) = head.record(model.runtime.stream(), &input);
        model
            .runtime
            .stream()
            .submit_replay_with_arguments(head.replay(&key), &args)
            .wait();
        let mut offset = 0;
        for (slot, position, count) in replacements {
            let (expected, _) = reference::mtp(
                &config,
                &bindings,
                &fixture.decoded,
                &retained_previous[slot],
                &retained_embedding[slot],
            );
            let actual: Vec<_> = output
                .read_typed::<u16>(offset * h, count * h)
                .iter()
                .map(|&x| half::bf16::from_bits(x).to_f32())
                .collect();
            assert_close(&actual, &expected[position * h..(position + count) * h], 0.025);
            offset += count;
        }
        state.commit(
            device,
            model.pages.buffer(),
            |_| unreachable!(),
            rollback_batch.microbatch(),
            &[3, 7],
        );
        state.finish_commit();
    }
}

#[test]
fn test_mtp_main_acceptance_and_replacement_against_cpu() {
    for (seed, dense, depth) in [
        (0, true, 1),
        // K=4 includes a three-row tail in a four-row scatter recording.
        (0x4350554d54503331, true, 4),
        (0x4350554d54503332, false, 3),
    ] {
        let mut fixture = Fixture::new(seed, dense);
        fixture.add_mtp();
        let config = Qwen4xExecutorConfig {
            max_requests: 2,
            max_tokens: 32,
            max_tokens_per_request: 32,
            num_cache_pages: 512,
            num_tokens_per_block: 512,
        };
        let mut model =
            init_qwen4x_model_with_mtp(&fixture.path, &fixture.path, NonZeroUsize::new(depth).unwrap(), config)
                .unwrap();
        let mut caches = [Cache::new(0), Cache::new(1)];
        let mut next_page = 0;
        let prompts = [
            vec![3, 7, 11, 13, 17, 19, 23, 29, 31],
            vec![37, 41, 43, 47, 53, 59, 61, 67, 71],
        ];
        let mut histories = [Vec::new(), Vec::new()];
        let prefill = prompts
            .iter()
            .enumerate()
            .map(|(i, tokens)| caches[i].request(&model, 0, tokens, false, &mut next_page))
            .collect();
        let decisions = execute_wave(&mut model, 1, prefill, &mut histories, None, &fixture);
        assert!(decisions.is_empty());
        let consumed = prompts[0].len() - depth;
        let decode = prompts
            .iter()
            .enumerate()
            .map(|(i, tokens)| caches[i].request(&model, consumed, &tokens[consumed..], true, &mut next_page))
            .collect();
        let mut decisions = execute_wave(&mut model, 2, decode, &mut histories, None, &fixture);
        for wave in 0..4 {
            let requests = decisions
                .iter()
                .enumerate()
                .map(|(i, decision)| {
                    let mut request = caches[i].request(
                        &model,
                        histories[i].len(),
                        &[decision.sampled_token],
                        true,
                        &mut next_page,
                    );
                    let QueryTokens::Decode { spec_tokens, .. } = &mut request.decoder_query_tokens else {
                        unreachable!()
                    };
                    *spec_tokens = decision.spec_tokens.iter().copied().map(Token::new).collect();
                    request
                })
                .collect();
            let accept = [wave % (depth + 1), depth - wave % (depth + 1)];
            decisions = execute_wave(
                &mut model,
                3 + wave as u64,
                requests,
                &mut histories,
                Some(accept),
                &fixture,
            );
            assert!(decisions.iter().all(|decision| decision.spec_tokens.len() == depth));
        }
        let mut mixed = Vec::new();
        let mut decode = caches[0].request(
            &model,
            histories[0].len(),
            &[decisions[0].sampled_token],
            true,
            &mut next_page,
        );
        let QueryTokens::Decode { spec_tokens, .. } = &mut decode.decoder_query_tokens else {
            unreachable!()
        };
        *spec_tokens = decisions[0].spec_tokens.iter().copied().map(Token::new).collect();
        mixed.push(decode);
        let known: Vec<_> = std::iter::once(decisions[1].sampled_token)
            .chain((0..depth + 3).map(|i| 79 + i as u32))
            .collect();
        mixed.push(caches[1].request(&model, histories[1].len(), &known, false, &mut next_page));
        let decisions = execute_wave(&mut model, 7, mixed, &mut histories, None, &fixture);
        assert_eq!(decisions.len(), 1);
        model.clear_replay_cache();
        let snapshot = fixture.path.join("mtp-state");
        let plan = if depth == 1 {
            ExecutorHibernationPlan::All
        } else {
            ExecutorHibernationPlan::selected(std::iter::once(0..1).collect(), std::iter::once(0..next_page).collect())
        };
        model.unload_state(&snapshot, &plan).unwrap();
        model.unload_weights();
        model.load_weights().unwrap();
        model.load_state(&snapshot, &plan).unwrap();
        let mut request = caches[0].request(
            &model,
            histories[0].len(),
            &[decisions[0].sampled_token],
            true,
            &mut next_page,
        );
        let QueryTokens::Decode { spec_tokens, .. } = &mut request.decoder_query_tokens else {
            unreachable!()
        };
        *spec_tokens = decisions[0].spec_tokens.iter().copied().map(Token::new).collect();
        execute_wave(&mut model, 8, vec![request], &mut histories, None, &fixture);
        model.reset_req_slots(&[1]);
        model.clear_replay_cache();
    }
}

fn execute_wave(
    model: &mut Qwen4xExecutor,
    seq: u64,
    requests: Vec<DeviceRequest>,
    histories: &mut [Vec<u32>; 2],
    accept: Option<[usize; 2]>,
    fixture: &Fixture,
) -> Vec<Qwen35DecodeDecision> {
    let mut request = BatchDeviceRequest {
        seq,
        dev_reqs: requests,
    };
    let batch = model.prepare_batch(&mut request);
    let micro = batch.microbatch();
    let mut recorder = model.begin_ops_recording(&batch);
    let hidden = model.embed_main(&mut recorder, &batch);
    let hidden = model.forward_main(&mut recorder, &batch, hidden);
    model.unembed_main(&mut recorder, &batch, &hidden);
    model.sample_main(&mut recorder, &batch, &());
    let start = Instant::now();
    let submission = model.submit_main(&recorder);
    submission.wait();
    let gpu_durations = submission.gpu_timestamp_durations();
    let mut output = model.read_main(&recorder, &batch, start.elapsed(), gpu_durations.as_deref());
    for (i, &slot) in micro.req_slots().iter().enumerate() {
        let begin = micro.cu_tokens()[i] as usize;
        let end = micro.cu_tokens()[i + 1] as usize;
        let input = &micro.flat_token_ids()[begin..end];
        let position = micro.token_indices()[i] as usize;
        assert_eq!(histories[slot as usize].len(), position);
        let mut full = histories[slot as usize].clone();
        full.extend(input.iter().map(|&t| t as u32));
        let expected = reference::forward(&fixture.config, &model.bindings, &fixture.decoded, &full);
        let h = fixture.config.text_config.hidden_size;
        let actual: Vec<_> = hidden
            .read_typed::<u16>(begin * h, (end - begin) * h)
            .iter()
            .map(|&x| half::bf16::from_bits(x).to_f32())
            .collect();
        assert_close(&actual, &expected[position * h..], 0.035);
        let accepted = if micro.is_decode_req(i) {
            let row = (0..i).filter(|&j| micro.is_decode_req(j)).count();
            if let Some(accept) = accept {
                let n = accept[slot as usize];
                let spec = micro.num_spec_tokens(i) as usize;
                output.decisions[row].validated_tokens = input[input.len() - spec..input.len() - spec + n]
                    .iter()
                    .map(|&t| t as u32)
                    .collect();
                output.decisions[row].validated_probs = vec![1.0 / 256.0; n];
            }
            input.len() - micro.num_spec_tokens(i) as usize + output.decisions[row].validated_tokens.len()
        } else {
            input.len()
        };
        histories[slot as usize].extend(input[..accepted].iter().map(|&t| t as u32));
    }
    assert!(model.run_spec(&batch, &output));
    let spec = model.embed_spec(&mut recorder, &batch, &hidden, &output);
    let spec = model.forward_spec(&mut recorder, &batch, spec);
    model.unembed_spec(&mut recorder, &batch, &spec);
    model.sample_spec(&mut recorder, &batch, &());
    let start = Instant::now();
    let submission = model.submit_spec(&recorder);
    submission.wait();
    let final_batch = model.mtp.as_ref().unwrap().final_batch.as_ref().unwrap().clone();
    let gpu_durations = submission.gpu_timestamp_durations();
    let output = model.read_spec(&recorder, &batch, output, start.elapsed(), gpu_durations.as_deref());
    let mtp = model.mtp.as_ref().unwrap();
    let h = fixture.config.text_config.hidden_size;
    let wide = h * fixture.config.text_config.hc_count;
    let embedding = &fixture.decoded[&model.bindings.embed.weight];
    let mut decision_index = 0;
    for (i, &slot) in micro.req_slots().iter().enumerate() {
        // Recompute every logical head from position zero, without cache slots or hidden routes.
        let tokens = &histories[slot as usize];
        let mut full = tokens.clone();
        if micro.is_decode_req(i) {
            let decision = &output.decisions[decision_index];
            full.push(decision.sampled_token);
            full.extend(decision.spec_tokens.iter().take(mtp.num_spec_tokens - 1));
            decision_index += 1;
        } else {
            full.extend(
                (1..=mtp.num_spec_tokens).map(|lane| *micro.token_ids_for_lane(i, lane).last().unwrap() as u32),
            );
        }
        let (_, mut channels) =
            reference::forward_with_channels(&fixture.config, &model.bindings, &fixture.decoded, tokens);
        let mut mixed = Vec::new();
        for module in 0..mtp.num_spec_tokens {
            let shifted_embedding = full[module + 1..module + 1 + tokens.len()]
                .iter()
                .flat_map(|&token| embedding[token as usize * h..(token as usize + 1) * h].iter().copied())
                .collect::<Vec<_>>();
            (mixed, channels) = reference::mtp(
                &mtp.config,
                &mtp.bindings,
                &fixture.decoded,
                &channels,
                &shifted_embedding,
            );
        }
        let begin = final_batch.cu_tokens()[i] as usize;
        let count = (final_batch.cu_tokens()[i + 1] - final_batch.cu_tokens()[i]) as usize;
        let position = final_batch.token_indices()[i] as usize;
        let read = |buffer: &Buffer, width| {
            buffer
                .read_typed::<u16>(begin * width, count * width)
                .iter()
                .map(|&value| half::bf16::from_bits(value).to_f32())
                .collect::<Vec<_>>()
        };
        assert_close(&read(&model.hidden_output, h), &mixed[position * h..], 0.035);
        assert_close(
            &read(mtp.body.component().channels(), wide),
            &channels[position * wide..],
            0.035,
        );
    }
    let decisions = output.decisions.clone();
    model.commit_batch(request, output);
    decisions
}
