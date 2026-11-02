use super::*;
use crate::recording::AudioRecordingWriter;

#[tokio::test]
async fn test_run_pcm_ranges() {
    let start = 160_000;
    let end = start + MAX_APPEND_SAMPLES + 17;
    let tail = end + 16_000;
    let audio: Vec<_> = (0..tail + 32)
        .flat_map(|sample| (sample as i16).to_le_bytes())
        .collect();
    let (mut writer, _path) = AudioRecordingWriter::create().unwrap();
    writer.append(&audio).unwrap();
    let reader = writer.reader();
    writer.stop().unwrap();
    let (input, commands) = mpsc::channel(1);
    let (requests, mut received) = mpsc::channel::<AudioStreamReq>(1);
    let (output, responses) = mpsc::channel(1);
    let sink = futures_util::sink::unfold(requests, |requests, request| {
        async move {
            requests.send(request).await.unwrap();
            Ok(requests)
        }
    });
    let producer = async {
        input.send(Command::Append(reader.clone(), start..end)).await.unwrap();
        input
            .send(Command::Send(AudioStreamReq {
                command: Some(AudioCommand::ReleaseBeforeSample(end)),
            }))
            .await
            .unwrap();
        input.send(Command::Append(reader, tail..tail + 32)).await.unwrap();
        let (reply, response) = oneshot::channel();
        input.send(Command::Finish(reply)).await.unwrap();
        assert!(response.await.unwrap().unwrap().finished);
    };
    let server = async {
        // Each network message obeys the service limit, even when one log range is longer.
        for range in [start..start + MAX_APPEND_SAMPLES, start + MAX_APPEND_SAMPLES..end] {
            let Some(AudioCommand::Append(append)) = received.recv().await.unwrap().command else {
                panic!("PCM must precede its release");
            };
            assert_eq!(append.start_sample, range.start);
            assert_eq!(
                append.pcm_s16le,
                audio[range.start as usize * 2..range.end as usize * 2]
            );
        }
        assert!(matches!(
            received.recv().await.unwrap().command,
            Some(AudioCommand::ReleaseBeforeSample(before)) if before == end
        ));
        let Some(AudioCommand::Append(append)) = received.recv().await.unwrap().command else {
            panic!("the next range follows its release");
        };
        assert_eq!(append.start_sample, tail);
        assert_eq!(append.pcm_s16le, audio[tail as usize * 2..]);
        assert!(matches!(
            received.recv().await.unwrap().command,
            Some(AudioCommand::Finish(_))
        ));
        output
            .send(Ok(AudioStreamResp {
                request_id: 0,
                result: None,
                stats: Some(Default::default()),
                finished: true,
            }))
            .await
            .unwrap();
    };
    tokio::time::timeout(Duration::from_secs(10), async {
        tokio::join!(
            run(
                sink,
                ReceiverStream::new(responses),
                commands,
                Arc::new(Mutex::new(AudioTransportStats::default())),
                Duration::from_secs(5),
            ),
            producer,
            server,
        );
    })
    .await
    .unwrap();
}
