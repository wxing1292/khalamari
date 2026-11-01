use super::*;

#[tokio::test]
async fn test_append_read_seal() {
    let (mut recording, _path) = AudioRecordingWriter::create().unwrap();
    let mut fast = recording.reader();
    let mut slow = recording.reader();
    recording.append(&[0, 128, 255, 255, 0, 0]).unwrap();
    assert_eq!(fast.read(0..2).unwrap(), [i16::MIN, -1]);
    assert!(slow.read(0..4).is_err());
    recording.append(&[1, 0, 255, 127]).unwrap();
    recording.stop().unwrap();
    assert_eq!(fast.read(2..5).unwrap(), [0, 1, i16::MAX]);
    assert_eq!(fast.available(6).await, 5);
    let end = slow.available(8).await;
    assert_eq!(slow.read(0..end).unwrap(), [i16::MIN, -1, 0, 1, i16::MAX]);
    assert_eq!(slow.read(1..4).unwrap(), [-1, 0, 1]);
}

#[tokio::test]
async fn test_read_cancel() {
    let (mut recording, _path) = AudioRecordingWriter::create().unwrap();
    let mut reader = recording.reader();
    {
        let mut waiting = std::pin::pin!(reader.available(512));
        assert!(futures_util::poll!(&mut waiting).is_pending());
    }
    recording.append(&[12, 0, 34, 0]).unwrap();
    recording.stop().unwrap();
    assert_eq!(reader.available(512).await, 2);
    assert_eq!(reader.read(0..2).unwrap(), [12, 34]);
}
