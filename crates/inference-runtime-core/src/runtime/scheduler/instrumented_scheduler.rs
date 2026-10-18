use std::cmp::max;
use std::collections::BTreeMap;
use std::time::Instant;

use ahash::AHashMap;
use comfy_table::Cell;
use comfy_table::Table;
use comfy_table::presets::UTF8_FULL;
use hdrhistogram::Histogram;

use crate::compute::BatchDevReq;
use crate::compute::BatchDevResp;
use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::compute::SpecStats;
use crate::runtime::CompletionReason;
use crate::runtime::RawComputeSlotSeq;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::scheduler::UserRequest;
use crate::runtime::tasks::AsyncTaskResp;

pub struct InstrumentedScheduler<Sch> {
    periodical: SchedulerStats,
    lifetime: SchedulerStats,

    batch_execution_starts: AHashMap<RawComputeSlotSeq, Instant>,

    scheduler: Sch,
}

impl<Sch> InstrumentedScheduler<Sch> {
    pub fn new(scheduler: Sch, num_spec_tokens: usize) -> Self {
        Self {
            periodical: SchedulerStats::new(num_spec_tokens),
            lifetime: SchedulerStats::new(num_spec_tokens),

            batch_execution_starts: AHashMap::new(),

            scheduler,
        }
    }

    pub fn print_periodical(&mut self) {
        if self.periodical.is_empty() {
            return;
        }

        tracing::info!(
            target: "inference-runtime-core::scheduler",
            phase = "scheduler.stats.periodical",
            scheduler_stats = %format_args!(
                "\nScheduler APIs\n{}\n\nSpec Acceptance\n{}\n\nGPU Execution Latency\n{}",
                self.periodical.api.table(),
                self.periodical.spec.table(),
                self.periodical.execution_table()
            ),
            "scheduler periodical stats"
        );
        self.periodical.reset();
    }

    pub fn print_lifetime(&self) {
        tracing::info!(
            target: "inference-runtime-core::scheduler",
            phase = "scheduler.stats.lifetime",
            scheduler_stats = %format_args!(
                "\nScheduler APIs\n{}\n\nSpec Acceptance\n{}\n\nGPU Execution Latency\n{}",
                self.lifetime.api.table(),
                self.lifetime.spec.table(),
                self.lifetime.execution_table()
            ),
            "scheduler lifetime stats"
        );
    }
}

#[allow(clippy::upper_case_acronyms)]
struct SchedulerAPIStats {
    num_enqueue: u64,
    num_async_task_resp: u64,
    hist_prepare: Histogram<u64>,
    hist_cancel: Histogram<u64>,
    hist_commit: Histogram<u64>,
    hist_batch_execution: Histogram<u64>,
}

impl SchedulerAPIStats {
    fn new() -> Self {
        Self {
            num_enqueue: 0,
            num_async_task_resp: 0,
            hist_prepare: Histogram::<u64>::new(4).unwrap(),
            hist_cancel: Histogram::<u64>::new(4).unwrap(),
            hist_commit: Histogram::<u64>::new(4).unwrap(),
            hist_batch_execution: Histogram::<u64>::new(4).unwrap(),
        }
    }

    fn is_empty(&self) -> bool {
        self.num_enqueue == 0
            && self.num_async_task_resp == 0
            && self.hist_prepare.is_empty()
            && self.hist_cancel.is_empty()
            && self.hist_commit.is_empty()
            && self.hist_batch_execution.is_empty()
    }

    fn reset(&mut self) {
        self.num_enqueue = 0;
        self.num_async_task_resp = 0;
        self.hist_prepare.reset();
        self.hist_cancel.reset();
        self.hist_commit.reset();
        self.hist_batch_execution.reset();
    }

    fn table(&self) -> Table {
        let mut table = Table::new();
        table.load_style(UTF8_FULL.with_rounded_corners());

        let mut header = vec![Cell::new("scheduler API"), Cell::new("count")];
        header.extend(COLUMNS.iter().map(|(name, _)| Cell::new(*name)));
        table.set_header(header);

        for (name, count) in [("enqueue", self.num_enqueue), ("async_task", self.num_async_task_resp)] {
            let mut row = vec![Cell::new(name), Cell::new(count)];
            row.extend(COLUMNS.iter().map(|_| Cell::new("-")));
            table.add_row(row);
        }

        for (name, histogram) in [
            ("prepare", &self.hist_prepare),
            ("cancel", &self.hist_cancel),
            ("commit", &self.hist_commit),
            ("execution", &self.hist_batch_execution),
        ] {
            let mut row = vec![Cell::new(name), Cell::new(histogram.len().to_string())];
            row.extend(COLUMNS.iter().map(|(_, col)| Cell::new(cell(histogram, *col))));
            table.add_row(row);
        }

        table
    }
}

impl SpecStats {
    fn table(&self) -> Table {
        let mut table = Table::new();
        table.load_style(UTF8_FULL.with_rounded_corners());

        let rate = |proposed, accepted| {
            if proposed == 0 {
                "N/A".to_string()
            } else {
                format!("{:.4}", accepted as f64 / proposed as f64)
            }
        };
        let proposed = self.proposed_by_index().iter().sum::<u64>();
        let accepted = self.accepted_by_index().iter().sum::<u64>();

        let mut header = vec![Cell::new("spec stat"), Cell::new("overall")];
        header.extend((0..self.len()).map(|index| Cell::new(format!("index@{index}"))));
        table.set_header(header);

        let mut proposed_row = vec![Cell::new("proposed"), Cell::new(proposed)];
        proposed_row.extend(self.proposed_by_index().iter().map(Cell::new));
        table.add_row(proposed_row);

        let mut accepted_row = vec![Cell::new("accepted"), Cell::new(accepted)];
        accepted_row.extend(self.accepted_by_index().iter().map(Cell::new));
        table.add_row(accepted_row);

        let mut rate_row = vec![Cell::new("rate"), Cell::new(rate(proposed, accepted))];
        rate_row.extend(
            self.proposed_by_index()
                .iter()
                .zip(self.accepted_by_index())
                .map(|(&proposed, &accepted)| Cell::new(rate(proposed, accepted))),
        );
        table.add_row(rate_row);

        table
    }
}

struct SchedulerStats {
    api: SchedulerAPIStats,
    spec: SpecStats,
    execution: BTreeMap<String, Histogram<u64>>,
}

impl SchedulerStats {
    fn new(num_spec_tokens: usize) -> Self {
        Self {
            api: SchedulerAPIStats::new(),
            spec: SpecStats::new(num_spec_tokens),
            execution: BTreeMap::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.api.is_empty() && self.spec.is_empty() && self.execution.values().all(Histogram::is_empty)
    }

    fn reset(&mut self) {
        self.api.reset();
        self.spec.reset();
        self.execution.clear();
    }

    fn execution_table(&self) -> Table {
        let mut table = Table::new();
        table.load_style(UTF8_FULL.with_rounded_corners());
        let mut header = vec![Cell::new("GPU stage"), Cell::new("count")];
        header.extend(COLUMNS.iter().map(|(name, _)| Cell::new(*name)));
        table.set_header(header);
        for (stage, histogram) in &self.execution {
            let mut row = vec![Cell::new(stage), Cell::new(histogram.len())];
            row.extend(COLUMNS.iter().map(|(_, col)| Cell::new(cell(histogram, *col))));
            table.add_row(row);
        }
        table
    }
}

impl<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp, Sch>
    Scheduler<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp> for InstrumentedScheduler<Sch>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
    BatchDeviceReq: BatchDevReq<DeviceReq>,
    BatchDeviceResp: BatchDevResp<DeviceResp>,
    Sch: Scheduler<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp>,
{
    fn enqueue(&mut self, user_req: UserReq) {
        self.scheduler.enqueue(user_req);
        self.periodical.api.num_enqueue += 1;
        self.lifetime.api.num_enqueue += 1;
    }

    fn handle_async_task_resp(&mut self, resp: Box<dyn AsyncTaskResp>) {
        self.scheduler.handle_async_task_resp(resp);
        self.periodical.api.num_async_task_resp += 1;
        self.lifetime.api.num_async_task_resp += 1;
    }

    fn can_flush(&self) -> bool {
        self.scheduler.can_flush()
    }

    fn prepare(&mut self) -> BatchDeviceReq {
        let instant = Instant::now();
        let result = self.scheduler.prepare();
        let batch_execution_start = self.batch_execution_starts.insert(result.seq(), Instant::now());
        assert!(
            batch_execution_start.is_none(),
            "instrumented scheduler requires unique in-flight compute slot sequences"
        );
        let latency = instant.elapsed().as_micros() as u64;
        let latency = max(1, latency);
        let _ = self.periodical.api.hist_prepare.record(latency);
        let _ = self.lifetime.api.hist_prepare.record(latency);
        result
    }

    fn cancel(&mut self, batch_dev_req: BatchDeviceReq) {
        let compute_slot_seq = batch_dev_req.seq();
        let instant = Instant::now();
        self.scheduler.cancel(batch_dev_req);
        self.batch_execution_starts
            .remove(&compute_slot_seq)
            .expect("instrumented scheduler cancellation requires a matching compute slot timestamp");
        let latency = instant.elapsed().as_micros() as u64;
        let latency = max(1, latency);
        let _ = self.periodical.api.hist_cancel.record(latency);
        let _ = self.lifetime.api.hist_cancel.record(latency);
    }

    fn commit(&mut self, batch_dev_resp: BatchDeviceResp) -> Vec<(UserReq, CompletionReason)> {
        let compute_slot_seq = batch_dev_resp.seq();
        let batch_execution = self
            .batch_execution_starts
            .remove(&compute_slot_seq)
            .expect("instrumented scheduler commit requires a matching compute slot timestamp")
            .elapsed();
        let batch_execution = max(1, batch_execution.as_micros() as u64);
        let _ = self.periodical.api.hist_batch_execution.record(batch_execution);
        let _ = self.lifetime.api.hist_batch_execution.record(batch_execution);

        for (stage, elapsed) in batch_dev_resp.execution_timings() {
            for stats in [&mut self.periodical, &mut self.lifetime] {
                let histogram = stats
                    .execution
                    .entry(stage.clone())
                    .or_insert_with(|| Histogram::new(4).unwrap());
                if let Some(elapsed) = elapsed {
                    let _ = histogram.record(elapsed.as_micros() as u64);
                }
            }
        }

        let num_spec_tokens = self.lifetime.spec.len();
        if num_spec_tokens != 0 {
            let delta = batch_dev_resp.spec_stats(num_spec_tokens);
            self.periodical.spec.accumulate(&delta);
            self.lifetime.spec.accumulate(&delta);
        }

        let instant = Instant::now();
        let completed_turns = self.scheduler.commit(batch_dev_resp);
        let latency = instant.elapsed().as_micros() as u64;
        let latency = max(1, latency);
        let _ = self.periodical.api.hist_commit.record(latency);
        let _ = self.lifetime.api.hist_commit.record(latency);
        completed_turns
    }
}

#[derive(Clone, Copy)]
enum Column {
    P0,
    P(f64),
    P100,
    Avg,
}

const COLUMNS: &[(&str, Column)] = &[
    ("p0", Column::P0),
    ("p10", Column::P(0.10)),
    ("p20", Column::P(0.20)),
    ("p30", Column::P(0.30)),
    ("p40", Column::P(0.40)),
    ("p50", Column::P(0.50)),
    ("p60", Column::P(0.60)),
    ("p70", Column::P(0.70)),
    ("p80", Column::P(0.80)),
    ("p90", Column::P(0.90)),
    ("p95", Column::P(0.95)),
    ("p99", Column::P(0.99)),
    ("p999", Column::P(0.999)),
    ("p100", Column::P100),
    ("avg", Column::Avg),
];

fn cell(hist: &Histogram<u64>, kind: Column) -> String {
    if hist.is_empty() {
        return "-".into();
    }

    fn fmt_us(us: u64) -> String {
        if us < 1_000 {
            format!("{us}us")
        } else if us < 1_000_000 {
            format!("{:.3}ms", us as f64 / 1_000.0)
        } else {
            format!("{:.3}s", us as f64 / 1_000_000.0)
        }
    }
    match kind {
        Column::Avg => fmt_us(hist.mean().round() as u64),
        Column::P0 => fmt_us(hist.min()),
        Column::P100 => fmt_us(hist.max()),
        Column::P(q) => fmt_us(hist.value_at_quantile(q)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::MockBatchDevReq;
    use crate::compute::MockBatchDevResp;
    use crate::compute::MockDevReq;
    use crate::compute::MockDevResp;
    use crate::runtime::scheduler::MockScheduler;
    use crate::runtime::scheduler::MockUserRequest;

    #[test]
    fn test_metrics() {
        type TestUserReq = MockUserRequest<MockDevReq, MockDevResp>;
        type TestBatchDeviceReq = MockBatchDevReq<MockDevReq>;
        type TestBatchDeviceResp = MockBatchDevResp<MockDevResp>;

        let mut inner =
            MockScheduler::<TestUserReq, MockDevReq, MockDevResp, TestBatchDeviceReq, TestBatchDeviceResp>::new();
        inner.expect_enqueue().once().return_once(drop);
        inner.expect_can_flush().once().return_const(true);
        inner.expect_prepare().times(2).returning({
            let mut compute_slot_seq = 1;
            move || {
                let batch_dev_req = batch_dev_req(compute_slot_seq);
                compute_slot_seq += 1;
                batch_dev_req
            }
        });
        inner.expect_cancel().once().return_once(drop);
        inner.expect_commit().once().return_once(|response| {
            drop(response);
            Vec::new()
        });

        let mut scheduler = InstrumentedScheduler::new(inner, 0);
        scheduler.enqueue(TestUserReq::new());
        assert!(scheduler.can_flush());
        let batch_dev_req = scheduler.prepare();
        scheduler.cancel(batch_dev_req);
        let batch_dev_req = scheduler.prepare();
        scheduler.commit(batch_dev_resp(batch_dev_req.seq()));

        assert_eq!(scheduler.lifetime.api.num_enqueue, 1);
        assert_eq!(scheduler.lifetime.api.num_async_task_resp, 0);
        assert_eq!(scheduler.lifetime.api.hist_prepare.len(), 2);
        assert_eq!(scheduler.lifetime.api.hist_cancel.len(), 1);
        assert_eq!(scheduler.lifetime.api.hist_commit.len(), 1);
        assert_eq!(scheduler.periodical.api.hist_batch_execution.len(), 1);
        assert_eq!(scheduler.lifetime.api.hist_batch_execution.len(), 1);
        assert!(scheduler.batch_execution_starts.is_empty());
        for stats in [&scheduler.periodical, &scheduler.lifetime] {
            assert_eq!(stats.execution["main"].len(), 1);
            assert_eq!(stats.execution["main"].min(), 1250);
            assert_eq!(stats.execution["dflash2"].len(), 1);
            assert_eq!(stats.execution["dflash2"].min(), 0);
            assert!(stats.execution["mtp"].is_empty());
            let table = stats.execution_table().to_string();
            assert!(table.contains("1.250ms"));
            assert!(table.contains("0us"));
        }
        scheduler.print_periodical();
        assert!(scheduler.periodical.is_empty());
        assert!(scheduler.periodical.execution.is_empty());
        assert_eq!(scheduler.lifetime.execution["main"].len(), 1);
    }

    #[test]
    fn test_spec_stats_table_uses_indexes_as_columns() {
        let mut stats = SpecStats::new(2);
        stats.record_spec_info(2, 1);
        stats.record_spec_info(2, 1);
        stats.record_spec_info(1, 0);

        let table = stats.table().to_string();
        assert!(table.contains("spec stat"));
        assert!(table.contains("overall"));
        assert!(table.contains("index@0"));
        assert!(table.contains("index@1"));
        assert!(table.contains("│ proposed  ┆ 5       ┆ 3       ┆ 2       │"));
        assert!(table.contains("│ accepted  ┆ 2       ┆ 2       ┆ 0       │"));
        assert!(table.contains("│ rate      ┆ 0.4000  ┆ 0.6667  ┆ 0.0000  │"));
    }

    fn batch_dev_req(seq: u64) -> MockBatchDevReq<MockDevReq> {
        let mut batch_dev_req = MockBatchDevReq::new();
        batch_dev_req.expect_seq().return_const(seq);
        batch_dev_req
    }

    fn batch_dev_resp(seq: u64) -> MockBatchDevResp<MockDevResp> {
        let mut batch_dev_resp = MockBatchDevResp::new();
        batch_dev_resp.expect_seq().return_const(seq);
        batch_dev_resp.expect_execution_timings().return_const(vec![
            ("main".into(), Some(std::time::Duration::from_micros(1250))),
            ("dflash2".into(), Some(std::time::Duration::ZERO)),
            ("mtp".into(), None),
        ]);
        batch_dev_resp
    }
}
