use std::sync::OnceLock;

static QWEN35_STATE_TRACE: OnceLock<bool> = OnceLock::new();
static GDN_STATE_TRACE: OnceLock<bool> = OnceLock::new();
static QWEN35_CHECK_FINITE: OnceLock<bool> = OnceLock::new();

pub fn qwen35_check_finite() -> bool {
    *QWEN35_CHECK_FINITE.get_or_init(|| {
        let enabled = environment_flag("KHALAMARI_QWEN35_CHECK_FINITE");
        if enabled {
            eprintln!("Qwen3.5 numerical checkpoints enabled: KHALAMARI_QWEN35_CHECK_FINITE=1");
        }
        enabled
    })
}

pub fn qwen35_state(message: impl FnOnce() -> String) {
    if trace_enabled(&QWEN35_STATE_TRACE, "KHALAMARI_QWEN35_STATE_TRACE") {
        eprintln!("qwen35_state {}", message());
    }
}

pub fn gdn_state(message: impl FnOnce() -> String) {
    if trace_enabled(&GDN_STATE_TRACE, "KHALAMARI_GDN_STATE_TRACE") {
        eprintln!("gdn_state {}", message());
    }
}

fn trace_enabled(cache: &OnceLock<bool>, name: &str) -> bool {
    *cache.get_or_init(|| environment_flag(name))
}

fn environment_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}
