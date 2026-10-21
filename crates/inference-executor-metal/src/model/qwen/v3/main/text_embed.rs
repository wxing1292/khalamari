use std::rc::Rc;

use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;

use crate::def::layer::ReplayLayer;
use crate::model::embedding::Embed;
use crate::model::embedding::EmbedInput;
use crate::replay::ReplayComponent;

const QWEN3_MAIN_TEXT_EMBED_NUM_ACTIVE_TOKENS: ReplayParameterKey =
    ReplayParameterKey::new("qwen3.main_text_embed.num_active_tokens");

pub struct Qwen3MainTextEmbed {
    embed: Option<Rc<Embed>>,
}

#[derive(Clone, Copy)]
pub struct Qwen3MainTextEmbedArgs<'a> {
    pub num_tokens: u32,
    pub token_ids: &'a Buffer,
    pub hidden_output: &'a Buffer,
}

impl Qwen3MainTextEmbed {
    pub fn new(embed: Rc<Embed>) -> Self {
        Self { embed: Some(embed) }
    }

    pub fn load_weights(&mut self, embed: Rc<Embed>) {
        assert!(self.embed.is_none(), "qwen3 Main embed weights are already loaded");
        self.embed = Some(embed);
    }

    pub fn unload_weights(&mut self) -> Rc<Embed> {
        self.embed.take().expect("qwen3 Main embed weights are not loaded")
    }

    fn embed(&self) -> &Embed {
        self.embed
            .as_deref()
            .expect("qwen3 Main embed weights must be loaded before execution")
    }

    pub fn record<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        args: Qwen3MainTextEmbedArgs<'a>,
    ) -> &'a Buffer {
        if let ReplayU32::Fixed(value) = num_active_tokens {
            assert_eq!(value, args.num_tokens);
            assert_eq!(value, num_total_tokens);
        } else {
            assert_eq!(args.num_tokens, num_total_tokens);
        }
        <Embed as ReplayLayer>::record(
            self.embed(),
            recorder,
            EmbedInput {
                num_total_tokens,
                num_active_tokens,
                token_ids: args.token_ids,
                output_hidden: args.hidden_output,
            },
        )
    }

    pub fn prepare_replay(&self, num_active_tokens: u32) -> (Qwen3MainTextEmbedReplayKey, ReplayArguments) {
        assert!(
            num_active_tokens > 0,
            "qwen3 MainTextEmbed replay requires active tokens"
        );
        let key = Qwen3MainTextEmbedReplayKey::new(num_active_tokens);
        let arguments = ReplayArguments::new().with_u32(QWEN3_MAIN_TEXT_EMBED_NUM_ACTIVE_TOKENS, num_active_tokens);
        (key, arguments)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3MainTextEmbedReplayKey {
    num_total_tokens: u32,
}

impl Qwen3MainTextEmbedReplayKey {
    pub fn new(num_total_tokens: u32) -> Self {
        Self { num_total_tokens }
    }
}

impl ReplayComponent for Qwen3MainTextEmbed {
    type Key = Qwen3MainTextEmbedReplayKey;
    type Input<'a> = Qwen3MainTextEmbedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Qwen3MainTextEmbedReplayKey::new(input.num_tokens)
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let key = self.replay_key(input);
        Qwen3MainTextEmbed::record(
            self,
            recorder,
            key.num_total_tokens,
            ReplayU32::Parameter(QWEN3_MAIN_TEXT_EMBED_NUM_ACTIVE_TOKENS),
            *input,
        );
    }
}
