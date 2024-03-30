use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,

    /// Model weights, in safetensors format.
    #[arg(long)]
    pub model_path: Option<String>,

    #[arg(long, default_value_t = 's')]
    pub model_size: char,

    /// Threshold for the model confidence level.
    #[arg(long, default_value_t = 0.25)]
    pub confidence_threshold: f32,

    /// Threshold for non-maximum suppression.
    #[arg(long, default_value_t = 0.45)]
    pub nms_threshold: f32,

    /// The size for the legend, 0 means no legend.
    #[arg(long, default_value_t = 14)]
    pub legend_size: u32,
}
