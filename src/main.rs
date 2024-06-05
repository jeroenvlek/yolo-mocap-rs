use std::error::Error;

use candle_core::{IndexOp, Module};
use clap::Parser;
use hf_hub::api::tokio::Api;
use opencv::prelude::*;
use tokio::runtime;

use crate::camera_calibration::{estimate_camera_matrix, mat_to_matrix3, mat_to_matrix4};
use args::Args;

use crate::pose_estimator::PoseEstimator;

mod args;
mod camera;
mod camera_calibration;
mod key_constants;
mod model;
mod pose_estimator;

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!("Args: {:?}", args);

    let (intrinsic_camera_matrix, extrinsic_camera_matrix) =
        estimate_camera_matrix(args.active_cam, 640, 480)?;

    let intrinsic_gl = mat_to_matrix3::<f64>(&intrinsic_camera_matrix)?;
    let extrinsic_gl = mat_to_matrix4::<f64>(&extrinsic_camera_matrix)?;

    println!("Intrinsic: \n{:?}", intrinsic_gl);
    println!("Extrinsic: \n{:?}", extrinsic_gl);

    // let rt = runtime::Runtime::new()?;
    // let model_path = rt.block_on(download_model(&args))?;
    //
    // let mut pose_estimator = PoseEstimator::new(
    //     args.cpu,
    //     model_path,
    //     args.model_size,
    //     args.active_cam,
    //     args.confidence_threshold,
    //     args.nms_threshold,
    // )?;
    // pose_estimator.run_estimation_loop()?;

    Ok(())
}

async fn download_model(args: &Args) -> anyhow::Result<std::path::PathBuf> {
    let path = match &args.model_path {
        Some(model_path) => std::path::PathBuf::from(model_path),
        None => {
            let api = Api::new()?;
            let api = api.model("lmz/candle-yolo-v8".to_string());
            let model_size = args.model_size;
            api.get(&format!("yolov8{model_size}-pose.safetensors"))
                .await?
        }
    };
    Ok(path)
}
