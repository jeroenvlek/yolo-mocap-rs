use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::object_detection::{non_maximum_suppression, Bbox, KeyPoint};
use opencv::core::{flip, MatTraitConst, Point, Rect, Scalar, Size};
use opencv::imgproc::resize;
use opencv::prelude::*;
use opencv::videoio::VideoCapture;
use opencv::{highgui, imgproc, videoio};

use crate::model::{Multiples, YoloV8Pose};

const MAX_CAM_INDEX: i32 = 10;

const KP_CONNECTIONS: [(usize, usize); 16] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
];

pub struct PoseEstimator {
    device: Device,
    model: YoloV8Pose,
    camera_index_list: Vec<i32>,
    active_cam: usize,
    confidence_threshold: f32,
    near_maximum_surpression_threshold: f32,
}

impl PoseEstimator {
    pub fn new(
        run_on_cpu: bool,
        model_path: PathBuf,
        model_size: char,
        active_cam: usize,
        confidence_threshold: f32,
        near_maximum_surpression_threshold: f32,
    ) -> anyhow::Result<PoseEstimator> {
        let device = candle_examples::device(run_on_cpu)?;
        println!("Using device: {:?}", device);
        let multiples = match model_size {
            'n' => Multiples::n(),
            's' => Multiples::s(),
            'm' => Multiples::m(),
            'l' => Multiples::l(),
            'x' => Multiples::x(),
            _ => panic!("Wrong model size uses"),
        };
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };
        let model = YoloV8Pose::load(vb, multiples, 1, (17, 3))?;
        println!("Model loaded");

        let camera_index_list = list_cameras()?;
        println!("Found camera list {:?}", camera_index_list);

        Ok(PoseEstimator {
            device,
            model,
            camera_index_list,
            active_cam,
            confidence_threshold,
            near_maximum_surpression_threshold,
        })
    }

    pub fn run_estimation_loop(&mut self) -> anyhow::Result<()> {
        loop {
            let mut cam =
                VideoCapture::new(self.camera_index_list[self.active_cam], videoio::CAP_V4L2)?;
            if !VideoCapture::is_opened(&cam)? {
                panic!("Could not open camera!");
            }
            let mut frame = Mat::default();
            cam.read(&mut frame)?;

            let size = frame.size()?;
            println!("Frame size ({}, {})", size.width, size.height);
            let (new_width, new_height) = {
                // Interpolation method
                if size.width < size.height {
                    let w = size.width * 640 / size.height;
                    // Sizes have to be divisible by 32.
                    (w / 32 * 32, 640)
                } else {
                    let h = size.height * 640 / size.width;
                    (640, h / 32 * 32)
                }
            };

            let mut flipped_image = Mat::default();
            flip(&frame, &mut flipped_image, 1)?;
            let mut source_image = Mat::default();
            resize(
                &flipped_image,
                &mut source_image,
                Size::new(new_width, new_height),
                0.0,
                0.0,
                imgproc::INTER_CUBIC,
            )?;

            let image_t = {
                let data = source_image.data_bytes()?;
                Tensor::from_slice(
                    data,
                    (new_height as usize, new_width as usize, 3),
                    &self.device,
                )?
                .permute((2, 0, 1))?
            };
            let image_t = (image_t.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;

            let predictions = self.model.forward(&image_t)?.squeeze(0)?;
            println!("generated predictions {predictions:?}");

            self.annotate_pose(&predictions, &mut flipped_image, new_width, new_height)?;

            if frame.size()?.width > 0 {
                highgui::imshow("window", &flipped_image)?;
            }
            let key = highgui::wait_key(10)?;
            if key > 0 && key != 255 {
                if key >= 48 && key <= 57 {
                    self.active_cam = (key - 48) as usize % self.camera_index_list.len();
                    println!("Switching to camera {}", self.active_cam)
                } else {
                    break;
                }
            }
        }
        Ok(())
    }

    pub fn annotate_pose(
        &self,
        pred: &Tensor,
        img: &mut Mat,
        w: i32,
        h: i32,
    ) -> anyhow::Result<()> {
        let pred = pred.to_device(&Device::Cpu)?;
        let (pred_size, npreds) = pred.dims2()?;
        if pred_size != 17 * 3 + 4 + 1 {
            panic!("unexpected pred-size {pred_size}");
        }
        let mut bboxes = vec![];
        // Extract the bounding boxes for which confidence is above the threshold.
        for index in 0..npreds {
            let pred = Vec::<f32>::try_from(pred.i((.., index))?)?;
            let confidence = pred[4];
            if confidence > self.confidence_threshold {
                let keypoints = (0..17)
                    .map(|i| KeyPoint {
                        x: pred[3 * i + 5],
                        y: pred[3 * i + 6],
                        mask: pred[3 * i + 7],
                    })
                    .collect::<Vec<_>>();
                let bbox = Bbox {
                    xmin: pred[0] - pred[2] / 2.,
                    ymin: pred[1] - pred[3] / 2.,
                    xmax: pred[0] + pred[2] / 2.,
                    ymax: pred[1] + pred[3] / 2.,
                    confidence,
                    data: keypoints,
                };
                bboxes.push(bbox)
            }
        }

        let mut bboxes = vec![bboxes];
        non_maximum_suppression(&mut bboxes, self.near_maximum_surpression_threshold);
        let bboxes = &bboxes[0];

        // Annotate the original image and print boxes information.
        let size = img.size()?;
        let (initial_h, initial_w) = (size.height, size.width);
        let w_ratio = initial_w as f32 / w as f32;
        let h_ratio = initial_h as f32 / h as f32;
        println!("w_ratio: {}, h_ratio: {}", w_ratio, h_ratio);
        for b in bboxes.iter() {
            println!("{b:?}");
            let xmin = (b.xmin * w_ratio) as i32;
            let ymin = (b.ymin * h_ratio) as i32;
            let dx = ((b.xmax - b.xmin) * w_ratio) as i32;
            let dy = ((b.ymax - b.ymin) * h_ratio) as i32;
            if dx >= 0 && dy >= 0 {
                imgproc::rectangle(
                    img,
                    Rect::new(xmin, ymin, dx, dy),
                    Scalar::new(255f64, 0f64, 0f64, 0.0f64),
                    2i32,
                    imgproc::LINE_8,
                    0i32,
                )?;
            }
            for kp in b.data.iter() {
                if kp.mask < 0.6 {
                    continue;
                }
                let x = (kp.x * w_ratio) as i32;
                let y = (kp.y * h_ratio) as i32;
                imgproc::circle(
                    img,
                    Point::new(x, y),
                    2i32,
                    Scalar::new(0f64, 255f64, 0f64, 0f64),
                    2i32,
                    imgproc::LINE_8,
                    0i32,
                )?;
            }

            for &(idx1, idx2) in KP_CONNECTIONS.iter() {
                let kp1 = &b.data[idx1];
                let kp2 = &b.data[idx2];
                if kp1.mask < 0.6 || kp2.mask < 0.6 {
                    continue;
                }
                imgproc::line(
                    img,
                    Point::new((kp1.x * w_ratio) as i32, (kp1.y * h_ratio) as i32),
                    Point::new((kp2.x * w_ratio) as i32, (kp2.y * h_ratio) as i32),
                    Scalar::new(255f64, 255f64, 0f64, 0f64),
                    2i32,
                    imgproc::LINE_8,
                    0i32,
                )?;
            }
        }
        Ok(())
    }
}

fn list_cameras() -> anyhow::Result<Vec<i32>> {
    let mut camera_list = Vec::with_capacity(MAX_CAM_INDEX as usize);
    for index in 0..MAX_CAM_INDEX {
        let cam = VideoCapture::new(index, videoio::CAP_V4L2)?;
        if VideoCapture::is_opened(&cam)? {
            camera_list.push(index);
        }
    }
    camera_list.shrink_to_fit();
    Ok(camera_list)
}
