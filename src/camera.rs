use opencv::hub_prelude::VideoCaptureTraitConst;
use opencv::videoio;
use opencv::videoio::VideoCapture;

const MAX_CAM_INDEX: i32 = 10;

pub fn list_cameras() -> anyhow::Result<Vec<i32>> {
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
