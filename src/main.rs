use opencv::highgui;
use opencv::prelude::*;
use opencv::videoio;
use opencv::videoio::VideoCapture;

const MAX_CAM_INDEX: i32 = 10;

fn main() -> anyhow::Result<()> {
    let camera_index_list = list_cameras()?;
    if camera_index_list.is_empty() {
        panic!("Could not find any camera!");
    }

    println!("Found {} cameras", camera_index_list.len());

    let mut active_cam = 2;
    loop {
        let mut cam = VideoCapture::new(camera_index_list[active_cam], videoio::CAP_ANY)?;
        if !VideoCapture::is_opened(&cam)? {
            panic!("Could not open camera!");
        }
        let mut frame = Mat::default();
        cam.read(&mut frame)?;
        if frame.size()?.width > 0 {
            highgui::imshow("window", &frame)?;
        }
        let key = highgui::wait_key(10)?;
        if key > 0 && key != 255 {
            if key >= 48 && key <= 47 {
                active_cam = (key - 48) as usize % camera_index_list.len();
                println!("Switching to camera {}", active_cam)
            } else {
                break;
            }
        }
    }
    Ok(())
}

fn list_cameras() -> anyhow::Result<Vec<i32>> {
    let mut camera_list = Vec::with_capacity(MAX_CAM_INDEX as usize);
    for index in 0..MAX_CAM_INDEX {
        let cam = VideoCapture::new(index, videoio::CAP_ANY)?;
        if VideoCapture::is_opened(&cam)? {
            camera_list.push(index);
        }
    }
    camera_list.shrink_to_fit();
    Ok(camera_list)
}
