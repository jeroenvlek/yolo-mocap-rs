use opencv::{
    calib3d::{
        CALIB_CB_ADAPTIVE_THRESH, CALIB_CB_NORMALIZE_IMAGE, calibrate_camera,
        find_chessboard_corners,
    },
    core::{Mat, Point3f, Size, TermCriteria_Type, Vector},
    highgui,
    imgproc::{COLOR_BGR2GRAY, corner_sub_pix, cvt_color},
    prelude::*,
    types::VectorOfMat,
    videoio,
};
use opencv::calib3d::draw_chessboard_corners;
use opencv::core::{Point2f, Size_, TermCriteria};
use opencv::prelude::*;
use crate::camera::list_cameras;
use crate::key_constants::ESCAPE_KEY;

const PATTERN_HEIGHT: i32 = 6;

const PATTERN_WIDTH: i32 = 9;

pub fn estimate_camera_matrix(active_cam: usize) -> anyhow::Result<()> {
    let pattern_size = Size::new(PATTERN_WIDTH, PATTERN_HEIGHT);
    let grid_points: Vector<Point3f> =  generate_grid_points(pattern_size);

    let mut object_points = VectorOfMat::new();
    let mut image_points = VectorOfMat::new();

    // Open the default camera
    let cam_list = list_cameras()?;
    let mut cam = videoio::VideoCapture::new(cam_list[active_cam], videoio::CAP_V4L2)?;
    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("Unable to open default camera!");
    }

    // Prepare to display frames
    highgui::named_window("Calibration", highgui::WINDOW_AUTOSIZE)?;

    let mut frame_size = Size::new(640, 480);

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;
        if frame.empty() {
            break;
        }
        frame_size = frame.size()?;

        let mut gray = Mat::default();
        cvt_color(&frame, &mut gray, COLOR_BGR2GRAY, 0)?;

        let mut corners: Vector<Point2f> = Vector::new();
        let found = find_chessboard_corners(
            &gray,
            pattern_size,
            &mut corners,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE,
        )?;

        if found {
            let criteria =
                TermCriteria::new(TermCriteria_Type::COUNT as i32 + TermCriteria_Type::EPS as i32, 30, 0.001)?;
            corner_sub_pix(
                &gray,
                &mut corners,
                Size::new(11, 11),
                Size::new(-1, -1),
                criteria,
            )?;

            // Collect object points and image points
            let mut objp_mat = Mat::from_slice_2d(&[&grid_points])?;
            object_points.push(objp_mat);
            let mut corners_mat = Mat::from_slice_2d(&[&corners])?;
            image_points.push(corners_mat);

            // Draw and display the corners
            draw_chessboard_corners(&mut frame, pattern_size, &corners, found)?;
        }

        highgui::imshow("Calibration", &frame)?;

        // Wait for a key press for 1ms
        let key = highgui::wait_key(1)?;
        if key == ESCAPE_KEY {
            // ESC key to break
            break;
        }
    }

    highgui::destroy_all_windows()?;

    // Perform camera calibration if sufficient points were collected
    if object_points.len() > 0 {
        let mut camera_matrix = Mat::default();
        let mut dist_coeffs = Mat::default();
        let mut rvecs = VectorOfMat::new();
        let mut tvecs = VectorOfMat::new();

        calibrate_camera(
            &object_points,
            &image_points,
            frame_size, // Replace with your actual frame size
            &mut camera_matrix,
            &mut dist_coeffs,
            &mut rvecs,
            &mut tvecs,
            0,
            TermCriteria::default()?,
        )?;

        println!("Camera Matrix:\n{:?}", camera_matrix);
        println!("Distortion Coefficients:\n{:?}", dist_coeffs);
    } else {
        println!("Not enough data for calibration.");
    }

    Ok(())
}

fn generate_grid_points(pattern_size: Size_<i32>) -> Vector<Point3f> {
    let mut grid_points = Vector::with_capacity((pattern_size.height * pattern_size.width) as usize);
    const Z: f32 = 0.0;
    for i in 0..pattern_size.height {
        for j in 0..pattern_size.width {
            grid_points.push(Point3f::new(j as f32, i as f32, Z));
        }
    }
    grid_points
}
