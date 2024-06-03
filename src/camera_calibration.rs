use crate::camera::list_cameras;
use crate::key_constants::ESCAPE_KEY;
use opencv::calib3d::{draw_chessboard_corners, rodrigues};
use opencv::core::{Point2f, Size_, TermCriteria, CV_32F};
use opencv::prelude::*;
use opencv::{
    calib3d::{
        calibrate_camera, find_chessboard_corners, CALIB_CB_ADAPTIVE_THRESH,
        CALIB_CB_NORMALIZE_IMAGE,
    },
    core::{Mat, Point3f, Size, TermCriteria_Type, Vector},
    highgui,
    imgproc::{corner_sub_pix, cvt_color, COLOR_BGR2GRAY},
    prelude::*,
    types::VectorOfMat,
    videoio,
};

const PATTERN_HEIGHT: i32 = 5;
const PATTERN_WIDTH: i32 = 5;
const NECESSARY_FRAMES: i32 = 10;

const WAIT_CORRECT_FRAME: i32 = 2000;
const WAIT_INCORRECT_FRAME: i32 = 1;

pub fn estimate_camera_matrix(
    active_cam: usize,
    frame_width: i32,
    frame_height: i32,
) -> anyhow::Result<(Mat, Mat)> {
    let pattern_size = Size::new(PATTERN_WIDTH, PATTERN_HEIGHT);
    let grid_points: Vector<Point3f> = generate_grid_points(pattern_size);

    let mut object_points = VectorOfMat::new();
    let mut image_points = VectorOfMat::new();

    // Open the default camera
    let cam_list = list_cameras()?;
    let mut cam = videoio::VideoCapture::new(cam_list[active_cam], videoio::CAP_V4L2)?;
    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("Unable to open default camera!");
    }

    highgui::named_window("Calibration", highgui::WINDOW_AUTOSIZE)?;
    let mut frame_size = Size::new(frame_width, frame_height);
    let mut correct_frames = 0;
    let mut pause_time_ms = WAIT_INCORRECT_FRAME;
    while correct_frames < NECESSARY_FRAMES {
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
            let criteria = TermCriteria::new(
                TermCriteria_Type::COUNT as i32 + TermCriteria_Type::EPS as i32,
                30,
                0.001,
            )?;
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

            correct_frames += 1;
            println!("Found pattern! Correct frames: {}", correct_frames);

            // let's wait a bit longer to reorient the pattern
            pause_time_ms = WAIT_CORRECT_FRAME;
        } else {
            println!("No pattern in the frame. Correct frames {}", correct_frames);
            pause_time_ms = WAIT_INCORRECT_FRAME;
        }

        highgui::imshow("Calibration", &frame)?;

        let key = highgui::wait_key(pause_time_ms)?;
        if key == ESCAPE_KEY {
            break;
        }
    }

    highgui::destroy_all_windows()?;

    if correct_frames < NECESSARY_FRAMES {
        return Err(anyhow::anyhow!("Not enough data for calibration."));
    }

    let mut intrinsic_camera_matrix = Mat::default();
    let mut distortion_coeffs = Mat::default();
    let mut rotation_vecs = VectorOfMat::new();
    let mut translation_vecs = VectorOfMat::new();

    calibrate_camera(
        &object_points,
        &image_points,
        frame_size,
        &mut intrinsic_camera_matrix,
        &mut distortion_coeffs,
        &mut rotation_vecs,
        &mut translation_vecs,
        0,
        TermCriteria::default()?,
    )?;

    let extrinsic_camera_matrix =
        extrinsic_matrix(&rotation_vecs.get(0)?, &translation_vecs.get(0)?)?;

    println!("Intrinsic camera matrix:\n{:?}", intrinsic_camera_matrix);
    println!("Extrinsic camera matrix: \n{:?}", extrinsic_camera_matrix);
    println!("Distortion Coefficients:\n{:?}", distortion_coeffs);

    Ok((intrinsic_camera_matrix, extrinsic_camera_matrix))
}

fn generate_grid_points(pattern_size: Size_<i32>) -> Vector<Point3f> {
    let mut grid_points =
        Vector::with_capacity((pattern_size.height * pattern_size.width) as usize);
    const Z: f32 = 0.0;
    for i in 0..pattern_size.height {
        for j in 0..pattern_size.width {
            grid_points.push(Point3f::new(j as f32, i as f32, Z));
        }
    }
    grid_points
}

/// Generate extrinsic matrix from rotation and translation vectors
fn extrinsic_matrix(rotation_vec: &Mat, translation_vec: &Mat) -> opencv::Result<Mat> {
    // Convert the rotation vector to a rotation matrix using Rodrigues formula
    let mut rotation_matrix = Mat::default();
    rodrigues(rotation_vec, &mut rotation_matrix, &mut Mat::default())?;

    // Create an empty 4x4 matrix for the extrinsic matrix
    let mut extrinsic_matrix = Mat::zeros(4, 4, CV_32F)?.to_mat()?;

    // Set the top-left 3x3 part to the rotation matrix
    {
        let mut roi = extrinsic_matrix.roi_mut(opencv::core::Rect::new(0, 0, 3, 3))?;
        rotation_matrix.copy_to(&mut roi)?;
    }

    // Set the top-right 3x1 part to the translation vector
    {
        let mut roi = extrinsic_matrix.roi_mut(opencv::core::Rect::new(3, 0, 1, 3))?;
        translation_vec.copy_to(&mut roi)?;
    }

    // Set the bottom-right element to 1
    *extrinsic_matrix.at_2d_mut(3, 3)? = 1.0;

    Ok(extrinsic_matrix)
}
