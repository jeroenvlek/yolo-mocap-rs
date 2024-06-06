use std::ops::{Add, Div};
use candle_core::quantized::k_quants::matmul;
use crate::camera::list_cameras;
use crate::key_constants::ESCAPE_KEY;
use opencv::calib3d::{draw_chessboard_corners, rodrigues};
use opencv::core::{flip, Point2f, Size_, TermCriteria, CV_32F, CV_64F, SVD, add, divide, multiply, mul_mat_mat, gemm};
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
use opencv::gapi::mul;
use vecmath::{Matrix3, Matrix4};

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
        let mut flipped_image = Mat::default();
        flip(&frame, &mut flipped_image, 1)?;
        frame = flipped_image;

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
            let objp_mat = Mat::from_slice_2d(&[&grid_points])?;
            object_points.push(objp_mat);
            let corners_mat = Mat::from_slice_2d(&[&corners])?;
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
        compose_extrinsic_matrix(&rotation_vecs.get(0)?, &translation_vecs.get(0)?)?;

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

/// Convert rotation vectors to rotation matrices
fn rotation_vecs_to_matrices(rotation_vecs: &VectorOfMat) -> anyhow::Result<Vector<Mat>> {
    let mut rotation_matrices = Vector::new();
    for i in 0..rotation_vecs.len() {
        let rotation_vec = rotation_vecs.get(i)?;
        let mut rotation_matrix = Mat::default();
        rodrigues(&rotation_vec, &mut rotation_matrix, &mut Mat::default())?;
        rotation_matrices.push(rotation_matrix);
    }
    Ok(rotation_matrices)
}

/// Average rotation matrices using Singular Value Decomposition (SVD)
fn average_rotation_matrices(rotation_matrices: &Vector<Mat>) -> anyhow::Result<Mat> {
    let size = rotation_matrices.get(0)?.size()?;
    let mut avg_rotation_matrix = Mat::zeros(size.height, size.width, CV_64F)?;
    for rotation_matrix in rotation_matrices {
        avg_rotation_matrix = (avg_rotation_matrix + rotation_matrix).into_result()?;
    }

    avg_rotation_matrix = (avg_rotation_matrix / (rotation_matrices.len() as f64)).into_result()?;

    // Use SVD to ensure the resulting matrix is a proper rotation matrix
    let mut w = Mat::default();
    let mut u = Mat::default();
    let mut vt = Mat::default();
    SVD::compute_ext(&avg_rotation_matrix.to_mat()?, &mut w, &mut u, &mut vt, 0)?;

    let nearest_rotation_matrix = (u * vt).into_result()?.to_mat()?;

    Ok(nearest_rotation_matrix)
}

// Convert the averaged rotation matrix back to a rotation vector
fn rotation_matrix_to_vector(rotation_matrix: &Mat) -> anyhow::Result<Mat> {
    let mut rotation_vec = Mat::default();
    rodrigues(&rotation_matrix, &mut rotation_vec, &mut Mat::default())?;
    Ok(rotation_vec)
}

// Average the translation vectors
fn average_translation_vectors(translation_vecs: &VectorOfMat) -> anyhow::Result<Mat> {
    let size = translation_vecs.get(0)?.size()?;
    let mut avg_translation = Mat::zeros(size.height, size.width, CV_64F)?;
    for translation_vec in translation_vecs.iter() {
        avg_translation = (avg_translation + translation_vec).into_result()?;
    }
    avg_translation = (avg_translation / (translation_vecs.len() as f64)).into_result()?;
    Ok(avg_translation.to_mat()?)
}

fn compute_extrinsic_matrix(rotation_vecs: &VectorOfMat, translation_vecs: &VectorOfMat) -> anyhow::Result<Mat> {
    // Convert rotation vectors to matrices
    let rotation_matrices = rotation_vecs_to_matrices(rotation_vecs)?;

    // Average the rotation matrices
    let avg_rotation_matrix = average_rotation_matrices(&rotation_matrices)?;

    // Convert the averaged rotation matrix back to a rotation vector
    let avg_rotation_vec = rotation_matrix_to_vector(&avg_rotation_matrix)?;

    // Average the translation vectors
    let avg_translation_vec = average_translation_vectors(translation_vecs)?;

    // Compose the extrinsic matrix
    compose_extrinsic_matrix(&avg_rotation_vec, &avg_translation_vec)
}

fn compose_extrinsic_matrix(rotation_vec: &Mat, translation_vec: &Mat) -> anyhow::Result<Mat> {
    // Convert the rotation vector to a rotation matrix using Rodrigues formula
    let mut rotation_matrix = Mat::default();
    rodrigues(rotation_vec, &mut rotation_matrix, &mut Mat::default())?;

    // Create an empty 4x4 matrix for the extrinsic matrix
    let mut extrinsic_matrix = Mat::zeros(4, 4, CV_64F)?.to_mat()?;

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



pub fn mat_to_matrix4<T>(mat: &Mat) -> opencv::Result<Matrix4<T>>
where
    T: DataType + Default + Copy,
{
    assert_eq!(mat.size()?.width, 4);
    assert_eq!(mat.size()?.height, 4);

    let mut data = [[T::default(); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            data[i][j] = *mat.at_2d::<T>(i as i32, j as i32)?;
        }
    }

    Ok(data)
}

pub fn mat_to_matrix3<T>(mat: &Mat) -> opencv::Result<Matrix3<T>>
where
    T: DataType + Default + Copy,
{
    assert_eq!(mat.size()?.width, 3);
    assert_eq!(mat.size()?.height, 3);

    let mut data = [[T::default(); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            data[i][j] = *mat.at_2d::<T>(i as i32, j as i32)?;
        }
    }

    Ok(data)
}
