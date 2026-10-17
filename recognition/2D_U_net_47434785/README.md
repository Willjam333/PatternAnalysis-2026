# Improved 2D U-Net for Prostate Segmentation

1. User Need, Scope & Acceptance Criteria

Medical specialists require precise prostate contouring on MRI images for effective radiotherapy procedures. Automated segmentation trained purely on global dice scores can mask critical boundary failure modes. These failure modes can include contours extending into adjacent organs which can potentially compromise treatment planning.

This project will compare an improved 2D U-Net against a standard 2D U-Net baseline line model. The comparison will identify whether architectural improvement reduces segmentation failure cases sufficiently to justify increases in computational cost.

Acceptance Citeria:

Achieve prostate dice coefficient of at least 0.75 on the test data set.

Compare dice coefficients and Intersection-over-Union between models on the same test data set.

Calculate mean dice coefficient for prostate base, middle and apex slices separately. Report the difference in values to identify potential failure modes for both baseline and improved model.

Document 3-5 segmentation failures, showing ground-truth comparison, quantitative error measurement and anatomical significance

Compare peak GPU VRAM & per-case inference time for the baseline and improved 2D U-Net model.

2. Model Choice & Course Concepts

The project uses the supplied HipMRI keras_slices_data dataset which provides separate training, validation & test folders containing compressed NifTi MRI slices and segmentation masks.

The proposed model is a 2D Improved U-Net [Normal Difficulty], following the architectural principles of Isensee et al. (2018), with standard 2D U-Net (Ronneberger et al., 2015) as the baseline.

Both models will utilise backpropagation, convolutional feature extraction, receptive field and hierarchical representation.

3. Preliminary Feasibility Evidence

An initial data audit confirmed 12,660 MRI slices comprising of 11,460 training, 660 validation and 540 test images (90.5% train, 5.2% validation, 4.3% test) all with corresponding segmentation masks. No patient overlap between the three partitions.

Loading of a small MRI sample using the supplied NifTI-loading approach was successful.

The data set contains six segmentation labels (0-5) and all images besides 60 training images have dimension of 256x128 pixels. The 60 training images have dimensions of 256x144 pixels.

From data inspection label 0 appears correspond to outside the body, label 1 the soft tissue region, label 2 the pelvic bones, label 3 the bladder, label 4 the rectum and label 5 the prostate.

4. Risks, Budget & fallback

Risks: The data set contain 60 images with inconsistent dimensions, requiring standardisation without compromising mask alignment. Both models lack interslice context so improvement in dice score may not translate to between clinical boundary accuracy. Label pixel occupation imbalance may reduce training effectiveness if all pixels are treated equally. Want to prioritise prostate label pixel accuracy.

Computing Budget: Training will use UQ's Rangpur GPU cluster.

The next experiment will be to standardise image dimension and run a short standard U-Net training experiment to verify decreasing loss and measure resource requirements.

If improved U-Net fails to improve boundary accuracy, the standard U-Net will be recommended as the more efficient alternative, supported by the experimental results.