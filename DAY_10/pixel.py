def average_pixel_intensity(image_matrix):
    """
    Calculate average pixel intensity of a grayscale image matrix.
    image_matrix: 2D list of pixel values (0–255).
    """
    total_pixels = sum(len(row) for row in image_matrix)
    if total_pixels == 0:
        raise ValueError("Image matrix must contain pixels.")
    total_intensity = sum(sum(row) for row in image_matrix)
    return total_intensity / total_pixels