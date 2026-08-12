def calculate_bmi(weight_kg, height_m):
    """
    Calculate Body Mass Index (BMI).
    Formula: BMI = weight (kg) / (height (m)^2)
    """
    if height_m <= 0:
        raise ValueError("Height must be greater than zero.")
    return weight_kg / (height_m ** 2)