
def safe_divide_np(numerator, denominator, zero_zero_fill = 1):
    # Perform element-wise division using np.divide
    result = np.divide(numerator, denominator, where=denominator != 0)
        
    # Handle the case where denominator is zero
    result[denominator == 0] = np.inf
    
    # Handle the case where both numerator and denominator are zero
    result[(numerator == 0) & (denominator == 0)] = zero_zero_fill
    
    return result

# # Example usage
# numerator = np.array([[10, 0, 30, 0, 50], [10, 0, 30, 0, 50]])
# denominator = np.array([[2, 0, 5, 0, 0], [10, 0, 30, 0, 50]])

# result = safe_divide_np(numerator, denominator)
# print(result)
