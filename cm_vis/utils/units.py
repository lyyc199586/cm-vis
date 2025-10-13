# cm_vis/utils/units.py

def mm_to_inch(mm: float) -> float:
    return mm / 25.4

def cm_to_inch(cm: float) -> float:
    return cm / 2.54

def to_inch(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit in ['in', 'inch', 'inches']:
        return value
    elif unit in ['mm', 'millimeter', 'millimeters']:
        return mm_to_inch(value)
    elif unit in ['cm', 'centimeter', 'centimeters']:
        return cm_to_inch(value)
    else:
        raise ValueError(f"Unsupported unit: {unit}")
    
def figsize(width: float, height: float, unit='in') -> tuple[float, float]:
    return (to_inch(width, unit), to_inch(height, unit))