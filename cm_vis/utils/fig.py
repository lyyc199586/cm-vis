# cm_vis/utils/units.py
from matplotlib.patches import Rectangle

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

def lock_canvas(fig, show_frame=False, color='blue'):
    fig.patches.append(
        Rectangle((0, 0), 1, 1,
                  transform=fig.transFigure,
                  edgecolor=color if show_frame else 'none',
                  facecolor='none',
                  linewidth=2 if show_frame else 0)
    )