"""
Temporary patch for WFCommons compatibility with modern scipy versions.
This fixes the scipy.stats.trapz issue by aliasing it to scipy.stats.trapezoid.
"""

import scipy.stats

def patch_wfcommons():
    """Apply compatibility patch for WFCommons with modern scipy."""
    
    # Fix 1: Add trapz distribution as alias for trapezoid
    if not hasattr(scipy.stats, 'trapz'):
        scipy.stats.trapz = scipy.stats.trapezoid
        print("✓ Patched scipy.stats.trapz -> scipy.stats.trapezoid")
    
    # Check if there are other missing distributions that might cause issues
    missing_distributions = []
    common_distributions = ['norm', 'uniform', 'expon', 'gamma', 'beta', 'lognorm']
    
    for dist_name in common_distributions:
        if not hasattr(scipy.stats, dist_name):
            missing_distributions.append(dist_name)
    
    if missing_distributions:
        print(f"⚠ Warning: Missing scipy distributions: {missing_distributions}")
    else:
        print("✓ All common scipy distributions are available")
    
    print(f"✓ WFCommons compatibility patch applied (scipy {scipy.__version__})")

if __name__ == "__main__":
    patch_wfcommons()