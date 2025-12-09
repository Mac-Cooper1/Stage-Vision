"""
Test script for choose_gemini_image_config helper function.

Run with: python test_image_config.py

This is a standalone test that doesn't require external dependencies.
"""

from typing import Tuple

# Gemini 3 Pro Image Preview aspect ratio/size table (excluding 21:9)
GEMINI_IMAGE_CONFIGS = {
    "1:1": {
        "1K": (1024, 1024),
        "2K": (2048, 2048),
        "4K": (4096, 4096),
    },
    "2:3": {
        "1K": (848, 1264),
        "2K": (1696, 2528),
        "4K": (3392, 5056),
    },
    "3:2": {
        "1K": (1264, 848),
        "2K": (2528, 1696),
        "4K": (5056, 3392),
    },
    "3:4": {
        "1K": (896, 1200),
        "2K": (1792, 2400),
        "4K": (3584, 4800),
    },
    "4:3": {
        "1K": (1200, 896),
        "2K": (2400, 1792),
        "4K": (4800, 3584),
    },
    "4:5": {
        "1K": (928, 1152),
        "2K": (1856, 2304),
        "4K": (3712, 4608),
    },
    "5:4": {
        "1K": (1152, 928),
        "2K": (2304, 1856),
        "4K": (4608, 3712),
    },
    "9:16": {
        "1K": (768, 1376),
        "2K": (1536, 2752),
        "4K": (3072, 5504),
    },
    "16:9": {
        "1K": (1376, 768),
        "2K": (2752, 1536),
        "4K": (5504, 3072),
    },
}


def choose_gemini_image_config(width: int, height: int) -> Tuple[str, str]:
    """
    Given the input image dimensions, return (aspect_ratio_str, image_size_str)
    for gemini-3-pro-image-preview that best matches the original.
    """
    input_ar = width / height
    long_input = max(width, height)

    best_score = float('inf')
    best_config = ("16:9", "2K")

    for aspect_ratio_str, sizes in GEMINI_IMAGE_CONFIGS.items():
        for size_str, (w, h) in sizes.items():
            candidate_ar = w / h
            ar_diff = abs(candidate_ar - input_ar)

            long_candidate = max(w, h)
            size_diff = abs(long_candidate - long_input) / max(long_input, 1)

            score = ar_diff * 2.0 + size_diff

            if size_str == "2K":
                score -= 0.001

            if score < best_score:
                best_score = score
                best_config = (aspect_ratio_str, size_str)

    return best_config


def test_choose_gemini_image_config():
    """Test the aspect ratio and size selection logic."""

    test_cases = [
        # (width, height, description)
        (3024, 4032, "Vertical iPhone 12 Pro (3:4 portrait)"),
        (4032, 3024, "Horizontal iPhone 12 Pro (4:3 landscape)"),
        (1920, 1080, "Full HD 16:9 landscape"),
        (1080, 1920, "Full HD 9:16 portrait"),
        (2048, 2048, "Square 1:1"),
        (3000, 2000, "3:2 landscape (DSLR typical)"),
        (2000, 3000, "2:3 portrait"),
        (4000, 3000, "4:3 landscape"),
        (3000, 4000, "3:4 portrait"),
        (800, 600, "Small 4:3 image"),
        (6000, 4000, "Large 3:2 image"),
        (1200, 1500, "4:5 portrait (Instagram)"),
        (1500, 1200, "5:4 landscape"),
        (2100, 900, "Ultra-wide 21:9 (should NOT match 21:9)"),
        (900, 2100, "Ultra-tall (should match closest portrait)"),
    ]

    print("=" * 80)
    print("Testing choose_gemini_image_config")
    print("=" * 80)
    print()

    for width, height, description in test_cases:
        aspect_ratio, image_size = choose_gemini_image_config(width, height)
        input_ar = width / height

        # Get the actual output dimensions for verification
        output_dims = GEMINI_IMAGE_CONFIGS[aspect_ratio][image_size]
        output_ar = output_dims[0] / output_dims[1]

        print(f"Input:  {width}x{height} (AR={input_ar:.3f}) - {description}")
        print(f"Output: {aspect_ratio} @ {image_size} -> {output_dims[0]}x{output_dims[1]} (AR={output_ar:.3f})")
        print(f"AR diff: {abs(input_ar - output_ar):.4f}")
        print()

    # Verify 21:9 is never returned
    print("=" * 80)
    print("Verifying 21:9 is never returned...")
    print("=" * 80)

    # Test various ultra-wide ratios
    ultra_wide_tests = [
        (2100, 900),   # 2.33:1 (close to 21:9 = 2.33:1)
        (2520, 1080),  # Exactly 21:9
        (3360, 1440),  # 21:9 at larger size
    ]

    all_pass = True
    for width, height in ultra_wide_tests:
        aspect_ratio, _ = choose_gemini_image_config(width, height)
        if aspect_ratio == "21:9":
            print(f"FAIL: {width}x{height} returned 21:9!")
            all_pass = False
        else:
            print(f"PASS: {width}x{height} -> {aspect_ratio} (not 21:9)")

    print()
    if all_pass:
        print("All tests passed! 21:9 is correctly excluded.")
    else:
        print("SOME TESTS FAILED!")

    return all_pass


if __name__ == "__main__":
    test_choose_gemini_image_config()
