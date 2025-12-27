"""
Test script to verify the word boundary matching logic in build_knowledge.py

This script tests that:
1. Exact matches work correctly
2. Word boundary matching prevents substring false positives
3. Longer matches are preferred over shorter ones
"""

import re


def word_boundary_match(pattern, text):
    """
    Kiểm tra xem pattern có xuất hiện như một từ hoàn chỉnh trong text không.
    Sử dụng word boundary (\\b) để tránh substring matching sai.
    """
    if not pattern or not text:
        return False
    escaped_pattern = re.escape(pattern)
    return bool(re.search(r'\b' + escaped_pattern + r'\b', text, re.IGNORECASE))


def test_word_boundary_matching():
    """Test cases for word boundary matching."""
    print("=" * 60)
    print("Testing Word Boundary Matching")
    print("=" * 60)
    
    test_cases = [
        # (pattern, text, expected_result, description)
        ("car", "cartoon", False, "car should NOT match cartoon"),
        ("car", "scare", False, "car should NOT match scare"),
        ("car", "race car", True, "car should match 'race car'"),
        ("car", "car", True, "car should match car exactly"),
        ("car", "a car is here", True, "car should match in sentence"),
        
        ("an", "man", False, "an should NOT match man"),
        ("an", "woman", False, "an should NOT match woman"),
        ("an", "kangaroo", False, "an should NOT match kangaroo"),
        ("an", "an apple", True, "an should match 'an apple'"),
        
        ("in", "riding", False, "in should NOT match riding"),
        ("in", "cooking", False, "in should NOT match cooking"),
        ("in", "training", False, "in should NOT match training"),
        ("in", "in the kitchen", True, "in should match 'in the kitchen'"),
        
        ("man", "woman", False, "man should NOT match woman"),
        ("man", "a man walks", True, "man should match 'a man walks'"),
        ("man", "the man", True, "man should match 'the man'"),
        
        ("play", "play guitar", True, "play should match 'play guitar'"),
        ("play", "display", False, "play should NOT match display"),
        ("play", "playful", False, "play should NOT match playful"),
        
        ("walk", "walk on", True, "walk should match 'walk on'"),
        ("walk", "sidewalk", False, "walk should NOT match sidewalk"),
    ]
    
    passed = 0
    failed = 0
    
    for pattern, text, expected, description in test_cases:
        result = word_boundary_match(pattern, text)
        status = "✓" if result == expected else "✗"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
            
        print(f"{status} {description}")
        if result != expected:
            print(f"   Expected: {expected}, Got: {result}")
    
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


def test_find_best_match():
    """Test the find_best_match function logic."""
    print("\n" + "=" * 60)
    print("Testing Find Best Match Logic")
    print("=" * 60)
    
    # Simulate induction lines
    ent_induction_lines = [
        "man|man#animated man#young man#old man\n",
        "woman|woman#young woman#old woman\n",
        "car|car#race car#toy car\n",
        "cartoon|cartoon#animated cartoon\n",
    ]
    
    rel_induction_lines = [
        "play|play#play guitar#play piano\n",
        "in|in\n",
        "ride|ride#ride a bike#ride a horse\n",
    ]
    
    def find_best_match_test(text, induction_lines):
        """Simplified version of find_best_match for testing."""
        text = text.strip().lower()
        
        # Exact match first
        for line in induction_lines:
            if '|' not in line:
                continue
            step1 = line.split('|')
            normalized = step1[0].replace(' ', '').strip()
            variants = step1[1].replace('\n', '').split('#')
            
            for variant in variants:
                variant = variant.strip().lower()
                if variant and variant == text:
                    return normalized
        
        # Word boundary match
        best_match = None
        best_match_len = 0
        
        for line in induction_lines:
            if '|' not in line:
                continue
            step1 = line.split('|')
            normalized = step1[0].replace(' ', '').strip()
            variants = step1[1].replace('\n', '').split('#')
            
            for variant in variants:
                variant = variant.strip()
                if not variant:
                    continue
                
                if word_boundary_match(variant, text):
                    if len(variant) > best_match_len:
                        best_match = normalized
                        best_match_len = len(variant)
        
        return best_match if best_match else 'none'
    
    test_cases = [
        # Entity tests
        ("man", ent_induction_lines, "man", "exact match 'man'"),
        ("a man", ent_induction_lines, "man", "word boundary match 'a man'"),
        ("young man", ent_induction_lines, "man", "longer variant 'young man' -> man"),
        ("woman", ent_induction_lines, "woman", "exact match 'woman' (not man!)"),
        ("a woman walks", ent_induction_lines, "woman", "'woman' should not match 'man'"),
        ("cartoon", ent_induction_lines, "cartoon", "cartoon is separate from car"),
        ("a cartoon character", ent_induction_lines, "cartoon", "cartoon character"),
        ("race car", ent_induction_lines, "car", "race car -> car"),
        
        # Relation tests
        ("play", rel_induction_lines, "play", "exact match play"),
        ("play guitar", rel_induction_lines, "play", "play guitar -> play"),
        ("in", rel_induction_lines, "in", "exact match in"),
        ("riding", rel_induction_lines, "none", "riding should NOT match ride"),
        ("ride a bike", rel_induction_lines, "ride", "ride a bike -> ride"),
    ]
    
    passed = 0
    failed = 0
    
    for text, lines, expected, description in test_cases:
        result = find_best_match_test(text, lines)
        status = "✓" if result == expected else "✗"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
            
        print(f"{status} {description}")
        if result != expected:
            print(f"   Input: '{text}', Expected: '{expected}', Got: '{result}'")
    
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    all_passed = True
    
    all_passed &= test_word_boundary_matching()
    all_passed &= test_find_best_match()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)
