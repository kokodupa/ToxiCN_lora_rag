import json
from difflib import SequenceMatcher
from collections import defaultdict
from typing import Any
import json
import difflib
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import re


# Calculate precision, recall, and F1 score
def quadruple_hard_match(predictions):
    """Calculate precision, recall, and F1 score for model predictions"""
    
    true_positives = 0  # True positives: Number of correctly predicted quadruples
    predicted_positives = 0  # Predicted positives: Total number of quadruples predicted by the model
    actual_positives = 0  # Actual positives: Total number of quadruples in the ground truth

    for item in predictions:
        # Split target and predicted quadruples
        target_quadruples = item["ground_truth"].split("[SEP]")
        prediction_quadruples = item["output"].split("[SEP]")

        target_quadruples = [t.replace('[END]', '').strip() for t in target_quadruples if t.strip()]
        prediction_quadruples = [p.replace('[END]', '').strip() for p in prediction_quadruples if p.strip()]
        
        # Remove trailing whitespace from each quadruple to avoid comparison issues
        target_quadruples = [t.strip() for t in target_quadruples if t.strip()]
        prediction_quadruples = [p.strip() for p in prediction_quadruples if p.strip()]
        
        actual_positives += len(target_quadruples)  # Update total actual positives
        predicted_positives += len(prediction_quadruples)  # Update total predicted positives

        # Compare each predicted quadruple with target quadruples
        for pred in prediction_quadruples:
            for t in target_quadruples:
                if pred == t:  # If predicted quadruple exactly matches target quadruple
                    true_positives += 1
                    target_quadruples.remove(t)  # Ensure each target quadruple is matched only once
                    break  # Exit inner loop once a match is found
        
    # Calculate precision
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    
    # Calculate recall
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

# Quadruple Soft Matching
def quadruple_soft_match(predictions):
    """Calculate precision, recall, and F1 score for model predictions (soft matching)"""
    
    true_positives = 0  # True positives: Number of correctly predicted quadruples
    predicted_positives = 0  # Predicted positives: Total number of quadruples predicted by the model
    actual_positives = 0  # Actual positives: Total number of quadruples in the ground truth

    for item in predictions:
        # Split target and predicted quadruples
        target_quadruples = item["ground_truth"].split("[SEP]")
        prediction_quadruples = item["output"].split("[SEP]")

        target_quadruples = [t.replace('[END]', '').strip() for t in target_quadruples if t.strip()]
        prediction_quadruples = [p.replace('[END]', '').strip() for p in prediction_quadruples if p.strip()]
        
        # Remove trailing whitespace from each quadruple to avoid comparison issues
        target_quadruples = [t.strip() for t in target_quadruples if t.strip()]
        prediction_quadruples = [p.strip() for p in prediction_quadruples if p.strip()]
        
        actual_positives += len(target_quadruples)  # Update total actual positives
        predicted_positives += len(prediction_quadruples)  # Update total predicted positives
        
        # Compare each predicted quadruple with target quadruples
        for pred in prediction_quadruples:
            pred_parts = pred.split("|")
            target_parts = None
            
            matched = False  # Flag to indicate if a matching target quadruple is found
            
            # Find matching target quadruple
            for t in target_quadruples:
                target_parts = t.split("|")
                
                if len(target_parts) == 4 and len(pred_parts) == 4:
                    target_entity, target_speech, target_group, target_hate = target_parts
                    pred_entity, pred_speech, pred_group, pred_hate = pred_parts
                    
                    # Use soft matching for the first two components
                    similarity = SequenceMatcher(None, target_entity, pred_entity).ratio()
                    similarity_speech = SequenceMatcher(None, target_speech, pred_speech).ratio()
                    
                    # If similarity for first two components is >= 0.5 and last two components match exactly
                    if similarity >= 0.5 and similarity_speech >= 0.5 and target_group == pred_group and target_hate == pred_hate:
                        true_positives += 1
                        target_quadruples.remove(t)  # Ensure each target quadruple is matched only once
                        matched = True  # Mark as matched
                        break  # Exit inner loop and move to next predicted quadruple
        
    # Calculate precision
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    
    # Calculate recall
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score


# Triple Hard Matching
def triple_hard_match(predictions):
    """Calculate precision, recall, and F1 score for model predictions"""
    
    true_positives = 0  # True positives: Number of correctly predicted quadruples
    predicted_positives = 0  # Predicted positives: Total number of quadruples predicted by the model
    actual_positives = 0  # Actual positives: Total number of quadruples in the ground truth

    for item in predictions:
        # Split target and predicted quadruples
        target_quadruples = item["ground_truth"].split("[SEP]")
        prediction_quadruples = item["output"].split("[SEP]")

        target_quadruples = [t.replace('[END]', '').strip() for t in target_quadruples if t.strip()]
        prediction_quadruples = [p.replace('[END]', '').strip() for p in prediction_quadruples if p.strip()]
        
        # Remove trailing whitespace from each quadruple to avoid comparison issues
        target_quadruples = [t.strip() for t in target_quadruples if t.strip()]
        prediction_quadruples = [p.strip() for p in prediction_quadruples if p.strip()]
        
        actual_positives += len(target_quadruples)  # Update total actual positives
        predicted_positives += len(prediction_quadruples)  # Update total predicted positives

        # Compare each predicted quadruple with target quadruples
        for pred in prediction_quadruples:
            pred_parts = pred.split("|")
            
            for t in target_quadruples:
                target_parts = t.split("|")
                
                # Ensure the number of elements in quadruples is correct, then compare the first two and last elements
                if len(target_parts) == 4 and len(pred_parts) == 4:
                    target_entity, target_speech, _, target_hate = target_parts
                    pred_entity, pred_speech, _, pred_hate = pred_parts
                    
                    # If the first two elements and the last element are identical, consider it a match
                    if (target_entity == pred_entity and 
                        target_speech == pred_speech and
                        target_hate == pred_hate):
                        
                        true_positives += 1
                        target_quadruples.remove(t)  # Ensure each target quadruple is matched only once
                        break  # Exit inner loop once a match is found
        
    # Calculate precision
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    
    # Calculate recall
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score


# Triple Soft Matching
def triple_soft_match(predictions):
    """Calculate precision, recall, and F1 score for model predictions (soft matching)"""
    
    true_positives = 0  # True positives: Number of correctly predicted quadruples
    predicted_positives = 0  # Predicted positives: Total number of quadruples predicted by the model
    actual_positives = 0  # Actual positives: Total number of quadruples in the ground truth

    for item in predictions:
        # Split target and predicted quadruples
        target_quadruples = item["ground_truth"].split("[SEP]")
        prediction_quadruples = item["output"].split("[SEP]")

        target_quadruples = [t.replace('[END]', '').strip() for t in target_quadruples if t.strip()]
        prediction_quadruples = [p.replace('[END]', '').strip() for p in prediction_quadruples if p.strip()]
        
        # Remove trailing whitespace from each quadruple to avoid comparison issues
        target_quadruples = [t.strip() for t in target_quadruples if t.strip()]
        prediction_quadruples = [p.strip() for p in prediction_quadruples if p.strip()]
        
        actual_positives += len(target_quadruples)  # Update total actual positives
        predicted_positives += len(prediction_quadruples)  # Update total predicted positives
        
        # Compare each predicted quadruple with target quadruples
        for pred in prediction_quadruples:
            pred_parts = pred.split("|")
            target_parts = None
            
            matched = False  # Flag to indicate if a matching target quadruple is found
            
            # Find matching target quadruple
            for t in target_quadruples:
                target_parts = t.split("|")
                
                # If the first two elements have similarity > 0.5 and the last element matches exactly, consider it a match
                if len(target_parts) == 4 and len(pred_parts) == 4:
                    target_entity, target_speech, _, target_hate = target_parts  # Ignore the third element
                    pred_entity, pred_speech, _, pred_hate = pred_parts  # Ignore the third element
                    
                    # Use soft matching for the first two elements
                    similarity = SequenceMatcher(None, target_entity, pred_entity).ratio()
                    similarity_speech = SequenceMatcher(None, target_speech, pred_speech).ratio()
                    
                    # If the first two elements have similarity > 0.5 and the last element matches exactly
                    if similarity >= 0.5 and similarity_speech >= 0.5 and target_hate == pred_hate:
                        true_positives += 1
                        target_quadruples.remove(t)  # Ensure each target quadruple is matched only once
                        matched = True  # Mark as matched
                        break  # Exit inner loop and move to next predicted quadruple
        
    # Calculate precision
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    
    # Calculate recall
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score




# Pair Hard Matching
def pair_hard_match(predictions):
    """Calculate precision, recall, and F1 score for model predictions"""
    
    true_positives = 0  # True positives: Number of correctly predicted pairs
    predicted_positives = 0  # Predicted positives: Total number of pairs predicted by the model
    actual_positives = 0  # Actual positives: Total number of pairs in the ground truth

    for item in predictions:
        # Split target and predicted quadruples
        target_quadruples = item["ground_truth"].split("[SEP]")
        prediction_quadruples = item["output"].split("[SEP]")

        target_quadruples = [t.replace('[END]', '').strip() for t in target_quadruples if t.strip()]
        prediction_quadruples = [p.replace('[END]', '').strip() for p in prediction_quadruples if p.strip()]
        
        # Remove trailing whitespace from each quadruple to avoid comparison issues
        target_quadruples = [t.strip() for t in target_quadruples if t.strip()]
        prediction_quadruples = [p.strip() for p in prediction_quadruples if p.strip()]
        
        actual_positives += len(target_quadruples)  # Update total actual positives
        predicted_positives += len(prediction_quadruples)  # Update total predicted positives

        # Compare each predicted quadruple with target quadruples
        for pred in prediction_quadruples:
            pred_parts = pred.split("|")
            
            for t in target_quadruples:
                target_parts = t.split("|")
                
                # Ensure the number of elements in quadruples is correct, then compare the first two elements
                if len(target_parts) == 4 and len(pred_parts) == 4:
                    target_entity, target_speech, _, _ = target_parts
                    pred_entity, pred_speech, _, _ = pred_parts
                    
                    # If the first two elements are identical, consider it a match
                    if (target_entity == pred_entity and 
                        target_speech == pred_speech):
                        
                        true_positives += 1
                        target_quadruples.remove(t)  # Ensure each target quadruple is matched only once
                        break  # Exit inner loop once a matching pair is found
        
    # Calculate precision
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    
    # Calculate recall
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

# Pair Soft Matching
def pair_soft_match(predictions):
    """Calculate precision, recall, and F1 score for model predictions (soft matching)"""
    
    true_positives = 0  # True positives: Number of correctly predicted doublets
    predicted_positives = 0  # Predicted positives: Total number of doublets predicted by the model
    actual_positives = 0  # Actual positives: Total number of doublets in the ground truth

    for item in predictions:
        # Split target and predicted quadruples
        target_quadruples = item["ground_truth"].split("[SEP]")
        prediction_quadruples = item["output"].split("[SEP]")

        target_quadruples = [t.replace('[END]', '').strip() for t in target_quadruples if t.strip()]
        prediction_quadruples = [p.replace('[END]', '').strip() for p in prediction_quadruples if p.strip()]
        
        # Remove trailing whitespace from each quadruple to avoid comparison issues
        target_quadruples = [t.strip() for t in target_quadruples if t.strip()]
        prediction_quadruples = [p.strip() for p in prediction_quadruples if p.strip()]
        
        actual_positives += len(target_quadruples)  # Update total actual positives
        predicted_positives += len(prediction_quadruples)  # Update total predicted positives
        
        # Compare each predicted quadruple with target quadruples
        for pred in prediction_quadruples:
            pred_parts = pred.split("|")
            target_parts = None
            
            matched = False  # Flag to indicate if a matching target quadruple is found
            
            # Find matching target quadruple
            for t in target_quadruples:
                target_parts = t.split("|")
                
                # If the similarity of the first two elements exceeds 0.5, consider it a match
                if len(target_parts) == 4 and len(pred_parts) == 4:
                    target_entity, target_speech, _, _ = target_parts  # Ignore the last two elements
                    pred_entity, pred_speech, _, _ = pred_parts  # Ignore the last two elements
                    
                    # Use soft matching for the first two elements
                    similarity = SequenceMatcher(None, target_entity, pred_entity).ratio()
                    similarity_speech = SequenceMatcher(None, target_speech, pred_speech).ratio()
                    
                    # If the similarity of the first two elements is greater than 0.5, consider it a match
                    if similarity >= 0.5 and similarity_speech >= 0.5:
                        true_positives += 1
                        target_quadruples.remove(t)  # Ensure each target quadruple is matched only once
                        matched = True  # Mark as matched
                        break  # Exit inner loop and move to next predicted quadruple
        
    # Calculate precision
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    
    # Calculate recall
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

# Single Element Hard Matching
def single_hard_match(predictions):
    """Calculate precision, recall, and F1 score for model predictions (hard matching)"""
    
    true_positives = 0  # True positives: Number of correctly predicted elements
    predicted_positives = 0  # Predicted positives: Total number of elements predicted by the model
    actual_positives = 0  # Actual positives: Total number of elements in the ground truth

    for item in predictions:
        # Split target and predicted quadruples
        target_quadruples = item["ground_truth"].split("[SEP]")
        prediction_quadruples = item["output"].split("[SEP]")

        target_quadruples = [t.replace('[END]', '').strip() for t in target_quadruples if t.strip()]
        prediction_quadruples = [p.replace('[END]', '').strip() for p in prediction_quadruples if p.strip()]
        
        # Remove trailing whitespace from each quadruple to avoid comparison issues
        target_quadruples = [t.strip() for t in target_quadruples if t.strip()]
        prediction_quadruples = [p.strip() for p in prediction_quadruples if p.strip()]
        
        actual_positives += len(target_quadruples)  # Update total actual positives
        predicted_positives += len(prediction_quadruples)  # Update total predicted positives
        
        # Compare each predicted quadruple with target quadruples
        for pred in prediction_quadruples:
            pred_parts = pred.split("|")
            
            # Find matching target quadruple
            for t in target_quadruples:
                target_parts = t.split("|")
                
                # If the second element of the predicted and target quadruples exactly match, consider it a match
                if len(target_parts) == 4 and len(pred_parts) == 4:
                    _, target_speech, _, _ = target_parts  
                    _, pred_speech, _, _ = pred_parts 

                    # If the second element exactly matches
                    if target_speech == pred_speech:
                        true_positives += 1
                        target_quadruples.remove(t)  # Ensure each target quadruple is matched only once
                        break  # Exit inner loop and move to next predicted quadruple
        
    # Calculate precision
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    
    # Calculate recall
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score


# Single Element Soft Matching
def single_soft_match(predictions):
    """Calculate precision, recall, and F1 score for model predictions (soft matching)"""
    
    true_positives = 0  # True positives: Number of correctly predicted elements
    predicted_positives = 0  # Predicted positives: Total number of elements predicted by the model
    actual_positives = 0  # Actual positives: Total number of elements in the ground truth

    for item in predictions:
        # Split target and predicted quadruples
        target_quadruples = item["ground_truth"].split("[SEP]")
        prediction_quadruples = item["output"].split("[SEP]")

        target_quadruples = [t.replace('[END]', '').strip() for t in target_quadruples if t.strip()]
        prediction_quadruples = [p.replace('[END]', '').strip() for p in prediction_quadruples if p.strip()]
        
        # Remove trailing whitespace from each quadruple to avoid comparison issues
        target_quadruples = [t.strip() for t in target_quadruples if t.strip()]
        prediction_quadruples = [p.strip() for p in prediction_quadruples if p.strip()]
        
        actual_positives += len(target_quadruples)  # Update total actual positives
        predicted_positives += len(prediction_quadruples)  # Update total predicted positives
        
        # Compare each predicted quadruple with target quadruples
        for pred in prediction_quadruples:
            pred_parts = pred.split("|")
            target_parts = None
            
            matched = False  # Flag to indicate if a matching target quadruple is found
            
            # Find matching target quadruple
            for t in target_quadruples:
                target_parts = t.split("|")
                
                # If the similarity of the first element exceeds 0.5, consider it a match
                if len(target_parts) == 4 and len(pred_parts) == 4:
                    _, target_speech, _, _ = target_parts  
                    _, pred_speech, _, _ = pred_parts 
                    
                    # Use soft matching for the first element
                    similarity = SequenceMatcher(None, target_speech, pred_speech).ratio()
                    # If the similarity of the first element is greater than 0.5, consider it a match
                    if similarity >= 0.5:
                        true_positives += 1
                        target_quadruples.remove(t)  # Ensure each target quadruple is matched only once
                        matched = True  # Mark as matched
                        break  # Exit inner loop and move to next predicted quadruple
        
    # Calculate precision
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    
    # Calculate recall
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score


# Load data
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Main function
def main(input_json_path):
    # 1. Load data
    data = load_data(input_json_path)

    # 2. Calculate metrics
    result_score ={}
    result_score["quadruple_hard_match"] = quadruple_hard_match(data)
    result_score["quadruple_soft_match"] = quadruple_soft_match(data)
    result_score["triple_hard_match"] = triple_hard_match(data)
    result_score["triple_soft_match"] = triple_soft_match(data)
    result_score['pair_hard_match'] = pair_hard_match(data)
    result_score['pair_soft_match'] = pair_soft_match(data)

    result_score["single_element_hard_match"] = single_hard_match(data)
    result_score["single_element_soft_match"] = single_soft_match(data)
    print("四元组硬匹配结果：", result_score["quadruple_hard_match"])
    print("四元组软匹配结果：", result_score["quadruple_soft_match"])
    print("三元组硬匹配结果：", result_score["triple_hard_match"])
    print("三元组软匹配结果：", result_score["triple_soft_match"])
    print("二元组硬匹配结果：", result_score['pair_hard_match'])
    print("二元组软匹配结果：", result_score['pair_soft_match'])

    print("单元素硬匹配结果：", result_score["single_element_hard_match"])
    print("单元素软匹配结果：", result_score["single_element_soft_match"])
    return result_score

    

