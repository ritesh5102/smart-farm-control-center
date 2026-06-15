import pickle
import json
import os
import numpy as np

# Absolute paths
base_dir = os.path.dirname(__file__)
crop_rec_path = os.path.join(base_dir, "crop_recommendation_model.pkl")
fert_path = os.path.join(base_dir, "fertilizer_recommendation_model.pkl")
yield_path = os.path.join(base_dir, "yield_prediction_model.pkl")
out_js_path = os.path.join(base_dir, "static", "models.js")

os.makedirs(os.path.dirname(out_js_path), exist_ok=True)

# Helper to recurse Scikit-Learn decision trees
def serialize_sk_tree(tree, node_idx, feature_names, class_names):
    if tree.children_left[node_idx] == -1: # Is leaf
        val = tree.value[node_idx][0]
        # Normalize to probabilities
        total = float(sum(val))
        prob_dict = {class_names[i]: float(val[i]) / total for i in range(len(class_names)) if val[i] > 0}
        return {"leaf": True, "value": prob_dict}
    
    feat_idx = tree.feature[node_idx]
    feat_name = feature_names[feat_idx]
    threshold = float(tree.threshold[node_idx])
    
    return {
        "leaf": False,
        "feature": feat_name,
        "threshold": threshold,
        "left": serialize_sk_tree(tree, tree.children_left[node_idx], feature_names, class_names),
        "right": serialize_sk_tree(tree, tree.children_right[node_idx], feature_names, class_names)
    }

def serialize_random_forest(rf_model, feature_names):
    class_names = [str(c) for c in rf_model.classes_]
    forest_js = []
    for estimator in rf_model.estimators_:
        tree_js = serialize_sk_tree(estimator.tree_, 0, feature_names, class_names)
        forest_js.append(tree_js)
    return forest_js

def main():
    js_content = "/* Smart Farm AI/ML Client-Side Engine */\n\n"
    
    # 1. Crop Recommendation Forest
    print("Exporting Crop Recommendation model...")
    with open(crop_rec_path, 'rb') as f:
        crop_rec_model = pickle.load(f)
    crop_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    crop_forest_js = serialize_random_forest(crop_rec_model, crop_features)
    js_content += f"const CROP_REC_FOREST = {json.dumps(crop_forest_js, indent=2)};\n\n"
    
    # 2. Fertilizer Recommendation Forest
    print("Exporting Fertilizer Recommendation model...")
    with open(fert_path, 'rb') as f:
        fert_model_package = pickle.load(f)
    
    fert_model = fert_model_package['model']
    fert_crop_encoder = fert_model_package['crop_encoder']
    fert_features = ['N', 'P', 'K', 'ph', 'moisture', 'temperature', 'crop']
    fert_forest_js = serialize_random_forest(fert_model, fert_features)
    
    # Crop label encoder categories
    crop_categories = {str(c): int(i) for i, c in enumerate(fert_crop_encoder.classes_)}
    js_content += f"const FERT_CROP_ENCODER = {json.dumps(crop_categories, indent=2)};\n"
    js_content += f"const FERT_REC_FOREST = {json.dumps(fert_forest_js, indent=2)};\n\n"
    
    # 3. Yield Prediction XGBoost
    print("Exporting Yield Prediction model...")
    with open(yield_path, 'rb') as f:
        yield_model_package = pickle.load(f)
        
    xgb_model = yield_model_package['model']
    yield_crop_encoder = yield_model_package['crop_encoder']
    yield_season_encoder = yield_model_package['season_encoder']
    
    # Get XGBoost trees as JSON dump
    booster = xgb_model.get_booster()
    xgb_trees_dump = booster.get_dump(dump_format='json')
    xgb_forest_js = [json.loads(tree_str) for tree_str in xgb_trees_dump]
    
    # Base score (default is 0.5)
    base_score = float(xgb_model.base_score) if xgb_model.base_score is not None else 0.5
    
    # Crop and Season label encoder categories
    yield_crops = {str(c): int(i) for i, c in enumerate(yield_crop_encoder.classes_)}
    yield_seasons = {str(c): int(i) for i, c in enumerate(yield_season_encoder.classes_)}
    
    js_content += f"const YIELD_CROP_ENCODER = {json.dumps(yield_crops, indent=2)};\n"
    js_content += f"const YIELD_SEASON_ENCODER = {json.dumps(yield_seasons, indent=2)};\n"
    js_content += f"const YIELD_BASE_SCORE = {base_score};\n"
    js_content += f"const YIELD_XGB_FOREST = {json.dumps(xgb_forest_js, indent=2)};\n\n"
    
    # 4. JavaScript Evaluator Functions
    evaluator_code = """
// 1. Evaluate Decision Tree
function evaluateSkTree(node, features) {
    if (node.leaf) return node.value;
    const val = features[node.feature];
    if (val <= node.threshold) {
        return evaluateSkTree(node.left, features);
    } else {
        return evaluateSkTree(node.right, features);
    }
}

// 2. Evaluate Random Forest (Average probabilities)
function evaluateSkForest(forest, features) {
    const votes = {};
    for (const tree of forest) {
        const leafVal = evaluateSkTree(tree, features);
        for (const [cls, prob] of Object.entries(leafVal)) {
            votes[cls] = (votes[cls] || 0) + prob / forest.length;
        }
    }
    return votes;
}

// 3. Evaluate XGBoost Tree
function evaluateXgbTree(node, features) {
    if (node.leaf !== undefined) return node.leaf;
    const val = features[node.split];
    // Find the child matching the split route yes/no
    const nextNodeId = val < node.split_condition ? node.yes : node.no;
    const nextNode = node.children.find(child => child.nodeid === nextNodeId);
    if (!nextNode) return 0.0;
    return evaluateXgbTree(nextNode, features);
}

// 4. Evaluate XGBoost Forest
function evaluateXgbForest(forest, features, baseScore = 0.5) {
    let score = baseScore;
    for (const tree of forest) {
        score += evaluateXgbTree(tree, features);
    }
    return score;
}

// ==============================================================================
// PUBLIC API WRAPPERS
// ==============================================================================

// A. Crop Recommendation
function recommendCropClient(N, P, K, temp, hum, ph, rain) {
    const features = { N, P, K, temperature: temp, humidity: hum, ph, rainfall: rain };
    const probs = evaluateSkForest(CROP_REC_FOREST, features);
    
    // Sort and get top 3
    const recommendations = Object.entries(probs)
        .map(([crop, prob]) => ({ crop: crop.charAt(0).toUpperCase() + crop.slice(1), probability: prob }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 3);
        
    return recommendations;
}

// B. Fertilizer Recommendation
function recommendFertilizerClient(N, P, K, ph, moisture, temp, cropName) {
    // Map crop string to integer
    const encodedCrop = FERT_CROP_ENCODER[cropName] !== undefined ? FERT_CROP_ENCODER[cropName] : 0;
    const features = { N, P, K, ph, moisture, temperature: temp, crop: encodedCrop };
    
    const probs = evaluateSkForest(FERT_REC_FOREST, features);
    
    // Get class with highest probability
    const topFertilizer = Object.entries(probs)
        .sort((a, b) => b[1] - a[1])[0][0];
        
    return topFertilizer;
}

// C. Yield Prediction
function predictYieldClient(cropName, seasonName, area, rainfall, fertilizer) {
    // Map crop & season to integers
    const encodedCrop = YIELD_CROP_ENCODER[cropName] !== undefined ? YIELD_CROP_ENCODER[cropName] : YIELD_CROP_ENCODER[Object.keys(YIELD_CROP_ENCODER)[0]];
    const encodedSeason = YIELD_SEASON_ENCODER[seasonName] !== undefined ? YIELD_SEASON_ENCODER[seasonName] : YIELD_SEASON_ENCODER[Object.keys(YIELD_SEASON_ENCODER)[0]];
    
    const features = {
        Crop: encodedCrop,
        Season: encodedSeason,
        Area: area,
        Rainfall: rainfall,
        Fertilizer: fertilizer
    };
    
    let predYield = evaluateXgbForest(YIELD_XGB_FOREST, features, YIELD_BASE_SCORE);
    predYield = Math.max(0.0, predYield);
    const totalProduction = predYield * area;
    
    return {
        predicted_yield: predYield,
        total_production: totalProduction
    };
}
"""
    js_content += evaluator_code
    
    with open(out_js_path, 'w') as f:
        f.write(js_content)
        
    print(f"Successfully compiled all ML models to JavaScript at {out_js_path}")

if __name__ == "__main__":
    main()
